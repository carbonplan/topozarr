//! Native zarr write path: encode + store regions via zarrs, bypassing
//! zarr-python's sync bridge. Arrays must already exist (metadata is
//! created by zarr-python); regions must be shard-aligned.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use numpy::{
    PyArrayDescrMethods, PyArrayDyn, PyArrayMethods, PyUntypedArray, PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use zarrs::array::{Array, ArraySubset, CodecOptions};
use zarrs::filesystem::FilesystemStore;
use zarrs::storage::byte_range::ByteRangeIterator;
use zarrs::storage::storage_adapter::async_to_sync::{
    AsyncToSyncBlockOn, AsyncToSyncStorageAdapter,
};
use zarrs::storage::{
    AsyncReadableWritableListableStorage, Bytes, ListableStorageTraits, MaybeBytesIterator,
    OffsetBytesIterator, ReadableStorageTraits, ReadableWritableListableStorage,
    ReadableWritableListableStorageTraits, StorageError, StoreKey, StoreKeys, StoreKeysPrefixes,
    StorePrefix, WritableStorageTraits,
};
use zarrs_object_store::AsyncObjectStore;

type ZarrsArray = Array<dyn ReadableWritableListableStorageTraits>;

struct TokioBlockOn(tokio::runtime::Handle);

impl AsyncToSyncBlockOn for TokioBlockOn {
    fn block_on<F: core::future::Future>(&self, future: F) -> F::Output {
        self.0.block_on(future)
    }
}

fn err<E: std::fmt::Display>(e: E) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

/// Cumulative store-level PUT counters. Seconds are summed across threads,
/// so totals can exceed wall time under concurrency.
#[derive(Default)]
struct IoStats {
    put_nanos: AtomicU64,
    put_bytes: AtomicU64,
    put_ops: AtomicU64,
}

/// Max PUTs in flight on the async runtime; bounds memory (each holds its
/// encoded shard bytes) and connection pressure. `set` blocks a worker thread
/// once this many uploads are outstanding (backpressure).
const MAX_INFLIGHT_PUTS: usize = 64;

/// Decouples encode (CPU, on the caller's worker thread) from upload (network)
/// for S3: `spawn_set` fires the PUT onto the tokio runtime and returns, so the
/// worker thread keeps encoding while uploads overlap. `flush` awaits them.
struct PutQueue {
    async_store: AsyncReadableWritableListableStorage,
    rt: tokio::runtime::Handle,
    sem: Arc<tokio::sync::Semaphore>,
    pending: Mutex<Vec<tokio::task::JoinHandle<Result<(), String>>>>,
    stats: Arc<IoStats>,
}

impl PutQueue {
    fn spawn_set(&self, key: StoreKey, value: Bytes, nbytes: u64) {
        let store = self.async_store.clone();
        let sem = self.sem.clone();
        let stats = self.stats.clone();
        // backpressure: park the worker thread until an in-flight slot frees
        let permit = self
            .rt
            .block_on(sem.acquire_owned())
            .expect("put semaphore closed");
        let handle = self.rt.spawn(async move {
            let _permit = permit;
            let t0 = Instant::now();
            let r = store.set(&key, value).await.map_err(|e| e.to_string());
            stats
                .put_nanos
                .fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
            stats.put_bytes.fetch_add(nbytes, Ordering::Relaxed);
            stats.put_ops.fetch_add(1, Ordering::Relaxed);
            r
        });
        self.pending.lock().unwrap().push(handle);
    }

    /// Await every outstanding PUT, propagating the first error.
    fn flush(&self) -> Result<(), String> {
        let handles: Vec<_> = self.pending.lock().unwrap().drain(..).collect();
        self.rt.block_on(async {
            for h in handles {
                match h.await {
                    Ok(Ok(())) => {}
                    Ok(Err(e)) => return Err(e),
                    Err(_) => return Err("put task panicked".to_string()),
                }
            }
            Ok(())
        })
    }
}

/// Delegating storage wrapper. With `put_queue` (S3), `set` hands the PUT to
/// the async queue and returns immediately; otherwise (filesystem) it writes
/// synchronously and times the call so encode cost splits from upload cost.
struct TimingStorage {
    inner: ReadableWritableListableStorage,
    stats: Arc<IoStats>,
    put_queue: Option<Arc<PutQueue>>,
}

impl ReadableStorageTraits for TimingStorage {
    fn get_partial_many<'a>(
        &'a self,
        key: &StoreKey,
        byte_ranges: ByteRangeIterator<'a>,
    ) -> Result<MaybeBytesIterator<'a>, StorageError> {
        self.inner.get_partial_many(key, byte_ranges)
    }

    fn size_key(&self, key: &StoreKey) -> Result<Option<u64>, StorageError> {
        self.inner.size_key(key)
    }

    fn supports_get_partial(&self) -> bool {
        self.inner.supports_get_partial()
    }
}

impl WritableStorageTraits for TimingStorage {
    fn set(&self, key: &StoreKey, value: Bytes) -> Result<(), StorageError> {
        let nbytes = value.len() as u64;
        if let Some(q) = &self.put_queue {
            // async path: enqueue the upload and return so the worker thread
            // resumes encoding; errors surface in flush()
            q.spawn_set(key.clone(), value, nbytes);
            return Ok(());
        }
        let t0 = Instant::now();
        let result = self.inner.set(key, value);
        self.stats
            .put_nanos
            .fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
        self.stats.put_bytes.fetch_add(nbytes, Ordering::Relaxed);
        self.stats.put_ops.fetch_add(1, Ordering::Relaxed);
        result
    }

    fn set_partial_many(
        &self,
        key: &StoreKey,
        offset_values: OffsetBytesIterator,
    ) -> Result<(), StorageError> {
        self.inner.set_partial_many(key, offset_values)
    }

    fn erase(&self, key: &StoreKey) -> Result<(), StorageError> {
        self.inner.erase(key)
    }

    fn erase_prefix(&self, prefix: &StorePrefix) -> Result<(), StorageError> {
        self.inner.erase_prefix(prefix)
    }

    fn supports_set_partial(&self) -> bool {
        self.inner.supports_set_partial()
    }
}

impl ListableStorageTraits for TimingStorage {
    fn list(&self) -> Result<StoreKeys, StorageError> {
        self.inner.list()
    }

    fn list_prefix(&self, prefix: &StorePrefix) -> Result<StoreKeys, StorageError> {
        self.inner.list_prefix(prefix)
    }

    fn list_dir(&self, prefix: &StorePrefix) -> Result<StoreKeysPrefixes, StorageError> {
        self.inner.list_dir(prefix)
    }

    fn size_prefix(&self, prefix: &StorePrefix) -> Result<u64, StorageError> {
        self.inner.size_prefix(prefix)
    }
}

/// Build an S3-backed sync storage from an `s3://bucket/prefix` URL.
/// Credentials/region resolve through the object_store AWS provider chain
/// (env vars, profile, IMDS) plus any `options` overrides.
fn s3_storage(
    url: &str,
    options: &HashMap<String, String>,
    handle: tokio::runtime::Handle,
) -> PyResult<(
    ReadableWritableListableStorage,
    AsyncReadableWritableListableStorage,
)> {
    let rest = url.strip_prefix("s3://").expect("checked by caller");
    let (bucket, prefix) = match rest.split_once('/') {
        Some((b, p)) => (b, p.trim_end_matches('/')),
        None => (rest, ""),
    };
    if bucket.is_empty() {
        return Err(PyValueError::new_err(format!("no bucket in url {url:?}")));
    }

    let mut builder = object_store::aws::AmazonS3Builder::from_env().with_bucket_name(bucket);
    // generous network timeouts by default; the 5s object_store connect
    // default aborts under request pressure (see docs/usage.md)
    let mut opts: HashMap<&str, &str> =
        HashMap::from([("connect_timeout", "30s"), ("timeout", "120s")]);
    for (k, v) in options {
        opts.insert(k.as_str(), v.as_str());
    }
    for (k, v) in opts {
        let key = k
            .parse::<object_store::aws::AmazonS3ConfigKey>()
            .map_err(|e| PyValueError::new_err(format!("bad option {k:?}: {e}")))?;
        builder = builder.with_config(key, v);
    }
    let s3 = builder.build().map_err(err)?;

    let async_store: AsyncReadableWritableListableStorage = if prefix.is_empty() {
        Arc::new(AsyncObjectStore::new(s3))
    } else {
        Arc::new(AsyncObjectStore::new(
            object_store::prefix::PrefixStore::new(s3, prefix),
        ))
    };
    // sync adapter handles array-open reads; writes go async via PutQueue
    let sync: ReadableWritableListableStorage = Arc::new(AsyncToSyncStorageAdapter::new(
        async_store.clone(),
        TokioBlockOn(handle),
    ));
    Ok((sync, async_store))
}

/// Writes shard-aligned regions of existing zarr v3 arrays directly from
/// Rust: one shared store (connection pool included), zarrs codecs, no
/// per-region trip through zarr-python.
///
/// `write_region` encodes on the caller's worker thread (GIL released); for
/// S3 the resulting PUTs are uploaded asynchronously on the runtime so encode
/// and upload overlap. `flush()` awaits the outstanding PUTs.
#[pyclass]
pub struct RustWriter {
    storage: ReadableWritableListableStorage,
    arrays: Mutex<HashMap<String, Arc<ZarrsArray>>>,
    io_stats: Arc<IoStats>,
    // write_region totals (encode only on S3; encode+store on filesystem),
    // summed across caller threads
    write_nanos: AtomicU64,
    write_bytes: AtomicU64,
    write_ops: AtomicU64,
    // async upload queue for S3; None for filesystem stores
    put_queue: Option<Arc<PutQueue>>,
    // keeps the S3 block_on runtime alive; None for filesystem stores
    _runtime: Option<tokio::runtime::Runtime>,
}

impl RustWriter {
    fn array(&self, py: Python<'_>, path: &str) -> PyResult<Arc<ZarrsArray>> {
        if let Some(a) = self.arrays.lock().unwrap().get(path) {
            return Ok(a.clone());
        }
        let storage = self.storage.clone();
        let owned = path.to_string();
        let array = py
            .detach(move || ZarrsArray::open(storage, &owned))
            .map_err(err)?;
        let array = Arc::new(array);
        self.arrays
            .lock()
            .unwrap()
            .insert(path.to_string(), array.clone());
        Ok(array)
    }
}

#[pymethods]
impl RustWriter {
    /// `url` is a local directory path or `s3://bucket/prefix`. `options`
    /// are object_store S3/client config keys (e.g. `region`,
    /// `connect_timeout`); they override the built-in timeout defaults.
    #[new]
    #[pyo3(signature = (url, options=None))]
    fn new(url: &str, options: Option<HashMap<String, String>>) -> PyResult<Self> {
        let options = options.unwrap_or_default();
        let io_stats = Arc::new(IoStats::default());
        let (storage, put_queue, runtime) = if url.starts_with("s3://") {
            let runtime = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .map_err(err)?;
            let (sync, async_store) = s3_storage(url, &options, runtime.handle().clone())?;
            let queue = Arc::new(PutQueue {
                async_store,
                rt: runtime.handle().clone(),
                sem: Arc::new(tokio::sync::Semaphore::new(MAX_INFLIGHT_PUTS)),
                pending: Mutex::new(Vec::new()),
                stats: io_stats.clone(),
            });
            (sync, Some(queue), Some(runtime))
        } else {
            if !options.is_empty() {
                return Err(PyValueError::new_err(
                    "options are only supported for s3:// urls",
                ));
            }
            let store = FilesystemStore::new(url).map_err(err)?;
            (
                Arc::new(store) as ReadableWritableListableStorage,
                None,
                None,
            )
        };
        let storage: ReadableWritableListableStorage = Arc::new(TimingStorage {
            inner: storage,
            stats: io_stats.clone(),
            put_queue: put_queue.clone(),
        });
        Ok(Self {
            storage,
            arrays: Mutex::new(HashMap::new()),
            io_stats,
            write_nanos: AtomicU64::new(0),
            write_bytes: AtomicU64::new(0),
            write_ops: AtomicU64::new(0),
            put_queue,
            _runtime: runtime,
        })
    }

    /// Cumulative timing counters. `write_s`/`write_bytes`/`write_ops` cover
    /// worker-thread time in `write_region`: encode only on S3 (PUTs run async
    /// on the runtime), encode + store on filesystem. `put_s`/`put_bytes`/
    /// `put_ops` cover the uploads; on S3 these overlap encode and each other,
    /// so `put_s` is wall-independent and not subtracted from `write_s`.
    /// Seconds are summed across threads/tasks. Valid only after `flush()`.
    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let secs = |a: &AtomicU64| a.load(Ordering::Relaxed) as f64 / 1e9;
        let d = PyDict::new(py);
        d.set_item("write_s", secs(&self.write_nanos))?;
        d.set_item("write_bytes", self.write_bytes.load(Ordering::Relaxed))?;
        d.set_item("write_ops", self.write_ops.load(Ordering::Relaxed))?;
        d.set_item("put_s", secs(&self.io_stats.put_nanos))?;
        d.set_item("put_bytes", self.io_stats.put_bytes.load(Ordering::Relaxed))?;
        d.set_item("put_ops", self.io_stats.put_ops.load(Ordering::Relaxed))?;
        Ok(d)
    }

    /// Write `block` at `start` (per-axis offsets) of the array at `path`
    /// (zarr node path, e.g. `/0/elevation`). The subset must be aligned to
    /// the shard grid so no read-modify-write is needed.
    ///
    /// Borrows the numpy buffer (zero-copy for the contiguous case) and encodes
    /// with the GIL released, so worker threads no longer serialize on a copy
    /// under the GIL. On S3 the encoded shards are uploaded asynchronously (see
    /// `PutQueue`) so encode and upload overlap; on filesystem the write is
    /// synchronous. Encode concurrency is bounded by the caller's thread pool,
    /// and the inner zarrs codec parallelism is pinned to 1 to leave the budget
    /// to that outer pool.
    fn write_region(
        &self,
        py: Python<'_>,
        path: &str,
        start: Vec<u64>,
        block: &Bound<'_, PyUntypedArray>,
    ) -> PyResult<()> {
        let array = self.array(py, path)?;
        let shape: Vec<u64> = block.shape().iter().map(|&n| n as u64).collect();
        if start.len() != shape.len() {
            return Err(PyValueError::new_err(format!(
                "start has {} axes but block has {}",
                start.len(),
                shape.len()
            )));
        }
        let subset = ArraySubset::new_with_start_shape(start, shape).map_err(err)?;
        // the caller's thread pool provides outer (per-region) parallelism, so
        // pin zarrs' inner chunk/codec fan-out to 1 (run on this worker thread)
        // and leave the CPU budget to that outer pool
        let mut codec_options = CodecOptions::default();
        codec_options.set_store_empty_chunks(true);
        codec_options.set_concurrent_target(1);

        // Borrow the numpy buffer (zero-copy for the contiguous case) and
        // encode with the GIL released: zarrs takes `&[T]` as borrowed
        // `ArrayBytes`, so the only copy is the one the codec makes anyway,
        // and it no longer serializes worker threads on the GIL. A
        // non-contiguous view falls back to a single owned copy.
        macro_rules! write {
            ($t:ty) => {{
                let arr = block.cast::<PyArrayDyn<$t>>()?;
                let ro = arr.readonly();
                let view = ro.as_array();
                // C-contiguous: borrow zero-copy. Otherwise copy into standard
                // (row-major) layout so the bytes match zarrs' expected order
                // (a plain to_owned would keep F-order memory and transpose).
                let owned: Option<Vec<$t>> = match view.as_slice() {
                    Some(_) => None,
                    None => Some(
                        view.as_standard_layout()
                            .to_owned()
                            .into_raw_vec_and_offset()
                            .0,
                    ),
                };
                let data: &[$t] = match &owned {
                    Some(v) => v.as_slice(),
                    None => view.as_slice().expect("contiguous checked above"),
                };
                let nbytes = (data.len() * std::mem::size_of::<$t>()) as u64;
                let nanos = py.detach(|| -> Result<u64, String> {
                    let t0 = Instant::now();
                    array
                        .store_array_subset_opt(&subset, data, &codec_options)
                        .map_err(|e| e.to_string())?;
                    Ok(t0.elapsed().as_nanos() as u64)
                });
                (nanos, nbytes)
            }};
        }

        let dtype = block.dtype();
        let (nanos, nbytes) = if dtype.is_equiv_to(&numpy::dtype::<f32>(py)) {
            write!(f32)
        } else if dtype.is_equiv_to(&numpy::dtype::<f64>(py)) {
            write!(f64)
        } else if dtype.is_equiv_to(&numpy::dtype::<i16>(py)) {
            write!(i16)
        } else if dtype.is_equiv_to(&numpy::dtype::<i32>(py)) {
            write!(i32)
        } else if dtype.is_equiv_to(&numpy::dtype::<i64>(py)) {
            write!(i64)
        } else if dtype.is_equiv_to(&numpy::dtype::<u8>(py)) {
            write!(u8)
        } else if dtype.is_equiv_to(&numpy::dtype::<u16>(py)) {
            write!(u16)
        } else {
            return Err(PyTypeError::new_err(format!(
                "unsupported dtype {dtype}; expected one of u8, u16, i16, i32, i64, f32, f64"
            )));
        };

        let nanos = nanos.map_err(PyRuntimeError::new_err)?;
        self.write_nanos.fetch_add(nanos, Ordering::Relaxed);
        self.write_bytes.fetch_add(nbytes, Ordering::Relaxed);
        self.write_ops.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Await all outstanding async PUTs (S3) and propagate the first error.
    /// No-op for filesystem stores, which write synchronously. Must be called
    /// after each write batch before reading `stats()` or starting a new level.
    fn flush(&self, py: Python<'_>) -> PyResult<()> {
        if let Some(q) = &self.put_queue {
            py.detach(|| q.flush()).map_err(PyRuntimeError::new_err)?;
        }
        Ok(())
    }
}
