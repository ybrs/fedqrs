//! The Arrow C-stream boundary between Python and Rust.
//!
//! Data crosses in Arrow's C stream ABI, wrapped in a PyCapsule named
//! `arrow_array_stream` (the Arrow PyCapsule interface). Nothing is copied at
//! the boundary: a producer hands over a `FFI_ArrowArrayStream` struct and the
//! consumer pulls batches through it lazily.
//!
//! Two directions:
//!   - import: a Python object exposing `__arrow_c_stream__` (any pyarrow
//!     RecordBatchReader / Table, or our Python `reader` callback's result)
//!     becomes an arrow-rs `ArrowArrayStreamReader`.
//!   - export: an arrow-rs record-batch stream becomes an `ArrowStreamExport`
//!     pyclass that Python turns back into a `pa.RecordBatchReader` via
//!     `pa.RecordBatchReader.from_stream(...)`.

use std::ffi::CString;

use arrow::array::{RecordBatch, RecordBatchIterator, RecordBatchReader};
use arrow::datatypes::SchemaRef;
use arrow::error::ArrowError;
use arrow::ffi_stream::{ArrowArrayStreamReader, FFI_ArrowArrayStream};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyCapsule;

const STREAM_CAPSULE_NAME: &str = "arrow_array_stream";

/// Import an Arrow C stream exposed by a Python object into an arrow-rs reader.
///
/// The object must implement the Arrow PyCapsule interface (`__arrow_c_stream__`).
/// We take ownership of the underlying `FFI_ArrowArrayStream` by moving it out of
/// the capsule and leaving an empty (released) struct behind, so the capsule's
/// own destructor becomes a no-op and the stream is released exactly once.
pub fn import_arrow_stream(obj: &Bound<'_, PyAny>) -> PyResult<ArrowArrayStreamReader> {
    if !obj.hasattr("__arrow_c_stream__")? {
        return Err(PyValueError::new_err(
            "source reader did not return an Arrow stream (no __arrow_c_stream__)",
        ));
    }
    let capsule_obj = obj.call_method0("__arrow_c_stream__")?;
    let capsule = capsule_obj.downcast::<PyCapsule>().map_err(|_| {
        PyValueError::new_err("__arrow_c_stream__ did not return a PyCapsule")
    })?;

    let ptr = capsule.pointer() as *mut FFI_ArrowArrayStream;
    if ptr.is_null() {
        return Err(PyValueError::new_err("Arrow stream capsule holds a null pointer"));
    }
    // Move the stream out; the capsule keeps an empty struct whose release is a
    // no-op. This is the ownership transfer the PyCapsule protocol prescribes.
    let stream = unsafe { std::ptr::replace(ptr, FFI_ArrowArrayStream::empty()) };
    ArrowArrayStreamReader::try_new(stream)
        .map_err(|e| PyValueError::new_err(format!("invalid Arrow stream: {e}")))
}

/// A Rust-owned Arrow stream, ready to be pulled by Python.
///
/// Holds the C stream struct until Python asks for it via `__arrow_c_stream__`,
/// at which point ownership moves into a PyCapsule. A stream is one-shot, so a
/// second export attempt raises rather than handing out a spent stream.
///
/// `unsendable`: the raw C stream struct is neither Send nor Sync, and it is
/// always created and consumed on the one Python thread driving the query.
#[pyclass(unsendable)]
pub struct ArrowStreamExport {
    inner: Option<FFI_ArrowArrayStream>,
}

impl ArrowStreamExport {
    /// Wrap any `Send` record-batch reader as an exportable stream.
    pub fn new(reader: Box<dyn RecordBatchReader + Send>) -> Self {
        ArrowStreamExport {
            inner: Some(FFI_ArrowArrayStream::new(reader)),
        }
    }
}

/// Build an exportable stream from fully-read batches and their schema.
pub fn stream_from_batches(schema: SchemaRef, batches: Vec<RecordBatch>) -> ArrowStreamExport {
    let items: Vec<Result<RecordBatch, ArrowError>> = batches.into_iter().map(Ok).collect();
    ArrowStreamExport::new(Box::new(RecordBatchIterator::new(items.into_iter(), schema)))
}

#[pymethods]
impl ArrowStreamExport {
    /// Arrow PyCapsule interface: hand the C stream to the consumer.
    ///
    /// `requested_schema` (a schema-cast request) is accepted and ignored; the
    /// engine already produces the exact schema Python planned for.
    #[pyo3(signature = (requested_schema=None))]
    fn __arrow_c_stream__<'py>(
        &mut self,
        py: Python<'py>,
        requested_schema: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyCapsule>> {
        let _ = requested_schema;
        let stream = self.inner.take().ok_or_else(|| {
            PyRuntimeError::new_err("Arrow stream already consumed")
        })?;
        let name = CString::new(STREAM_CAPSULE_NAME).unwrap();
        // The capsule owns the struct; when Python's consumer moves the stream
        // out it nulls the release, so the capsule's Drop sees an empty struct.
        PyCapsule::new(py, stream, Some(name))
    }
}
