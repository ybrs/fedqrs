//! `fedqrs` — the federated-query execution engine, exposed to Python as a
//! native extension module.
//!
//! Python parses/binds/optimizes/plans a query, serializes the physical plan to
//! a two-layer IR (orchestration steps + relational fragments), and hands it
//! here as JSON. This crate reads every source natively, runs the fragments on
//! DataFusion, and streams the final result back to Python as an Arrow C-stream.
//! No intermediate data is ever revived into Python objects.

mod connectors;
mod engine;
mod ffi;

use fedqrs_core::ir;

use pyo3::exceptions::{PyValueError};
use pyo3::prelude::*;

use ffi::{import_arrow_stream, stream_from_batches, ArrowStreamExport};

/// Register a datasource for the engine to read natively. Called once per
/// source at session init. `params` is a dict: postgres needs `uri` (and an
/// `adbc_driver` path); duckdb needs `path`.
#[pyfunction]
fn register_datasource(name: String, kind: &str, params: &Bound<'_, PyAny>) -> PyResult<()> {
    let spec = connectors::spec_from_params(kind, params)?;
    connectors::register(name, spec);
    Ok(())
}

/// Low-level entry: fetch `sql` from a registered source natively and return
/// the Arrow result as a stream. The engine uses the same `connectors::fetch`
/// internally; this exposes it for validation.
#[pyfunction]
fn fetch_to_stream(name: &str, sql: &str) -> PyResult<ArrowStreamExport> {
    let (schema, batches) = connectors::fetch(name, sql)?;
    Ok(stream_from_batches(schema, batches))
}

/// Parallel ctid-partitioned scan of a Postgres table, returned as one Arrow
/// stream. Exposed for benchmarking the parallel read against DuckDB.
#[pyfunction]
#[pyo3(signature = (name, table, select_list, partitions, schema=None))]
fn fetch_parallel_to_stream(
    name: &str,
    table: &str,
    select_list: &str,
    partitions: usize,
    schema: Option<String>,
) -> PyResult<ArrowStreamExport> {
    let (result_schema, batches) =
        connectors::fetch_parallel(name, schema.as_deref(), table, None, select_list, partitions, None)?;
    Ok(stream_from_batches(result_schema, batches))
}

/// Execute a query IR and return the result as an exportable Arrow stream.
/// Everything runs in Rust; the result is the only thing that crosses back.
#[pyfunction]
fn execute_ir(ir_json: &str) -> PyResult<ArrowStreamExport> {
    let ir: ir::Ir = serde_json::from_str(ir_json)
        .map_err(|e| PyValueError::new_err(format!("invalid IR JSON: {e}")))?;
    engine::execute(&ir)
}

/// Smoke-test entry point: proves the extension loads and returns a value.
#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// Import an Arrow stream from a Python producer and hand it straight back.
/// Exercises both directions of the FFI boundary with no engine in between.
#[pyfunction]
fn roundtrip(source: &Bound<'_, PyAny>) -> PyResult<ArrowStreamExport> {
    let reader = import_arrow_stream(source)?;
    Ok(ArrowStreamExport::new(Box::new(reader)))
}

#[pymodule]
fn fedqrs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(roundtrip, m)?)?;
    m.add_function(wrap_pyfunction!(execute_ir, m)?)?;
    m.add_function(wrap_pyfunction!(register_datasource, m)?)?;
    m.add_function(wrap_pyfunction!(fetch_to_stream, m)?)?;
    m.add_function(wrap_pyfunction!(fetch_parallel_to_stream, m)?)?;
    m.add_class::<ArrowStreamExport>()?;
    Ok(())
}
