//! Native source connectors and the datasource registry.
//!
//! Python registers each datasource once at session init via
//! `register_datasource`; the engine then fetches from a source by name, in
//! Rust, over a native driver. Fetched data stays in Rust (as Arrow) for the
//! rest of the query - it is never revived into Python objects.
//!
//! Postgres reads go over ADBC (the same C driver the Python path uses), which
//! decodes the wire straight into Arrow. DuckDB lands in a later phase.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use arrow::array::{RecordBatch, RecordBatchReader};
use arrow::datatypes::SchemaRef;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum DsKind {
    Postgres,
    DuckDb,
}

/// Connection parameters for one registered datasource. Parameters are stored
/// (not a live handle) for now; pooling live connections is a later step.
#[derive(Clone)]
pub struct DsSpec {
    pub kind: DsKind,
    /// Postgres: the connection URI. DuckDB: the database file path.
    pub uri: String,
    /// Path to the ADBC driver shared library (Postgres only).
    pub adbc_driver: Option<String>,
}

static REGISTRY: OnceLock<Mutex<HashMap<String, DsSpec>>> = OnceLock::new();

fn registry() -> &'static Mutex<HashMap<String, DsSpec>> {
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Register (or replace) a datasource under `name`.
pub fn register(name: String, spec: DsSpec) {
    registry().lock().unwrap().insert(name, spec);
}

fn spec(name: &str) -> PyResult<DsSpec> {
    registry()
        .lock()
        .unwrap()
        .get(name)
        .cloned()
        .ok_or_else(|| PyRuntimeError::new_err(format!("datasource '{name}' is not registered")))
}

/// The kind of a registered datasource (used to pick the SQL dialect).
pub fn kind(name: &str) -> PyResult<DsKind> {
    Ok(spec(name)?.kind)
}

/// Run `sql` against the named source over its native driver and return the
/// full Arrow result held in Rust.
pub fn fetch(name: &str, sql: &str) -> PyResult<(SchemaRef, Vec<RecordBatch>)> {
    let s = spec(name)?;
    match s.kind {
        DsKind::Postgres => fetch_postgres(&s, sql),
        DsKind::DuckDb => Err(PyRuntimeError::new_err(
            "duckdb native connector not yet implemented",
        )),
    }
}

fn fetch_postgres(s: &DsSpec, sql: &str) -> PyResult<(SchemaRef, Vec<RecordBatch>)> {
    use adbc_core::options::{AdbcVersion, OptionDatabase, OptionValue};
    use adbc_core::{Connection, Database, Driver, Statement};

    let driver_path = s.adbc_driver.as_deref().ok_or_else(|| {
        PyValueError::new_err("postgres datasource requires an 'adbc_driver' path")
    })?;

    let mut driver = adbc_driver_manager::ManagedDriver::load_dynamic_from_filename(
        driver_path,
        None,
        AdbcVersion::V100,
    )
    .map_err(|e| PyRuntimeError::new_err(format!("load adbc driver: {e}")))?;

    let opts = [(OptionDatabase::Uri, OptionValue::String(s.uri.clone()))];
    let mut database = driver
        .new_database_with_opts(opts)
        .map_err(|e| PyRuntimeError::new_err(format!("adbc database: {e}")))?;
    let mut conn = database
        .new_connection()
        .map_err(|e| PyRuntimeError::new_err(format!("adbc connection: {e}")))?;
    let mut stmt = conn
        .new_statement()
        .map_err(|e| PyRuntimeError::new_err(format!("adbc statement: {e}")))?;
    stmt.set_sql_query(sql)
        .map_err(|e| PyRuntimeError::new_err(format!("adbc set sql: {e}")))?;
    let reader = stmt
        .execute()
        .map_err(|e| PyRuntimeError::new_err(format!("adbc execute: {e}")))?;

    let schema = reader.schema();
    let mut batches = Vec::new();
    for batch in reader {
        batches.push(batch.map_err(|e| PyRuntimeError::new_err(format!("adbc batch: {e}")))?);
    }
    Ok((schema, batches))
}

/// Parse the `register_datasource` params dict into a spec.
pub fn spec_from_params(kind: &str, params: &Bound<'_, PyAny>) -> PyResult<DsSpec> {
    let get = |key: &str| -> PyResult<Option<String>> {
        match params.get_item(key) {
            Ok(v) => Ok(Some(v.extract::<String>()?)),
            Err(_) => Ok(None),
        }
    };
    match kind {
        "postgres" | "postgresql" => {
            let uri = get("uri")?
                .ok_or_else(|| PyValueError::new_err("postgres datasource needs 'uri'"))?;
            let adbc_driver = get("adbc_driver")?;
            Ok(DsSpec { kind: DsKind::Postgres, uri, adbc_driver })
        }
        "duckdb" => {
            let uri = get("path")?
                .ok_or_else(|| PyValueError::new_err("duckdb datasource needs 'path'"))?;
            Ok(DsSpec { kind: DsKind::DuckDb, uri, adbc_driver: None })
        }
        other => Err(PyValueError::new_err(format!(
            "unknown datasource kind '{other}'"
        ))),
    }
}
