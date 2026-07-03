//! Native source connectors and the datasource registry.
//!
//! Python registers each datasource once at session init via
//! `register_datasource`; the engine then fetches from a source by name, in
//! Rust, over a native driver. Fetched data stays in Rust (as Arrow) for the
//! rest of the query - it is never revived into Python objects.
//!
//! Postgres reads go over ADBC (the same C driver the Python path uses), which
//! decodes the wire straight into Arrow. DuckDB lands in a later phase.

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use arrow::array::{ArrayRef, Int64Array, RecordBatch, RecordBatchReader};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
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

/// A cached, reused Postgres connection (driver + database kept alive with it).
struct PgConn {
    _driver: adbc_driver_manager::ManagedDriver,
    _database: adbc_driver_manager::ManagedDatabase,
    conn: adbc_driver_manager::ManagedConnection,
}

thread_local! {
    // One live connection per datasource name, on the query-driving thread.
    // ADBC handles are not Send, and all fetches run on this one thread, so a
    // thread-local cache pools connections without any locking.
    static PG_CACHE: RefCell<HashMap<String, PgConn>> = RefCell::new(HashMap::new());
}

/// Run `sql` against the named source over its native driver and return the
/// full Arrow result held in Rust.
pub fn fetch(name: &str, sql: &str) -> PyResult<(SchemaRef, Vec<RecordBatch>)> {
    let s = spec(name)?;
    match s.kind {
        DsKind::Postgres => fetch_postgres(name, &s, sql),
        DsKind::DuckDb => Err(PyRuntimeError::new_err(
            "duckdb native connector not yet implemented",
        )),
    }
}

// Returns a String error (not PyErr) so it can also run on a worker thread that
// does not hold the GIL (the parallel-scan path spawns such threads).
fn open_pg(s: &DsSpec) -> Result<PgConn, String> {
    use adbc_core::options::{AdbcVersion, OptionDatabase, OptionValue};
    use adbc_core::{Database, Driver};

    let driver_path = s
        .adbc_driver
        .as_deref()
        .ok_or_else(|| "postgres datasource requires an 'adbc_driver' path".to_string())?;
    let mut driver = adbc_driver_manager::ManagedDriver::load_dynamic_from_filename(
        driver_path,
        None,
        AdbcVersion::V100,
    )
    .map_err(|e| format!("load adbc driver: {e}"))?;
    let opts = [(OptionDatabase::Uri, OptionValue::String(s.uri.clone()))];
    let mut database = driver
        .new_database_with_opts(opts)
        .map_err(|e| format!("adbc database: {e}"))?;
    let conn = database
        .new_connection()
        .map_err(|e| format!("adbc connection: {e}"))?;
    Ok(PgConn { _driver: driver, _database: database, conn })
}

fn fetch_postgres(
    name: &str,
    s: &DsSpec,
    sql: &str,
) -> PyResult<(SchemaRef, Vec<RecordBatch>)> {
    use adbc_core::{Connection, Statement};

    PG_CACHE.with(|cache| {
        let mut map = cache.borrow_mut();
        if !map.contains_key(name) {
            let conn = open_pg(s).map_err(PyRuntimeError::new_err)?;
            map.insert(name.to_string(), conn);
        }
        let pg = map.get_mut(name).unwrap();

        let mut stmt = pg
            .conn
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
            batches
                .push(batch.map_err(|e| PyRuntimeError::new_err(format!("adbc batch: {e}")))?);
        }
        normalize_numeric(schema, batches)
            .map_err(|e| PyRuntimeError::new_err(format!("numeric normalize: {e}")))
    })
}

/// Postgres `numeric`/`decimal` columns arrive over ADBC as an opaque
/// string-backed extension type; DataFusion is strictly typed and will not do
/// arithmetic on them. Cast such columns to `Float64` at the boundary (parsing
/// the string values) so downstream operators see a real number. Float64 is a
/// pragmatic choice; exact decimal semantics (Decimal128) can come later.
fn normalize_numeric(
    schema: SchemaRef,
    batches: Vec<RecordBatch>,
) -> Result<(SchemaRef, Vec<RecordBatch>), arrow::error::ArrowError> {
    let numeric: Vec<usize> = schema
        .fields()
        .iter()
        .enumerate()
        .filter(|(_, f)| {
            matches!(
                f.metadata().get("ADBC:postgresql:typname").map(String::as_str),
                Some("numeric") | Some("decimal")
            )
        })
        .map(|(i, _)| i)
        .collect();
    if numeric.is_empty() {
        return Ok((schema, batches));
    }

    let fields: Vec<Arc<Field>> = schema
        .fields()
        .iter()
        .enumerate()
        .map(|(i, f)| {
            if numeric.contains(&i) {
                // Drop the extension metadata; the column is now a plain float.
                Arc::new(Field::new(f.name(), DataType::Float64, f.is_nullable()))
            } else {
                f.clone()
            }
        })
        .collect();
    let new_schema = Arc::new(Schema::new(fields));

    let mut out = Vec::with_capacity(batches.len());
    for batch in batches {
        let mut cols: Vec<ArrayRef> = Vec::with_capacity(batch.num_columns());
        for (i, col) in batch.columns().iter().enumerate() {
            if numeric.contains(&i) {
                cols.push(arrow::compute::cast(col, &DataType::Float64)?);
            } else {
                cols.push(col.clone());
            }
        }
        out.push(RecordBatch::try_new(new_schema.clone(), cols)?);
    }
    Ok((new_schema, out))
}

// --- parallel partitioned scan -----------------------------------------------
//
// A large whole-table read over one Postgres connection is bandwidth-bound. We
// match DuckDB's postgres scanner: split the table's heap into `ctid` page
// ranges and read them concurrently over N connections (each a binary COPY),
// then concatenate the Arrow. `ctid` is a row's physical (page, tuple) address,
// so page ranges partition the table with no partition column and no overlap.

/// Escape a single-quoted SQL string literal.
fn esc(s: &str) -> String {
    s.replace('\'', "''")
}

/// Quote a (schema.)table reference.
fn qualified_table(schema: Option<&str>, table: &str) -> String {
    let t = table.replace('"', "\"\"");
    match schema {
        Some(s) => format!("\"{}\".\"{}\"", s.replace('"', "\"\""), t),
        None => format!("\"{t}\""),
    }
}

/// The table's heap page count (`pg_class.relpages`), for sizing the ranges.
fn relpages(name: &str, schema: Option<&str>, table: &str) -> PyResult<u32> {
    let pred = match schema {
        Some(s) => format!("c.relname = '{}' AND n.nspname = '{}'", esc(table), esc(s)),
        None => format!("c.relname = '{}'", esc(table)),
    };
    let sql = format!(
        "SELECT c.relpages::bigint FROM pg_class c \
         JOIN pg_namespace n ON n.oid = c.relnamespace WHERE {pred}"
    );
    let (_, batches) = fetch(name, &sql)?;
    for batch in &batches {
        if batch.num_rows() > 0 {
            let col = batch
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .ok_or_else(|| PyRuntimeError::new_err("relpages is not int8"))?;
            return Ok(col.value(0).max(0) as u32);
        }
    }
    Err(PyRuntimeError::new_err(format!(
        "table '{table}' not found for relpages"
    )))
}

/// Split [0, pages) into `partitions` half-open page ranges; the last extends to
/// the max page so rows added since the last ANALYZE are still covered.
fn ctid_ranges(pages: u32, partitions: usize) -> Vec<(u32, u32)> {
    if pages == 0 {
        return vec![(0, u32::MAX)];
    }
    let partitions = (partitions.max(1) as u32).min(pages);
    let chunk = (pages / partitions).max(1);
    let mut ranges = Vec::new();
    let mut lo = 0u32;
    while lo < pages {
        let hi = lo.saturating_add(chunk);
        ranges.push((lo, hi));
        lo = hi;
    }
    if let Some(last) = ranges.last_mut() {
        last.1 = u32::MAX;
    }
    ranges
}

/// Read one page range on its own connection (runs on a worker thread).
fn read_partition(spec: &DsSpec, sql: &str) -> Result<(SchemaRef, Vec<RecordBatch>), String> {
    use adbc_core::{Connection, Statement};

    // conn and stmt must outlive the reader (dropping the statement invalidates
    // the ADBC C stream), so they stay in scope through the read below.
    let mut conn = open_pg(spec)?;
    let mut stmt = conn.conn.new_statement().map_err(|e| format!("adbc statement: {e}"))?;
    stmt.set_sql_query(sql).map_err(|e| format!("adbc set sql: {e}"))?;
    let reader = stmt.execute().map_err(|e| format!("adbc execute: {e}"))?;
    let schema = reader.schema();
    let mut batches = Vec::new();
    for batch in reader {
        batches.push(batch.map_err(|e| format!("adbc batch: {e}"))?);
    }
    normalize_numeric(schema, batches).map_err(|e| format!("numeric normalize: {e}"))
}

/// Read `select_list` from a Postgres table with `partitions`-way parallel
/// ctid-partitioned binary COPY reads, concatenated into one Arrow result.
pub fn fetch_parallel(
    name: &str,
    schema: Option<&str>,
    table: &str,
    select_list: &str,
    partitions: usize,
) -> PyResult<(SchemaRef, Vec<RecordBatch>)> {
    let s = spec(name)?;
    if s.kind != DsKind::Postgres {
        return Err(PyRuntimeError::new_err("parallel fetch is Postgres-only"));
    }
    let pages = relpages(name, schema, table)?;
    let table_ref = qualified_table(schema, table);

    let mut handles = Vec::new();
    for (lo, hi) in ctid_ranges(pages, partitions) {
        let worker_spec = s.clone();
        let sql = format!(
            "SELECT {select_list} FROM {table_ref} \
             WHERE ctid >= '({lo},0)'::tid AND ctid < '({hi},0)'::tid"
        );
        handles.push(std::thread::spawn(move || read_partition(&worker_spec, &sql)));
    }

    let mut result_schema: Option<SchemaRef> = None;
    let mut all = Vec::new();
    for handle in handles {
        let joined = handle
            .join()
            .map_err(|_| PyRuntimeError::new_err("partition thread panicked"))?;
        let (schema, batches) = joined.map_err(PyRuntimeError::new_err)?;
        result_schema.get_or_insert(schema);
        all.extend(batches);
    }
    let result_schema = result_schema
        .ok_or_else(|| PyRuntimeError::new_err("parallel fetch produced no partitions"))?;
    Ok((result_schema, all))
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
