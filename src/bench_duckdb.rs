//! DuckDB-side uuid experiment, run entirely through the duckdb crate's own
//! (bundled) Arrow — no Python/pyarrow overhead. It answers two questions the
//! Python measurements raised:
//!
//!   1. Does DuckDB emit uuid as a native binary type, or as a string, and is
//!      `CAST(uuid AS BLOB)` cheaper or more expensive than the default string?
//!   2. For the engine's flow — bring an Arrow result into a coordinator DuckDB
//!      and merge-join it — is the binary representation faster, or the same?

use std::time::Instant;

use duckdb::arrow::record_batch::RecordBatch;
use duckdb::vtab::arrow::{arrow_recordbatch_to_query_params, ArrowVTab};
use duckdb::Connection;

const ROWS: usize = 100_000;
const BUILD: usize = 1_000;
// The duckdb crate's arrow() table function copies a whole batch into a single
// DuckDB vector and asserts it fits, so batches must stay within the vector
// size. (The Python duckdb the engine uses has no such limit.)
const VEC: usize = 2048;

fn time_best<F: FnMut() -> usize>(label: &str, mut f: F) {
    let rows = f(); // warmup
    let mut best = f64::MAX;
    for _ in 0..5 {
        let t = Instant::now();
        let _ = f();
        best = best.min(t.elapsed().as_secs_f64() * 1000.0);
    }
    println!("  {label:48} {best:7.2} ms   ({rows} rows)");
}

fn export(db: &Connection, sql: &str) -> Vec<RecordBatch> {
    let mut stmt = db.prepare(sql).expect("prepare");
    let mut out = Vec::new();
    for batch in stmt.query_arrow([]).expect("query_arrow") {
        out.push(batch);
    }
    out
}

fn count_rows(batches: &[RecordBatch]) -> usize {
    let mut total = 0;
    for batch in batches {
        total += batch.num_rows();
    }
    total
}

fn arrow_type(db: &Connection, sql: &str) -> String {
    let batches = export(db, sql);
    format!("{:?}", batches[0].schema().field(0).data_type())
}

// Slice batches to <= VEC rows. Arrow slicing shares buffers (zero-copy).
fn slice_to_vec(batches: &[RecordBatch]) -> Vec<RecordBatch> {
    let mut out = Vec::new();
    for batch in batches {
        let mut offset = 0;
        while offset < batch.num_rows() {
            let len = (batch.num_rows() - offset).min(VEC);
            out.push(batch.slice(offset, len));
            offset += len;
        }
    }
    out
}

// Engine flow done right: register each probe batch as a ZERO-COPY Arrow scan
// (no copy into a DuckDB table) and hash-join it against the small build table.
fn merge_zerocopy(db: &Connection, probe: &[RecordBatch], build: &str) -> usize {
    let sql = format!("SELECT count(*) FROM arrow(?, ?) p JOIN {build} b ON p.id = b.id");
    let mut stmt = db.prepare(&sql).expect("prepare merge");
    let mut total: i64 = 0;
    for batch in probe {
        let params = arrow_recordbatch_to_query_params(batch.clone());
        total += stmt
            .query_row(params, |row| row.get::<_, i64>(0))
            .expect("merge");
    }
    total as usize
}

// Pure in-DuckDB join (no Arrow round trip) to isolate the join-key cost by type.
fn pure_join(db: &Connection, key: &str) -> usize {
    let sql = format!("SELECT count(*) FROM t p JOIN samp b ON {key}");
    let count: i64 = db.query_row(&sql, [], |row| row.get(0)).expect("pure join");
    count as usize
}

fn setup(db: &Connection) {
    db.register_table_function::<ArrowVTab>("arrow")
        .expect("register arrow vtab");
    db.execute_batch(&format!(
        "CREATE TABLE t AS SELECT gen_random_uuid() AS id, gen_random_uuid() AS tid \
           FROM range({ROWS});
         CREATE TABLE samp AS SELECT id FROM t LIMIT {BUILD};
         CREATE TABLE build_str AS SELECT CAST(id AS VARCHAR) AS id FROM samp;
         CREATE TABLE build_bin AS SELECT CAST(id AS BLOB) AS id FROM samp;"
    ))
    .expect("setup tables");
}

pub fn run() {
    let db = Connection::open_in_memory().expect("duckdb open");
    setup(&db);

    println!("\n== DuckDB uuid experiment (Rust, {ROWS} rows, no Python) ==");
    println!(
        "  uuid default arrow type : {}",
        arrow_type(&db, "SELECT id FROM t")
    );
    println!(
        "  CAST AS BLOB arrow type : {}",
        arrow_type(&db, "SELECT CAST(id AS BLOB) id FROM t")
    );

    println!("  -- export (DuckDB source -> Arrow) --");
    time_best("export 1col uuid -> STRING", || {
        count_rows(&export(&db, "SELECT id FROM t"))
    });
    time_best("export 1col uuid -> BLOB", || {
        count_rows(&export(&db, "SELECT CAST(id AS BLOB) id FROM t"))
    });
    time_best("export 2col uuid -> STRING", || {
        count_rows(&export(&db, "SELECT id, tid FROM t"))
    });
    time_best("export 2col uuid -> BLOB", || {
        count_rows(&export(
            &db,
            "SELECT CAST(id AS BLOB) id, CAST(tid AS BLOB) tid FROM t",
        ))
    });

    println!("  -- pure join inside DuckDB (no Arrow), by key type --");
    println!(
        "  match counts: native={} string={} blob={} (must be equal)",
        pure_join(&db, "p.id = b.id"),
        pure_join(&db, "CAST(p.id AS VARCHAR) = CAST(b.id AS VARCHAR)"),
        pure_join(&db, "CAST(p.id AS BLOB) = CAST(b.id AS BLOB)")
    );
    time_best("pure join, NATIVE uuid key", || pure_join(&db, "p.id = b.id"));
    time_best("pure join, STRING key", || {
        pure_join(&db, "CAST(p.id AS VARCHAR) = CAST(b.id AS VARCHAR)")
    });
    time_best("pure join, BLOB key", || {
        pure_join(&db, "CAST(p.id AS BLOB) = CAST(b.id AS BLOB)")
    });

    println!("  -- merge: ZERO-COPY register Arrow probe + hash join (no copy) --");
    // Pre-slice the probe to the binding's vector-size limit (zero-copy slices).
    let probe_str = slice_to_vec(&export(&db, "SELECT CAST(id AS VARCHAR) id FROM t"));
    let probe_bin = slice_to_vec(&export(&db, "SELECT CAST(id AS BLOB) id FROM t"));
    println!(
        "  match STRING={} BLOB={} (must be equal)",
        merge_zerocopy(&db, &probe_str, "build_str"),
        merge_zerocopy(&db, &probe_bin, "build_bin")
    );
    time_best("merge STRING (register+join, no copy)", || {
        merge_zerocopy(&db, &probe_str, "build_str")
    });
    time_best("merge BLOB (register+join, no copy)", || {
        merge_zerocopy(&db, &probe_bin, "build_bin")
    });
}
