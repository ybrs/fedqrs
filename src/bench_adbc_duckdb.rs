//! Same-stack apples-to-apples: fetch the probe from Postgres over ADBC, then
//! merge-join it in DuckDB — the engine's actual local engine — using the
//! VENDORED-FORK arrow() table function that streams batches > 2048 rows. This
//! is the real zero-copy single-query path the upstream crate couldn't do.

use std::time::Instant;

use arrow_array::{Array, RecordBatch, StringArray};
use duckdb::arrow::compute::concat_batches;
use duckdb::vtab::arrow::{arrow_recordbatch_to_query_params, ArrowVTab};
use duckdb::Connection;

use crate::bench_adbc_datafusion::{adbc_connect, fetch};

const BUILD: usize = 1_000;

fn time_best<F: FnMut() -> usize>(label: &str, mut f: F) {
    let rows = f(); // warmup
    let mut best = f64::MAX;
    for _ in 0..5 {
        let t = Instant::now();
        let _ = f();
        best = best.min(t.elapsed().as_secs_f64() * 1000.0);
    }
    println!("  {label:46} {best:7.2} ms   ({rows} rows)");
}

// Build the small side from the first BUILD probe keys.
fn build_str(probe: &RecordBatch) -> RecordBatch {
    let col = probe
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("utf8");
    let take = BUILD.min(col.len());
    let mut vals = Vec::with_capacity(take);
    for i in 0..take {
        vals.push(col.value(i).to_string());
    }
    RecordBatch::try_new(probe.schema(), vec![std::sync::Arc::new(StringArray::from(vals))])
        .expect("build batch")
}

// Register both Arrow inputs as zero-copy arrow() scans and hash-join them.
fn merge_duckdb(db: &Connection, probe: &RecordBatch, build: &RecordBatch) -> usize {
    let pp = arrow_recordbatch_to_query_params(probe.clone());
    let bp = arrow_recordbatch_to_query_params(build.clone());
    let params = [pp[0], pp[1], bp[0], bp[1]];
    let sql = "SELECT count(*) FROM arrow(?, ?) p JOIN arrow(?, ?) b ON p.id = b.id";
    let mut stmt = db.prepare(sql).expect("prepare");
    let count: i64 = stmt.query_row(params, |row| row.get(0)).expect("merge");
    count as usize
}

pub fn run(driver: &str, uri: &str, sql: &str) {
    let db = Connection::open_in_memory().expect("duckdb");
    db.register_table_function::<ArrowVTab>("arrow")
        .expect("register arrow vtab (forked: streams >2048)");
    let mut conn = adbc_connect(driver, uri);

    println!("\n== ADBC Postgres fetch -> DuckDB merge (Rust, forked vtab, zero-copy) ==");
    println!("  probe sql: {sql}");

    let batches = fetch(&mut conn, sql);
    let probe = concat_batches(&batches[0].schema(), &batches).expect("concat probe");
    println!("  fetched {} rows", probe.num_rows());

    let build = build_str(&probe);
    println!(
        "  match count = {} (single zero-copy query over 11k rows)",
        merge_duckdb(&db, &probe, &build)
    );

    time_best("ADBC fetch (reused conn, I/O only)", || {
        fetch(&mut conn, sql).iter().map(|b| b.num_rows()).sum()
    });
    time_best("DuckDB merge (forked arrow() vtab, 1 query)", || {
        merge_duckdb(&db, &probe, &build)
    });
}
