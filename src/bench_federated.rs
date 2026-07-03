//! TRUE two-source federated query, one-to-one with the engine's real query
//! (`catalog_files F ⨝ file_access A ON F.id = A.file_id`):
//!   - probe = Postgres `perf_uuid`, fetched over ADBC,
//!   - build = a real DuckDB database `file_access`, fetched over a DuckDB conn,
//! both registered into a coordinator and merge-joined. Times each upstream
//! fetch and the merge, on both a DuckDB coordinator (forked vtab) and a
//! DataFusion coordinator.

use std::sync::Arc;
use std::time::Instant;

use arrow_array::{Int64Array, RecordBatch};
use duckdb::arrow::compute::concat_batches;
use duckdb::vtab::arrow::{arrow_recordbatch_to_query_params, ArrowVTab};
use duckdb::Connection;

use datafusion::datasource::MemTable;
use datafusion::prelude::SessionContext;
use tokio::runtime::Runtime;

use crate::bench_adbc_datafusion::{adbc_connect, fetch};

const PG_SQL: &str = "SELECT id::text AS id FROM perf_uuid";
const DUCK_SQL: &str = "SELECT file_id::text AS id FROM file_access";

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

fn duck_fetch(db: &Connection, sql: &str) -> Vec<RecordBatch> {
    let mut stmt = db.prepare(sql).expect("prepare");
    let mut out = Vec::new();
    for batch in stmt.query_arrow([]).expect("query_arrow") {
        out.push(batch);
    }
    out
}

fn concat(batches: &[RecordBatch]) -> RecordBatch {
    concat_batches(&batches[0].schema(), batches).expect("concat")
}

fn merge_duckdb(coord: &Connection, probe: &RecordBatch, build: &RecordBatch) -> usize {
    let pp = arrow_recordbatch_to_query_params(probe.clone());
    let bp = arrow_recordbatch_to_query_params(build.clone());
    let params = [pp[0], pp[1], bp[0], bp[1]];
    let sql = "SELECT count(*) FROM arrow(?, ?) p JOIN arrow(?, ?) b ON p.id = b.id";
    let mut stmt = coord.prepare(sql).expect("prepare merge");
    let count: i64 = stmt.query_row(params, |row| row.get(0)).expect("merge");
    count as usize
}

fn merge_datafusion(rt: &Runtime, probe: &RecordBatch, build: &RecordBatch) -> usize {
    let ctx = SessionContext::new();
    let p = MemTable::try_new(probe.schema(), vec![vec![probe.clone()]]).expect("p");
    let b = MemTable::try_new(build.schema(), vec![vec![build.clone()]]).expect("b");
    ctx.register_table("probe", Arc::new(p)).expect("rp");
    ctx.register_table("build", Arc::new(b)).expect("rb");
    rt.block_on(async {
        let frame = ctx
            .sql("SELECT count(*) FROM probe p JOIN build b ON p.id = b.id")
            .await
            .expect("plan");
        let batches = frame.collect().await.expect("collect");
        batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("int64")
            .value(0) as usize
    })
}

pub fn run(driver: &str, uri: &str, duck_path: &str) {
    let rt = Runtime::new().expect("tokio");
    let mut pg = adbc_connect(driver, uri);
    let src = Connection::open(duck_path).expect("open duckdb source");
    let coord = Connection::open_in_memory().expect("coordinator");
    coord
        .register_table_function::<ArrowVTab>("arrow")
        .expect("register arrow vtab");

    println!("\n== TRUE two-source federated query (Rust, both sources real) ==");
    println!("  probe (postgres/ADBC): {PG_SQL}");
    println!("  build (duckdb file):   {DUCK_SQL}");

    let probe = concat(&fetch(&mut pg, PG_SQL));
    let build = concat(&duck_fetch(&src, DUCK_SQL));
    println!("  probe rows={} build rows={}", probe.num_rows(), build.num_rows());
    println!(
        "  match count: duckdb={} datafusion={} (must be equal)",
        merge_duckdb(&coord, &probe, &build),
        merge_datafusion(&rt, &probe, &build)
    );

    time_best("fetch postgres probe (ADBC, I/O)", || {
        fetch(&mut pg, PG_SQL).iter().map(|b| b.num_rows()).sum()
    });
    time_best("fetch duckdb build (duckdb conn, I/O)", || {
        duck_fetch(&src, DUCK_SQL).iter().map(|b| b.num_rows()).sum()
    });
    time_best("merge in DuckDB coordinator (forked vtab)", || {
        merge_duckdb(&coord, &probe, &build)
    });
    time_best("merge in DataFusion coordinator", || {
        merge_datafusion(&rt, &probe, &build)
    });
    time_best("TOTAL (both fetches + DuckDB merge)", || {
        let p = concat(&fetch(&mut pg, PG_SQL));
        let b = concat(&duck_fetch(&src, DUCK_SQL));
        merge_duckdb(&coord, &p, &b)
    });
}
