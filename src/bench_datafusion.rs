//! DataFusion is a pure-Rust, Arrow-native query engine: it ingests Arrow
//! RecordBatches directly (no FFI, no copy, no per-batch row cap), so it is a
//! clean way to measure the merge join with string vs native-binary uuid keys
//! — the part the duckdb crate's single-batch arrow() helper could not do in
//! one query.

use std::sync::Arc;
use std::time::Instant;

use datafusion::arrow::array::{FixedSizeBinaryArray, RecordBatch, StringArray};
use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::prelude::SessionContext;
use tokio::runtime::Runtime;
use uuid::Uuid;

const ROWS: usize = 100_000;
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

fn string_batch(uuids: &[Uuid], take: usize) -> RecordBatch {
    let array = StringArray::from_iter_values(uuids.iter().take(take).map(|u| u.to_string()));
    let schema = Schema::new(vec![Field::new("id", DataType::Utf8, false)]);
    RecordBatch::try_new(Arc::new(schema), vec![Arc::new(array)]).expect("string batch")
}

fn binary_batch(uuids: &[Uuid], take: usize) -> RecordBatch {
    let bytes = uuids.iter().take(take).map(|u| *u.as_bytes());
    let array = FixedSizeBinaryArray::try_from_iter(bytes).expect("binary array");
    let schema = Schema::new(vec![Field::new("id", DataType::FixedSizeBinary(16), false)]);
    RecordBatch::try_new(Arc::new(schema), vec![Arc::new(array)]).expect("binary batch")
}

fn merge(rt: &Runtime, probe: RecordBatch, build: RecordBatch) -> usize {
    let ctx = SessionContext::new();
    ctx.register_batch("probe", probe).expect("register probe");
    ctx.register_batch("build", build).expect("register build");
    rt.block_on(async {
        let sql = "SELECT count(*) FROM probe p JOIN build b ON p.id = b.id";
        let frame = ctx.sql(sql).await.expect("plan");
        let batches = frame.collect().await.expect("collect");
        let array = batches[0].column(0);
        let counts = array
            .as_any()
            .downcast_ref::<datafusion::arrow::array::Int64Array>()
            .expect("int64");
        counts.value(0) as usize
    })
}

pub fn run() {
    let rt = Runtime::new().expect("tokio runtime");
    let uuids: Vec<Uuid> = (0..ROWS).map(|_| Uuid::new_v4()).collect();

    // Build the Arrow batches once; clone() inside the loop is just Arc bumps,
    // so the timing covers only register + join, not array construction.
    let probe_str = string_batch(&uuids, ROWS);
    let build_str = string_batch(&uuids, BUILD);
    let probe_bin = binary_batch(&uuids, ROWS);
    let build_bin = binary_batch(&uuids, BUILD);

    println!("\n== DataFusion uuid merge (Rust, Arrow-native, {ROWS} rows) ==");
    println!(
        "  match STRING={} BINARY={} (must be equal)",
        merge(&rt, probe_str.clone(), build_str.clone()),
        merge(&rt, probe_bin.clone(), build_bin.clone())
    );
    time_best("merge STRING keys (Arrow-native, no copy)", || {
        merge(&rt, probe_str.clone(), build_str.clone())
    });
    time_best("merge BINARY(16) keys (Arrow-native, no copy)", || {
        merge(&rt, probe_bin.clone(), build_bin.clone())
    });
}
