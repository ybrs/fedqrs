//! Apples-to-apples end-to-end Rust path: fetch the probe from real Postgres
//! over ADBC (the same C driver the Python engine uses), then merge-join it in
//! DataFusion (Arrow-native, zero-copy register). Times the fetch (I/O) and the
//! merge (compute) separately, so they can be compared to the Python engine's
//! ADBC-fetch + DuckDB-merge numbers.

use std::sync::Arc;
use std::time::Instant;

use arrow_array::{Array, FixedSizeBinaryArray, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use datafusion::datasource::MemTable;
use datafusion::prelude::SessionContext;
use tokio::runtime::Runtime;

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

// Open the ADBC connection once (driver load + Postgres connect are one-time,
// just like the engine's pooled connection). Returns the live connection.
pub fn adbc_connect(
    driver_path: &str,
    uri: &str,
) -> impl adbc_core::Connection {
    use adbc_core::options::{AdbcVersion, OptionDatabase, OptionValue};
    use adbc_core::{Database, Driver};

    let mut driver = adbc_driver_manager::ManagedDriver::load_dynamic_from_filename(
        driver_path,
        None,
        AdbcVersion::V100,
    )
    .expect("load adbc driver");
    let opts = [(OptionDatabase::Uri, OptionValue::String(uri.to_string()))];
    let database = driver.new_database_with_opts(opts).expect("adbc database");
    database.new_connection().expect("adbc connection")
}

// Execute + fetch on an already-open connection (this is the I/O we want to time).
pub fn fetch<C: adbc_core::Connection>(conn: &mut C, sql: &str) -> Vec<RecordBatch> {
    use adbc_core::Statement;
    let mut stmt = conn.new_statement().expect("adbc statement");
    stmt.set_sql_query(sql).expect("adbc set sql");
    let reader = stmt.execute().expect("adbc execute");
    let mut out = Vec::new();
    for batch in reader {
        out.push(batch.expect("adbc batch"));
    }
    out
}

fn count_rows(batches: &[RecordBatch]) -> usize {
    batches.iter().map(|b| b.num_rows()).sum()
}

// Build the small side from the first BUILD probe keys, as strings.
fn build_str(probe: &[RecordBatch]) -> RecordBatch {
    let col = probe[0]
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("probe id is utf8");
    let take = BUILD.min(col.len());
    let mut vals = Vec::with_capacity(take);
    for i in 0..take {
        vals.push(col.value(i).to_string());
    }
    RecordBatch::try_new(probe[0].schema(), vec![Arc::new(StringArray::from(vals))])
        .expect("build_str batch")
}

// Re-encode string-uuid batches as native FixedSizeBinary(16).
fn to_binary(probe: &[RecordBatch]) -> Vec<RecordBatch> {
    let schema = Arc::new(Schema::new(vec![Field::new(
        "id",
        DataType::FixedSizeBinary(16),
        false,
    )]));
    let mut out = Vec::with_capacity(probe.len());
    for batch in probe {
        let col = batch
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("utf8");
        let mut rows = Vec::with_capacity(col.len());
        for i in 0..col.len() {
            rows.push(*uuid::Uuid::parse_str(col.value(i)).expect("uuid").as_bytes());
        }
        let array = FixedSizeBinaryArray::try_from_iter(rows.into_iter()).expect("bin");
        out.push(RecordBatch::try_new(schema.clone(), vec![Arc::new(array)]).expect("bin batch"));
    }
    out
}

fn merge(rt: &Runtime, probe: &[RecordBatch], build: RecordBatch) -> usize {
    let ctx = SessionContext::new();
    let provider = MemTable::try_new(probe[0].schema(), vec![probe.to_vec()]).expect("memtable");
    ctx.register_table("probe", Arc::new(provider)).expect("register probe");
    ctx.register_batch("build", build).expect("register build");
    rt.block_on(async {
        let sql = "SELECT count(*) FROM probe p JOIN build b ON p.id = b.id";
        let frame = ctx.sql(sql).await.expect("plan");
        let batches = frame.collect().await.expect("collect");
        batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("int64")
            .value(0) as usize
    })
}

pub fn run(driver: &str, uri: &str, sql: &str) {
    let rt = Runtime::new().expect("tokio runtime");
    let mut conn = adbc_connect(driver, uri); // one-time, like a pooled connection
    println!("\n== ADBC Postgres fetch -> DataFusion merge (Rust, end-to-end) ==");
    println!("  probe sql: {sql}");

    let probe = fetch(&mut conn, sql);
    let rows = count_rows(&probe);
    println!(
        "  fetched {rows} rows, probe type: {:?}",
        probe[0].schema().field(0).data_type()
    );

    time_best("ADBC fetch (reused conn, I/O only)", || {
        count_rows(&fetch(&mut conn, sql))
    });

    let build = build_str(&probe);
    let probe_bin = to_binary(&probe);
    let build_bin = to_binary(std::slice::from_ref(&build))
        .into_iter()
        .next()
        .unwrap();
    println!(
        "  match STRING={} BINARY={} (must be equal)",
        merge(&rt, &probe, build.clone()),
        merge(&rt, &probe_bin, build_bin.clone())
    );

    time_best("DataFusion merge, STRING keys", || merge(&rt, &probe, build.clone()));
    time_best("DataFusion merge, BINARY keys", || {
        merge(&rt, &probe_bin, build_bin.clone())
    });
    time_best("TOTAL string: ADBC fetch + DataFusion merge", || {
        let p = fetch(&mut conn, sql);
        let b = build_str(&p);
        merge(&rt, &p, b)
    });
}
