// Extra connector paths: ADBC (C driver via adbc_core) and connectorx (native Rust).
use std::time::Instant;

pub fn time_best<F: FnMut() -> usize>(label: &str, mut f: F) {
    let rows = f(); // warmup
    let mut best = f64::MAX;
    for _ in 0..3 {
        let t = Instant::now();
        let _ = f();
        best = best.min(t.elapsed().as_secs_f64() * 1000.0);
    }
    println!("  {label:24} {best:7.1} ms   ({rows} rows)");
}

pub fn bench_adbc(driver_path: &str, uri: &str, sql: &str) {
    use adbc_core::options::{AdbcVersion, OptionDatabase, OptionValue};
    use adbc_core::{Connection, Database, Driver, Statement};

    let mut driver = match adbc_driver_manager::ManagedDriver::load_dynamic_from_filename(
        driver_path,
        None,
        AdbcVersion::V100,
    ) {
        Ok(d) => d,
        Err(e) => {
            println!("  {:24} (skipped: {e})", "adbc");
            return;
        }
    };
    let opts = [(OptionDatabase::Uri, OptionValue::String(uri.to_string()))];
    let database = driver.new_database_with_opts(opts).expect("adbc database");
    let mut conn = database.new_connection().expect("adbc connection");

    time_best("adbc", || {
        let mut stmt = conn.new_statement().expect("adbc statement");
        stmt.set_sql_query(sql).expect("adbc set sql");
        let mut reader = stmt.execute().expect("adbc execute");
        let mut rows = 0usize;
        while let Some(batch) = reader.next() {
            rows += batch.expect("adbc batch").num_rows();
        }
        rows
    });
}

pub fn bench_connectorx(uri: &str, sql: &str) {
    use connectorx::prelude::*;

    let source_conn = match SourceConn::try_from(uri) {
        Ok(s) => s,
        Err(e) => {
            println!("  {:24} (skipped: {e})", "connectorx");
            return;
        }
    };
    time_best("connectorx", || {
        let queries = [CXQuery::naked(sql)];
        let dest = get_arrow(&source_conn, None, &queries, None).expect("connectorx get_arrow");
        let batches = dest.arrow().expect("connectorx arrow");
        batches.iter().map(|b| b.num_rows()).sum()
    });
}
