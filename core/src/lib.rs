//! `fedqrs-core` — the pyo3-free core of the federated execution engine.
//!
//! It holds the serializable IR, the expression-to-DataFusion translation, the
//! source-SQL emitter, and the scan-partitioning / selectivity helpers. Keeping
//! this free of pyo3 lets it build and unit-test with a plain `cargo test`; the
//! `fedqrs` crate wraps it with the Arrow FFI boundary and native connectors.

pub mod expr;
pub mod ir;
pub mod partition;
pub mod sql;
pub mod types;
