//! Lab 5: prefix-cache identity, ownership, multi-replica cache-aware routing and a
//! Mooncake-trace replay under a measured prefill cost model.
#[allow(unused_imports, unused_variables)]
mod cache_student;
#[allow(unused_imports, unused_variables)]
mod cluster_student;
pub mod cost;
pub mod hash;
pub mod json;
#[allow(unused_imports, unused_variables)]
mod trace_student;
pub use cache_student::*;
pub use cluster_student::*;
pub use cost::{CostModel, FullHitRule};
pub use hash::{hash_block, ExtraKey, HashRequest, MmItem, NONE_HASH};
pub use trace_student::*;
