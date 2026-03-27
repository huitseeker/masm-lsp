//! Interprocedural analysis for unconstrained advice reaching U32 and non-zero sinks.

mod address;
mod domain;
mod inter;
mod merkle;
mod nonzero;
mod provenance;
mod summary;
mod state;
mod transfer;
mod u32;
mod walker;

#[cfg(test)]
mod corpus;

#[cfg(test)]
mod tests;

pub use inter::{infer_unconstrained_advice, infer_unconstrained_advice_in_workspace};
pub use summary::{
    AdviceDiagnostic, AdviceDiagnosticsMap, AdviceSinkKind, AdviceSummary, AdviceSummaryMap,
};
