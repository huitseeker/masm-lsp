//! Interprocedural analysis for unconstrained advice reaching U32 and non-zero sinks.

mod address;
mod domain;
mod grouping;
mod inter;
mod merkle;
mod nonzero;
mod provenance;
mod summary;
mod state;
mod transfer;
mod u32_domain;
mod u32;
mod walker;

#[cfg(test)]
mod corpus;

#[cfg(test)]
mod stdlib_eval;

#[cfg(test)]
mod tests;

pub use inter::{infer_unconstrained_advice, infer_unconstrained_advice_in_workspace};
pub use grouping::{AdviceRootCauseGroup, group_advice_diagnostics_by_origin};
pub use summary::{
    AdviceDiagnostic, AdviceDiagnosticsMap, AdviceSinkKind, AdviceSummary, AdviceSummaryMap,
    CallArgumentRequirement,
};
pub use u32_domain::U32Validity;
