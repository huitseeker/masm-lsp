//! Shared helpers for frontend-visible MASM procedures.

use std::sync::Arc;

use miden_assembly_syntax::{
    ast::{
        FunctionType, Module, Procedure, SymbolResolutionError, TypeResolver, types::Type as AstType,
    },
    debuginfo::SourceManager,
};
use miden_debug_types::Spanned;

use super::{ProcedureMetadata, SummaryKey};
use crate::StackSignature;

/// Build normalized procedure metadata for a frontend-visible procedure.
pub(crate) fn build_procedure_metadata(
    module: &Module,
    source_manager: Arc<dyn SourceManager>,
    procedure: &Procedure,
) -> ProcedureMetadata {
    let stack_signature =
        declared_stack_signature(module, source_manager, procedure.signature());
    ProcedureMetadata::new(
        summary_key_for_name(module_path(module), procedure.name().as_str()),
        stack_signature,
        procedure.num_locals(),
        procedure.span(),
    )
}

/// Build a stable summary key from a fully-qualified module path and local procedure name.
pub(crate) fn summary_key_for_name(module_path: &str, procedure_name: &str) -> SummaryKey {
    SummaryKey::new(format!("{module_path}::{procedure_name}"))
}

/// Compute stack-effect counts from a declared procedure signature.
pub(crate) fn declared_stack_signature(
    module: &Module,
    source_manager: Arc<dyn SourceManager>,
    signature: Option<&FunctionType>,
) -> Option<StackSignature> {
    let signature = signature?;
    let resolver = module.type_resolver(source_manager);

    stack_signature(signature, &resolver)
}

/// Resolve a function signature into flat MASM input/output felt counts.
fn stack_signature<R>(signature: &FunctionType, resolver: &R) -> Option<StackSignature>
where
    R: TypeResolver<SymbolResolutionError>,
{
    let inputs = signature
        .args
        .iter()
        .try_fold(0usize, |count, arg| resolved_felts(arg.resolve_type(resolver), count))?;
    let outputs = signature
        .results
        .iter()
        .try_fold(0usize, |count, result| {
            resolved_felts(result.resolve_type(resolver), count)
        })?;

    Some(StackSignature { inputs, outputs })
}

/// Add the felt width of a resolved type to an accumulated stack count.
fn resolved_felts(
    ty: Result<Option<AstType>, SymbolResolutionError>,
    count: usize,
) -> Option<usize> {
    let ty = ty.ok().flatten()?;
    let felts = match ty {
        AstType::Unknown | AstType::Never | AstType::List(_) => return None,
        _ => ty.size_in_felts(),
    };

    count.checked_add(felts)
}

fn module_path(module: &Module) -> &str {
    module.path().as_ref()
}
