//! `miden-assembly-syntax`-backed implementation of the minimal frontend traits.

use std::sync::Arc;

use miden_assembly_syntax::{ast::Module, debuginfo::SourceManager};

use super::{
    body::build_body,
    procedure::build_procedure_metadata,
    AnalysisBody,
    AnalysisFrontend,
    AnalysisProcedure,
    ProcedureMetadata,
};

/// Analysis frontend backed directly by parsed MASM syntax modules.
///
/// This backend intentionally implements only the minimal structural contract. Parsed syntax plus
/// local semantic analysis is enough to normalize bodies, locals, and raw invocation targets, but
/// it cannot always distinguish external procedure aliases from module aliases once the referenced
/// dependency is outside the loaded module snapshot.
///
/// A future linker-backed backend can implement [`super::PreciseAnalysisFrontend`] by loading the
/// visible libraries into `miden-vm`'s linker and translating exact symbol resolutions into
/// concrete callee summary keys while still reusing this minimal body normalization layer.
#[derive(Debug, Clone)]
pub struct SyntaxAnalysisFrontend<'a> {
    modules: Vec<&'a Module>,
    source_manager: Arc<dyn SourceManager>,
}

impl<'a> SyntaxAnalysisFrontend<'a> {
    /// Create a new syntax-backed frontend over parsed MASM modules.
    pub fn new<I>(modules: I, source_manager: Arc<dyn SourceManager>) -> Self
    where
        I: IntoIterator<Item = &'a Module>,
    {
        Self {
            modules: modules.into_iter().collect(),
            source_manager,
        }
    }

    /// Return the syntax modules visible through this frontend.
    pub fn modules(&self) -> &[&'a Module] {
        &self.modules
    }

    fn build_procedure(
        &self,
        module: &'a Module,
        procedure: &'a miden_assembly_syntax::ast::Procedure,
    ) -> SyntaxProcedure<'a> {
        let body = build_body(module, self.source_manager.clone(), procedure.body());
        let metadata = build_procedure_metadata(module, self.source_manager.clone(), procedure);

        SyntaxProcedure {
            procedure,
            metadata,
            body,
        }
    }
}

impl AnalysisFrontend for SyntaxAnalysisFrontend<'_> {
    type Procedure<'a>
        = SyntaxProcedure<'a>
    where
        Self: 'a;

    fn procedures(&self) -> Vec<Self::Procedure<'_>> {
        self.modules
            .iter()
            .flat_map(|module| {
                module
                    .procedures()
                    .map(move |procedure| self.build_procedure(module, procedure))
            })
            .collect()
    }
}

/// Analysis-visible procedure backed directly by a parsed syntax procedure.
#[derive(Debug)]
pub struct SyntaxProcedure<'a> {
    procedure: &'a miden_assembly_syntax::ast::Procedure,
    metadata: ProcedureMetadata,
    body: AnalysisBody,
}

impl<'a> SyntaxProcedure<'a> {
    /// Return the underlying parsed MASM procedure.
    pub fn procedure(&self) -> &'a miden_assembly_syntax::ast::Procedure {
        self.procedure
    }
}

impl AnalysisProcedure for SyntaxProcedure<'_> {
    fn metadata(&self) -> &ProcedureMetadata {
        &self.metadata
    }

    fn body(&self) -> &AnalysisBody {
        &self.body
    }
}

#[cfg(test)]
mod tests {
    use masm_decompiler::frontend::testing::workspace_from_modules;

    use crate::{
        analysis_frontend::{AnalysisFrontend, AnalysisProcedure},
        StackSignature,
    };

    use super::SyntaxAnalysisFrontend;

    #[test]
    fn syntax_frontend_uses_declared_signature_and_locals_metadata() {
        let workspace = workspace_from_modules(&[(
            "math::word_ops",
            "@locals(8)\npub proc widen(input: word, flag: i1) -> (u32, word)\n    push.1\nend\n",
        )]);
        let modules: Vec<_> = workspace
            .modules()
            .map(|program| program.module())
            .collect();
        let frontend = SyntaxAnalysisFrontend::new(modules, workspace.source_manager());

        let procedures = frontend.procedures();
        let procedure = procedures.first().expect("procedure");
        let metadata = procedure.metadata();

        assert_eq!(
            metadata.stack_signature(),
            Some(StackSignature {
                inputs: 5,
                outputs: 5,
            })
        );
        assert_eq!(metadata.locals(), 8);
        assert_eq!(metadata.summary_key().as_str(), "math::word_ops::widen");
    }

    #[test]
    fn syntax_frontend_exposes_minimal_invocation_shape_without_precise_resolution() {
        let workspace = workspace_from_modules(&[(
            "app::main",
            "use ::pkg::math::add->plus\nproc caller\n    exec.plus\nend\n",
        )]);
        let modules: Vec<_> = workspace
            .modules()
            .map(|program| program.module())
            .collect();
        let frontend = SyntaxAnalysisFrontend::new(modules, workspace.source_manager());

        let procedures = frontend.procedures();
        let procedure = procedures.first().expect("procedure");
        let invocation = match procedure.body().ops().first() {
            Some(super::super::AnalysisOp::Inst(instruction)) => {
                instruction.invocation().expect("invocation")
            }
            _ => panic!("expected invocation"),
        };

        assert!(matches!(
            invocation.target(),
            super::super::AnalysisInvocationTarget::Symbol(target) if target == "plus"
        ));
    }
}
