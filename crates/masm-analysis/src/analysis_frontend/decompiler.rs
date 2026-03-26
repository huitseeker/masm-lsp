//! `masm-decompiler`-backed implementation of the analysis frontend traits.

use masm_decompiler::frontend::Workspace;
use std::{collections::HashSet, sync::Arc};

use super::{
    body::{build_body, collect_precise_invocations, KnownTargets, ResolvedInvocation},
    procedure::build_procedure_metadata,
    summary_key_for_name, AnalysisBody, AnalysisFrontend, AnalysisProcedure,
    PreciseAnalysisFrontend, PreciseAnalysisProcedure, ProcedureMetadata,
};

/// Analysis frontend backed by a `masm-decompiler` workspace.
#[derive(Debug)]
pub struct DecompilerAnalysisFrontend<'a> {
    workspace: &'a Workspace,
}

impl<'a> DecompilerAnalysisFrontend<'a> {
    /// Create a new analysis frontend view over a decompiler workspace.
    pub fn new(workspace: &'a Workspace) -> Self {
        Self { workspace }
    }

    /// Build a normalized frontend-visible procedure from the workspace syntax tree.
    fn build_procedure(
        &self,
        module: &miden_assembly_syntax::ast::Module,
        procedure: &'a miden_assembly_syntax::ast::Procedure,
        known_targets: &KnownTargets,
    ) -> DecompilerProcedure<'a> {
        let body = build_body(module, self.workspace.source_manager(), procedure.body());
        let metadata = build_procedure_metadata(module, self.workspace.source_manager(), procedure);
        let resolved_invocations = collect_precise_invocations(
            module,
            self.workspace.source_manager(),
            procedure.body(),
            known_targets,
            body.id(),
        );

        DecompilerProcedure {
            procedure,
            metadata,
            body,
            resolved_invocations,
        }
    }
}

impl AnalysisFrontend for DecompilerAnalysisFrontend<'_> {
    type Procedure<'a>
        = DecompilerProcedure<'a>
    where
        Self: 'a;

    fn procedures(&self) -> Vec<Self::Procedure<'_>> {
        let known_targets = Arc::new(workspace_known_targets(self.workspace));
        self.workspace
            .modules()
            .flat_map(move |program| {
                let module = program.module();
                let known_targets = known_targets.clone();
                program.procedures().map(move |procedure| {
                    self.build_procedure(module, procedure, known_targets.as_ref())
                })
            })
            .collect()
    }
}

/// Analysis-visible procedure backed by a `masm-decompiler` AST procedure.
#[derive(Debug)]
pub struct DecompilerProcedure<'a> {
    procedure: &'a miden_assembly_syntax::ast::Procedure,
    metadata: ProcedureMetadata,
    body: AnalysisBody,
    resolved_invocations: Vec<ResolvedInvocation>,
}

impl<'a> DecompilerProcedure<'a> {
    /// Return the underlying syntax procedure.
    pub fn procedure(&self) -> &'a miden_assembly_syntax::ast::Procedure {
        self.procedure
    }
}

impl AnalysisProcedure for DecompilerProcedure<'_> {
    fn metadata(&self) -> &ProcedureMetadata {
        &self.metadata
    }

    fn body(&self) -> &AnalysisBody {
        &self.body
    }
}

impl PreciseAnalysisProcedure for DecompilerProcedure<'_> {
    fn resolved_summary_key(
        &self,
        invocation: &super::AnalysisInvocation,
    ) -> Option<&super::SummaryKey> {
        self.resolved_invocations
            .iter()
            .find(|resolved| {
                resolved.id() == invocation.id()
                    && resolved.span() == invocation.span()
                    && resolved.body_id() == invocation.body_id()
            })
            .map(|resolved| resolved.summary_key())
    }
}

impl PreciseAnalysisFrontend for DecompilerAnalysisFrontend<'_> {
    type PreciseProcedure<'a>
        = DecompilerProcedure<'a>
    where
        Self: 'a;

    fn precise_procedures(&self) -> Vec<Self::PreciseProcedure<'_>> {
        <Self as AnalysisFrontend>::procedures(self)
    }
}

fn workspace_known_targets(workspace: &Workspace) -> KnownTargets {
    let (procedures, modules) = workspace.modules().fold(
        (HashSet::new(), HashSet::new()),
        |(mut procedures, mut modules), program| {
            let module_path =
                <miden_assembly_syntax::ast::Path as AsRef<str>>::as_ref(program.module().path())
                    .to_string();
            modules.insert(module_path.clone());
            procedures.extend(program.procedures().map(|procedure| {
                summary_key_for_name(&module_path, procedure.name().as_str())
                    .as_str()
                    .to_string()
            }));
            (procedures, modules)
        },
    );

    KnownTargets::new(procedures, modules)
}

#[cfg(test)]
mod tests {
    use masm_decompiler::frontend::testing::workspace_from_modules;

    use crate::{
        analysis_frontend::{
            AnalysisFrontend, AnalysisInvocationKind, AnalysisInvocationTarget,
            AnalysisLocalAccess, AnalysisLocalLane, AnalysisOp, AnalysisProcedure,
            PreciseAnalysisFrontend, PreciseAnalysisProcedure,
        },
        StackSignature,
    };

    use super::DecompilerAnalysisFrontend;

    #[test]
    fn decompiler_frontend_exposes_fully_qualified_summary_keys() {
        let workspace = workspace_from_modules(&[
            ("math::u32_ops", "proc add\n    push.1\nend\n"),
            ("math::word_ops", "proc reverse\n    push.1\nend\n"),
        ]);

        let frontend = DecompilerAnalysisFrontend::new(&workspace);
        let summary_keys: Vec<_> = frontend
            .procedures()
            .into_iter()
            .map(|procedure| procedure.metadata().summary_key().as_str().to_string())
            .collect();

        assert_eq!(
            summary_keys,
            vec![
                "math::u32_ops::add".to_string(),
                "math::word_ops::reverse".to_string(),
            ]
        );
    }

    #[test]
    fn decompiler_frontend_uses_declared_signature_and_locals_metadata() {
        let workspace = workspace_from_modules(&[(
            "math::word_ops",
            "@locals(8)\npub proc widen(input: word, flag: i1) -> (u32, word)\n    push.1\nend\n",
        )]);

        let frontend = DecompilerAnalysisFrontend::new(&workspace);
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
        assert_eq!(metadata.inputs(), Some(5));
        assert_eq!(metadata.outputs(), Some(5));
        assert_eq!(metadata.locals(), 8);
        assert_eq!(metadata.summary_key().as_str(), "math::word_ops::widen");
    }

    #[test]
    fn decompiler_frontend_preserves_unknown_stack_signature_metadata() {
        let workspace = workspace_from_modules(&[(
            "math::word_ops",
            "pub proc widen(input: UnknownWord) -> word\n    push.1\nend\n",
        )]);

        let frontend = DecompilerAnalysisFrontend::new(&workspace);
        let procedures = frontend.procedures();
        let procedure = procedures.first().expect("procedure");
        let metadata = procedure.metadata();

        assert_eq!(metadata.stack_signature(), None);
        assert_eq!(metadata.inputs(), None);
        assert_eq!(metadata.outputs(), None);
        assert_eq!(metadata.summary_key().as_str(), "math::word_ops::widen");
    }

    #[test]
    fn decompiler_frontend_normalizes_control_flow_and_instruction_shapes() {
        let workspace = workspace_from_modules(&[(
            "math::word_ops",
            "@locals(4)\nproc inspect\n    locaddr.0\n    if.true\n        exec.foo\n    else\n        repeat.2\n            loc_storew_be.0\n        end\n    end\n    while.true\n        loc_loadw_le.0\n    end\nend\n",
        )]);

        let frontend = DecompilerAnalysisFrontend::new(&workspace);
        let procedures = frontend.procedures();
        let procedure = procedures.first().expect("procedure");
        let body = procedure.body();

        assert_eq!(body.len(), 3);
        assert!(matches!(
            body.ops().first(),
            Some(AnalysisOp::Inst(instruction))
                if matches!(
                    instruction.local_access(),
                    Some(AnalysisLocalAccess::Address { index: 0, .. })
                )
        ));
        assert!(matches!(
            body.ops().get(1),
            Some(AnalysisOp::If { then_body, else_body, .. })
                if matches!(
                    then_body.ops().first(),
                    Some(AnalysisOp::Inst(instruction))
                        if matches!(
                            instruction.invocation(),
                            Some(invocation)
                                if invocation.kind() == AnalysisInvocationKind::Exec
                                    && matches!(
                                        invocation.target(),
                                        AnalysisInvocationTarget::Symbol(target)
                                            if target == "foo"
                                    )
                                    && matches!(
                                        procedure.resolved_summary_key(invocation),
                                        Some(key) if key.as_str() == "math::word_ops::foo"
                                    )
                        )
                ) && matches!(
                    else_body.ops().first(),
                    Some(AnalysisOp::Repeat { body, .. })
                        if matches!(
                            body.ops().first(),
                            Some(AnalysisOp::Inst(instruction))
                                if matches!(
                                    instruction.local_access(),
                                    Some(AnalysisLocalAccess::Store {
                                        index: 0,
                                        lane: AnalysisLocalLane::WordBigEndian,
                                        ..
                                    })
                                )
                        )
                )
        ));
        assert!(matches!(
            body.ops().get(2),
            Some(AnalysisOp::While { body, .. })
                if matches!(
                    body.ops().first(),
                    Some(AnalysisOp::Inst(instruction))
                        if matches!(
                            instruction.local_access(),
                            Some(AnalysisLocalAccess::Load {
                                index: 0,
                                lane: AnalysisLocalLane::WordLittleEndian,
                                ..
                            })
                        )
                )
        ));
    }

    #[test]
    fn decompiler_frontend_assigns_distinct_invocation_ids_in_body_order() {
        let workspace = workspace_from_modules(&[(
            "app::main",
            "proc caller\n    exec.foo\n    if.true\n        call.bar\n    else\n        syscall.baz\n    end\nend\n",
        )]);

        let frontend = DecompilerAnalysisFrontend::new(&workspace);
        let procedures = frontend.procedures();
        let procedure = procedures.first().expect("procedure");
        let mut invocation_ids = Vec::new();

        for op in procedure.body().ops() {
            match op {
                AnalysisOp::Inst(instruction) => {
                    if let Some(invocation) = instruction.invocation() {
                        invocation_ids.push(invocation.id().ordinal());
                    }
                }
                AnalysisOp::If {
                    then_body,
                    else_body,
                    ..
                } => {
                    let then_invocation = then_body
                        .ops()
                        .first()
                        .and_then(|op| match op {
                            AnalysisOp::Inst(instruction) => instruction.invocation(),
                            _ => None,
                        })
                        .expect("then invocation");
                    let else_invocation = else_body
                        .ops()
                        .first()
                        .and_then(|op| match op {
                            AnalysisOp::Inst(instruction) => instruction.invocation(),
                            _ => None,
                        })
                        .expect("else invocation");
                    invocation_ids.push(then_invocation.id().ordinal());
                    invocation_ids.push(else_invocation.id().ordinal());
                }
                _ => {}
            }
        }

        assert_eq!(invocation_ids, vec![0, 1, 2]);
    }

    #[test]
    fn decompiler_precise_frontend_does_not_cross_match_procedures() {
        let workspace = workspace_from_modules(&[
            (
                "alpha",
                "proc callee\n    push.1\nend\nproc caller\n    call.callee\nend\n",
            ),
            (
                "beta",
                "proc callee\n    push.1\nend\nproc caller\n    call.callee\nend\n",
            ),
        ]);

        let frontend = DecompilerAnalysisFrontend::new(&workspace);
        let mut callers: Vec<_> = frontend
            .procedures()
            .into_iter()
            .filter(|procedure| {
                procedure
                    .metadata()
                    .summary_key()
                    .as_str()
                    .ends_with("::caller")
            })
            .collect();
        assert_eq!(callers.len(), 2);

        let first_caller = callers.remove(0);
        let second_caller = callers.remove(0);
        let first_invocation = first_caller
            .body()
            .ops()
            .first()
            .and_then(|op| match op {
                AnalysisOp::Inst(instruction) => instruction.invocation().cloned(),
                _ => None,
            })
            .expect("first invocation");
        let second_invocation = second_caller
            .body()
            .ops()
            .first()
            .and_then(|op| match op {
                AnalysisOp::Inst(instruction) => instruction.invocation().cloned(),
                _ => None,
            })
            .expect("second invocation");

        assert_eq!(
            first_caller
                .resolved_summary_key(&first_invocation)
                .map(|key| key.as_str()),
            Some("alpha::callee")
        );
        assert_eq!(
            second_caller
                .resolved_summary_key(&second_invocation)
                .map(|key| key.as_str()),
            Some("beta::callee")
        );
        assert_eq!(first_caller.resolved_summary_key(&second_invocation), None);
        assert_eq!(second_caller.resolved_summary_key(&first_invocation), None);
    }

    #[test]
    fn decompiler_precise_frontend_leaves_external_dependency_aliases_opaque() {
        let workspace = workspace_from_modules(&[(
            "app::main",
            "use ::math::add->plus\nproc caller\n    exec.plus\nend\n",
        )]);

        let frontend = DecompilerAnalysisFrontend::new(&workspace);
        let procedures = frontend.precise_procedures();
        let procedure = procedures.first().expect("procedure");
        let invocation = match procedure.body().ops().first() {
            Some(AnalysisOp::Inst(instruction)) => instruction.invocation().expect("invocation"),
            _ => panic!("expected root invocation"),
        };

        assert_eq!(procedure.resolved_summary_key(invocation), None);
    }

    #[test]
    fn decompiler_precise_frontend_resolves_dependency_procedure_alias_when_present() {
        let workspace = workspace_from_modules(&[
            ("pkg::math", "pub proc add\n    push.1\nend\n"),
            (
                "app::main",
                "use ::math::add->plus\nproc caller\n    exec.plus\nend\n",
            ),
        ]);

        let frontend = DecompilerAnalysisFrontend::new(&workspace);
        let procedures = frontend.precise_procedures();
        let procedure = procedures.last().expect("procedure");
        let invocation = procedure
            .body()
            .ops()
            .first()
            .and_then(|op| match op {
                AnalysisOp::Inst(instruction) => instruction.invocation(),
                _ => None,
            })
            .expect("invocation");

        assert_eq!(
            procedure
                .resolved_summary_key(invocation)
                .map(|key| key.as_str()),
            Some("pkg::math::add")
        );
    }
}
