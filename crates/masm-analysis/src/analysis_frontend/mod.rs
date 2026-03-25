//! Trait-based frontend interfaces for MASM analysis inputs.
//!
//! The split in this module is intentional:
//! - [`AnalysisFrontend`] is the minimal structural contract that an abstract interpreter can use
//!   to walk procedures, blocks, instructions, and raw invocation sites.
//! - [`PreciseAnalysisFrontend`] is an optional refinement for backends that can classify some
//!   invocation targets as concrete procedures instead of leaving them as raw syntax.
//!
//! This keeps the abstract interpreter architecture didactic. The transfer loop can be written in
//! terms of the minimal contract first, and analyses that need stronger interprocedural callee
//! identities can opt into the richer capability explicitly.

use miden_debug_types::SourceSpan;

use crate::StackSignature;

mod body;
mod decompiler;
mod procedure;
mod syntax;

pub use body::{
    AnalysisBody, AnalysisInstruction, AnalysisInvocation, AnalysisInvocationKind,
    AnalysisInvocationTarget, AnalysisLocalAccess, AnalysisLocalLane, AnalysisOp,
};

pub use decompiler::{DecompilerAnalysisFrontend, DecompilerProcedure};
pub(crate) use procedure::summary_key_for_name;
pub use syntax::{SyntaxAnalysisFrontend, SyntaxProcedure};

/// Stable key used to identify a procedure summary across analysis passes.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SummaryKey(String);

impl SummaryKey {
    /// Create a summary key from its canonical string form.
    pub fn new(key: impl Into<String>) -> Self {
        Self(key.into())
    }

    /// Return the canonical string form of this summary key.
    pub fn as_str(&self) -> &str {
        self.0.as_str()
    }
}

/// Metadata about one procedure as exposed to analysis frontends.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProcedureMetadata {
    summary_key: SummaryKey,
    stack_signature: Option<StackSignature>,
    locals: u16,
    span: SourceSpan,
}

impl ProcedureMetadata {
    /// Create procedure metadata for one analysis-visible procedure.
    pub fn new(
        summary_key: SummaryKey,
        stack_signature: Option<StackSignature>,
        locals: u16,
        span: SourceSpan,
    ) -> Self {
        Self {
            summary_key,
            stack_signature,
            locals,
            span,
        }
    }

    /// Return the declared stack signature of the procedure, if known.
    pub fn stack_signature(&self) -> Option<StackSignature> {
        self.stack_signature
    }

    /// Return the summary key of the procedure.
    pub fn summary_key(&self) -> &SummaryKey {
        &self.summary_key
    }

    /// Return the number of stack inputs declared for the procedure, if known.
    pub fn inputs(&self) -> Option<usize> {
        self.stack_signature.map(|signature| signature.inputs)
    }

    /// Return the number of stack outputs declared for the procedure, if known.
    pub fn outputs(&self) -> Option<usize> {
        self.stack_signature.map(|signature| signature.outputs)
    }

    /// Return the number of declared locals for the procedure.
    pub fn locals(&self) -> u16 {
        self.locals
    }

    /// Return the source span covering the procedure definition.
    pub fn span(&self) -> SourceSpan {
        self.span
    }
}

/// Analysis-facing view of a MASM procedure.
pub trait AnalysisProcedure {
    /// Return metadata describing the procedure.
    fn metadata(&self) -> &ProcedureMetadata;

    /// Return the normalized procedure body exposed by this frontend.
    fn body(&self) -> &AnalysisBody;
}

/// Minimal structural frontend contract shared by all MASM analysis inputs.
///
/// Backends implementing this trait only promise normalized procedure bodies and metadata. They do
/// not promise precise callable classification for invocation targets.
pub trait AnalysisFrontend {
    /// The procedure handle type exposed by this frontend.
    type Procedure<'a>: AnalysisProcedure
    where
        Self: 'a;

    /// Return all procedures currently visible to this frontend.
    fn procedures(&self) -> Vec<Self::Procedure<'_>>;
}

/// Richer procedure contract for backends that can classify invocation targets precisely.
///
/// Backends implementing this trait refine the minimal structural view with callee identities for
/// invocation sites they can classify as concrete procedures.
pub trait PreciseAnalysisProcedure: AnalysisProcedure {
    /// Return the resolved callee summary key for `invocation`, if this backend can classify it as
    /// a concrete procedure.
    fn resolved_summary_key(&self, invocation: &AnalysisInvocation) -> Option<&SummaryKey>;
}

/// Optional frontend capability for backends that can resolve invocations more precisely than the
/// minimal structural contract.
///
/// Abstract interpretation code should depend on this trait only when it truly needs concrete
/// procedure identities. Analyses that can operate on raw invocation shapes should stay on
/// [`AnalysisFrontend`] to remain portable across minimal and precise backends.
pub trait PreciseAnalysisFrontend: AnalysisFrontend {
    /// The precise procedure handle type exposed by this frontend.
    type PreciseProcedure<'a>: PreciseAnalysisProcedure
    where
        Self: 'a;

    /// Return all procedures currently visible to this precise frontend.
    fn precise_procedures(&self) -> Vec<Self::PreciseProcedure<'_>>;
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        path::{Path, PathBuf},
    };

    use masm_decompiler::frontend::testing::workspace_from_modules;
    use miden_debug_types::SourceSpan;

    use super::{
        AnalysisBody, AnalysisFrontend, AnalysisInvocation, AnalysisInvocationTarget, AnalysisOp,
        AnalysisProcedure, DecompilerAnalysisFrontend, PreciseAnalysisFrontend,
        PreciseAnalysisProcedure, ProcedureMetadata, SummaryKey, SyntaxAnalysisFrontend,
    };
    use crate::StackSignature;

    #[test]
    fn summary_key_round_trips() {
        let key = SummaryKey::new("miden::math::add");
        assert_eq!(key.as_str(), "miden::math::add");
    }

    #[test]
    fn procedure_metadata_accessors_return_constructor_values() {
        let key = SummaryKey::new("miden::math::add");
        let metadata = ProcedureMetadata::new(
            key.clone(),
            Some(StackSignature {
                inputs: 2,
                outputs: 1,
            }),
            4,
            SourceSpan::UNKNOWN,
        );

        assert_eq!(metadata.summary_key(), &key);
        assert_eq!(
            metadata.stack_signature(),
            Some(StackSignature {
                inputs: 2,
                outputs: 1,
            })
        );
        assert_eq!(metadata.inputs(), Some(2));
        assert_eq!(metadata.outputs(), Some(1));
        assert_eq!(metadata.locals(), 4);
        assert_eq!(metadata.span(), SourceSpan::UNKNOWN);
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct ProcedureSnapshot {
        metadata: ProcedureMetadata,
        body: AnalysisBody,
    }

    fn snapshots_for<F>(frontend: &F) -> Vec<ProcedureSnapshot>
    where
        F: AnalysisFrontend,
    {
        frontend
            .procedures()
            .into_iter()
            .map(|procedure| ProcedureSnapshot {
                metadata: procedure.metadata().clone(),
                body: procedure.body().clone(),
            })
            .collect()
    }

    fn decompiler_snapshots(mods: &[(&str, &str)]) -> Vec<ProcedureSnapshot> {
        let workspace = workspace_from_modules(mods);
        let frontend = DecompilerAnalysisFrontend::new(&workspace);
        snapshots_for(&frontend)
    }

    fn syntax_snapshots(mods: &[(&str, &str)]) -> Vec<ProcedureSnapshot> {
        let workspace = workspace_from_modules(mods);
        let modules: Vec<_> = workspace
            .modules()
            .map(|program| program.module())
            .collect();
        let frontend = SyntaxAnalysisFrontend::new(modules, workspace.source_manager());
        snapshots_for(&frontend)
    }

    fn first_invocation(snapshot: &ProcedureSnapshot) -> Option<&AnalysisInvocation> {
        match snapshot.body.ops().first() {
            Some(AnalysisOp::Inst(instruction)) => instruction.invocation(),
            _ => None,
        }
    }

    fn core_examples_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("examples")
            .join("core")
    }

    fn collect_core_example_files(root: &Path, files: &mut Vec<PathBuf>) {
        let entries = fs::read_dir(root).expect("read core example directory");
        for entry in entries {
            let entry = entry.expect("read core example entry");
            let path = entry.path();
            if path.is_dir() {
                collect_core_example_files(&path, files);
                continue;
            }

            let is_masm = path
                .extension()
                .and_then(|ext| ext.to_str())
                .is_some_and(|ext| ext.eq_ignore_ascii_case("masm"));
            if is_masm {
                files.push(path);
            }
        }
    }

    fn module_name_for_core_example(path: &Path) -> String {
        let relative = path
            .strip_prefix(core_examples_dir())
            .expect("core example file should be under examples/core");
        let mut segments = vec!["miden".to_string(), "core".to_string()];
        let stem = relative
            .file_stem()
            .and_then(|stem| stem.to_str())
            .expect("example file stem should be valid utf-8");

        if let Some(parent) = relative.parent() {
            segments.extend(
                parent
                    .components()
                    .map(|component| component.as_os_str().to_string_lossy().to_string()),
            );
        }

        if stem != "mod" {
            segments.push(stem.to_string());
        }

        segments.join("::")
    }

    fn core_example_modules() -> Vec<(String, String)> {
        let mut files = Vec::new();
        collect_core_example_files(&core_examples_dir(), &mut files);
        files.sort();

        files.into_iter()
            .map(|path| {
                (
                    module_name_for_core_example(&path),
                    fs::read_to_string(&path).expect("read core example"),
                )
            })
            .collect()
    }

    #[test]
    fn syntax_and_decompiler_frontends_match_for_shared_metadata_and_body_shape() {
        let modules = &[
            ("math::helpers", "pub proc callee\n    push.1\nend\n"),
            (
                "app::main",
                "use math::helpers\n@locals(8)\npub proc entry(input: word, flag: i1) -> word\n    locaddr.0\n    if.true\n        exec.helpers::callee\n    else\n        repeat.2\n            loc_storew_be.0\n        end\n    end\n    while.true\n        call.helpers::callee\n        loc_loadw_le.0\n    end\nend\n",
            ),
        ];

        assert_eq!(decompiler_snapshots(modules), syntax_snapshots(modules));
    }

    #[test]
    fn syntax_and_decompiler_frontends_match_on_core_example_corpus() {
        let modules = core_example_modules();
        assert!(
            !modules.is_empty(),
            "expected at least one core example module"
        );
        let module_refs: Vec<_> = modules
            .iter()
            .map(|(name, source)| (name.as_str(), source.as_str()))
            .collect();

        assert_eq!(
            decompiler_snapshots(&module_refs),
            syntax_snapshots(&module_refs)
        );
    }

    #[test]
    fn syntax_frontend_extraction_is_deterministic_on_core_example_corpus() {
        let modules = core_example_modules();
        assert!(
            !modules.is_empty(),
            "expected at least one core example module"
        );
        let module_refs: Vec<_> = modules
            .iter()
            .map(|(name, source)| (name.as_str(), source.as_str()))
            .collect();

        assert_eq!(syntax_snapshots(&module_refs), syntax_snapshots(&module_refs));
    }

    #[test]
    fn syntax_and_decompiler_frontends_preserve_unknown_stack_signatures_equally() {
        let modules = &[(
            "app::main",
            "pub proc entry(input: UnknownWord) -> word\n    push.1\nend\n",
        )];

        assert_eq!(decompiler_snapshots(modules), syntax_snapshots(modules));
    }

    #[test]
    fn decompiler_precise_frontend_resolves_aliased_symbol_invocations() {
        let modules = &[
            ("tmp::aliases", "pub proc add\n    push.1\nend\n"),
            (
                "app::main",
                "use ::tmp::aliases::add->plus\nproc caller\n    exec.plus\nend\n",
            ),
        ];

        let workspace = workspace_from_modules(modules);
        let frontend = DecompilerAnalysisFrontend::new(&workspace);
        let procedures = frontend.precise_procedures();
        let procedure = procedures.last().expect("caller procedure");
        let snapshot = ProcedureSnapshot {
            metadata: procedure.metadata().clone(),
            body: procedure.body().clone(),
        };
        let invocation = first_invocation(&snapshot).expect("invocation");

        assert_eq!(
            procedure
                .resolved_summary_key(invocation)
                .map(|key| key.as_str()),
            Some("tmp::aliases::add")
        );
        assert!(matches!(
            invocation.target(),
            AnalysisInvocationTarget::Symbol(target) if target == "plus"
        ));
    }

    #[test]
    fn decompiler_precise_frontend_resolves_dependency_procedure_alias_in_workspace() {
        let modules = &[
            ("pkg::math", "pub proc add\n    push.1\nend\n"),
            (
                "app::main",
                "use ::math::add->plus\nproc caller\n    exec.plus\nend\n",
            ),
        ];

        let workspace = workspace_from_modules(modules);
        let frontend = DecompilerAnalysisFrontend::new(&workspace);
        let procedures = frontend.precise_procedures();
        let procedure = procedures.last().expect("caller procedure");
        let snapshot = ProcedureSnapshot {
            metadata: procedure.metadata().clone(),
            body: procedure.body().clone(),
        };
        let invocation = first_invocation(&snapshot).expect("invocation");

        assert_eq!(
            procedure
                .resolved_summary_key(invocation)
                .map(|key| key.as_str()),
            Some("pkg::math::add")
        );
        assert!(matches!(
            invocation.target(),
            AnalysisInvocationTarget::Symbol(target) if target == "plus"
        ));
    }

    #[test]
    fn decompiler_precise_frontend_preserves_unresolved_qualified_path_targets() {
        let modules = &[(
            "app::main",
            "proc caller\n    exec.utils::target_proc\nend\n",
        )];

        let workspace = workspace_from_modules(modules);
        let frontend = DecompilerAnalysisFrontend::new(&workspace);
        let procedures = frontend.precise_procedures();
        let procedure = procedures.first().expect("caller procedure");
        let snapshot = ProcedureSnapshot {
            metadata: procedure.metadata().clone(),
            body: procedure.body().clone(),
        };
        let invocation = first_invocation(&snapshot).expect("invocation");

        assert_eq!(
            procedure
                .resolved_summary_key(invocation)
                .map(|key| key.as_str()),
            Some("utils::target_proc")
        );
        assert!(matches!(
            invocation.target(),
            AnalysisInvocationTarget::Path(target) if target == "utils::target_proc"
        ));
    }

    #[test]
    fn decompiler_precise_frontend_ignores_absolute_module_targets() {
        let modules = &[("app::main", "proc caller\n    exec.::math::helpers\nend\n")];

        let workspace = workspace_from_modules(modules);
        let frontend = DecompilerAnalysisFrontend::new(&workspace);
        let procedures = frontend.precise_procedures();
        let procedure = procedures.first().expect("caller procedure");
        let snapshot = ProcedureSnapshot {
            metadata: procedure.metadata().clone(),
            body: procedure.body().clone(),
        };
        let invocation = first_invocation(&snapshot).expect("invocation");

        assert_eq!(procedure.resolved_summary_key(invocation), None);
        assert!(matches!(
            invocation.target(),
            AnalysisInvocationTarget::Path(target) if target == "::math::helpers"
        ));
    }

    #[test]
    fn decompiler_precise_frontend_leaves_unresolved_symbol_targets_without_summary_keys() {
        let modules = &[("app::main", "proc caller\n    exec.missing\nend\n")];

        let workspace = workspace_from_modules(modules);
        let frontend = DecompilerAnalysisFrontend::new(&workspace);
        let procedures = frontend.precise_procedures();
        let procedure = procedures.first().expect("caller procedure");
        let snapshot = ProcedureSnapshot {
            metadata: procedure.metadata().clone(),
            body: procedure.body().clone(),
        };
        let invocation = first_invocation(&snapshot).expect("invocation");

        assert_eq!(procedure.resolved_summary_key(invocation), None);
        assert!(matches!(
            invocation.target(),
            AnalysisInvocationTarget::Symbol(target) if target == "missing"
        ));
    }

    #[test]
    fn decompiler_precise_frontend_leaves_local_non_procedure_targets_without_summary_keys() {
        let modules = &[(
            "app::main",
            "const MISSING = 42\nproc caller\n    exec.MISSING\nend\n",
        )];

        let workspace = workspace_from_modules(modules);
        let frontend = DecompilerAnalysisFrontend::new(&workspace);
        let procedures = frontend.precise_procedures();
        let procedure = procedures.first().expect("caller procedure");
        let snapshot = ProcedureSnapshot {
            metadata: procedure.metadata().clone(),
            body: procedure.body().clone(),
        };
        let invocation = first_invocation(&snapshot).expect("invocation");

        assert_eq!(procedure.resolved_summary_key(invocation), None);
        assert!(matches!(
            invocation.target(),
            AnalysisInvocationTarget::Symbol(target) if target == "MISSING"
        ));
    }

    #[test]
    fn decompiler_precise_frontend_leaves_module_symbol_targets_without_summary_keys() {
        let modules = &[
            ("math::helpers", "pub proc callee\n    push.1\nend\n"),
            (
                "app::main",
                "use math::helpers\nproc caller\n    exec.helpers\nend\n",
            ),
        ];

        let workspace = workspace_from_modules(modules);
        let frontend = DecompilerAnalysisFrontend::new(&workspace);
        let procedures = frontend.precise_procedures();
        let procedure = procedures.last().expect("caller procedure");
        let snapshot = ProcedureSnapshot {
            metadata: procedure.metadata().clone(),
            body: procedure.body().clone(),
        };
        let invocation = first_invocation(&snapshot).expect("invocation");

        assert_eq!(procedure.resolved_summary_key(invocation), None);
        assert!(matches!(
            invocation.target(),
            AnalysisInvocationTarget::Symbol(target) if target == "helpers"
        ));
    }

    #[test]
    fn decompiler_precise_frontend_leaves_dependency_module_aliases_without_summary_keys() {
        let modules = &[(
            "app::main",
            "use ::pkg::math->m\nproc caller\n    exec.m\nend\n",
        )];

        let workspace = workspace_from_modules(modules);
        let frontend = DecompilerAnalysisFrontend::new(&workspace);
        let procedures = frontend.precise_procedures();
        let procedure = procedures.first().expect("caller procedure");
        let snapshot = ProcedureSnapshot {
            metadata: procedure.metadata().clone(),
            body: procedure.body().clone(),
        };
        let invocation = first_invocation(&snapshot).expect("invocation");

        assert_eq!(procedure.resolved_summary_key(invocation), None);
        assert!(matches!(
            invocation.target(),
            AnalysisInvocationTarget::Symbol(target) if target == "m"
        ));
    }

    #[test]
    fn decompiler_precise_frontend_leaves_external_constant_aliases_without_summary_keys() {
        let modules = &[(
            "app::main",
            "use ::pkg::math::CONST->x\nproc caller\n    exec.x\nend\n",
        )];

        let workspace = workspace_from_modules(modules);
        let frontend = DecompilerAnalysisFrontend::new(&workspace);
        let procedures = frontend.precise_procedures();
        let procedure = procedures.first().expect("caller procedure");
        let snapshot = ProcedureSnapshot {
            metadata: procedure.metadata().clone(),
            body: procedure.body().clone(),
        };
        let invocation = first_invocation(&snapshot).expect("invocation");

        assert_eq!(procedure.resolved_summary_key(invocation), None);
        assert!(matches!(
            invocation.target(),
            AnalysisInvocationTarget::Symbol(target) if target == "x"
        ));
    }
}
