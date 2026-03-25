//! Normalized MASM procedure bodies exposed to analysis frontends.

use std::{collections::HashSet, sync::Arc};

use miden_assembly_syntax::{
    ast::{
        Block, Export, Immediate, Instruction, InvocationTarget, ItemIndex, LocalSymbolResolver,
        Module, Op, SymbolResolution, SymbolResolutionError,
    },
    debuginfo::SourceManager,
};
use miden_debug_types::{SourceSpan, Span, Spanned};

use super::{summary_key_for_name, SummaryKey};

/// Known module and procedure targets visible to the frontend.
#[derive(Clone)]
pub(crate) struct KnownTargets {
    procedures: Arc<HashSet<String>>,
    modules: Arc<HashSet<String>>,
}

impl KnownTargets {
    pub(crate) fn new(procedures: HashSet<String>, modules: HashSet<String>) -> Self {
        Self {
            procedures: Arc::new(procedures),
            modules: Arc::new(modules),
        }
    }

    pub(crate) fn contains_procedure(&self, key: &str) -> bool {
        self.procedures.contains(key)
    }

    pub(crate) fn contains_module(&self, module_path: &str) -> bool {
        self.modules.contains(module_path)
    }
}

/// A normalized MASM block for frontend consumers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnalysisBody {
    span: SourceSpan,
    ops: Vec<AnalysisOp>,
}

impl AnalysisBody {
    /// Create a normalized body from its source span and root operations.
    pub fn new(span: SourceSpan, ops: Vec<AnalysisOp>) -> Self {
        Self { span, ops }
    }

    /// Return the source span of this body.
    pub fn span(&self) -> SourceSpan {
        self.span
    }

    /// Return the root operations in this body.
    pub fn ops(&self) -> &[AnalysisOp] {
        &self.ops
    }

    /// Return the number of root operations in this body.
    pub fn len(&self) -> usize {
        self.ops.len()
    }

    /// Return true if this body is empty.
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }
}

/// A normalized MASM operation exposed by an analysis frontend.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AnalysisOp {
    /// A conditional branch with normalized `then` and `else` blocks.
    If {
        /// Source span of the `if` operation.
        span: SourceSpan,
        /// Normalized `then` branch body.
        then_body: AnalysisBody,
        /// Normalized `else` branch body.
        else_body: AnalysisBody,
    },
    /// A condition-controlled loop.
    While {
        /// Source span of the `while` operation.
        span: SourceSpan,
        /// Normalized loop body.
        body: AnalysisBody,
    },
    /// A counter-controlled loop.
    Repeat {
        /// Source span of the `repeat` operation.
        span: SourceSpan,
        /// Repeat count as written after semantic analysis.
        count: Immediate<u32>,
        /// Normalized loop body.
        body: AnalysisBody,
    },
    /// A primitive instruction together with analysis-relevant classification.
    Inst(AnalysisInstruction),
}

impl AnalysisOp {
    /// Return the source span of this operation.
    pub fn span(&self) -> SourceSpan {
        match self {
            Self::If { span, .. } | Self::While { span, .. } | Self::Repeat { span, .. } => *span,
            Self::Inst(instruction) => instruction.span(),
        }
    }
}

/// A normalized primitive instruction exposed by an analysis frontend.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnalysisInstruction {
    span: SourceSpan,
    instruction: Instruction,
    invocation: Option<AnalysisInvocation>,
    local_access: Option<AnalysisLocalAccess>,
}

impl AnalysisInstruction {
    /// Create a normalized instruction with optional invocation and local-access metadata.
    pub fn new(
        span: SourceSpan,
        instruction: Instruction,
        invocation: Option<AnalysisInvocation>,
        local_access: Option<AnalysisLocalAccess>,
    ) -> Self {
        Self {
            span,
            instruction,
            invocation,
            local_access,
        }
    }

    /// Return the source span of this instruction.
    pub fn span(&self) -> SourceSpan {
        self.span
    }

    /// Return the underlying MASM instruction.
    pub fn instruction(&self) -> &Instruction {
        &self.instruction
    }

    /// Return normalized invocation metadata if this is a call-like instruction.
    pub fn invocation(&self) -> Option<&AnalysisInvocation> {
        self.invocation.as_ref()
    }

    /// Return normalized local-access metadata if this instruction touches locals.
    pub fn local_access(&self) -> Option<&AnalysisLocalAccess> {
        self.local_access.as_ref()
    }
}

/// A normalized call-like instruction target.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnalysisInvocation {
    kind: AnalysisInvocationKind,
    target: AnalysisInvocationTarget,
    span: SourceSpan,
}

impl AnalysisInvocation {
    /// Create normalized invocation metadata.
    pub fn new(
        kind: AnalysisInvocationKind,
        target: AnalysisInvocationTarget,
        span: SourceSpan,
    ) -> Self {
        Self {
            kind,
            target,
            span,
        }
    }

    /// Return the invocation kind.
    pub fn kind(&self) -> AnalysisInvocationKind {
        self.kind
    }

    /// Return the normalized target text and classification.
    pub fn target(&self) -> &AnalysisInvocationTarget {
        &self.target
    }

    /// Return the source span of the invocation target.
    pub fn span(&self) -> SourceSpan {
        self.span
    }
}

/// A precisely resolved invocation reported by a richer frontend backend.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ResolvedInvocation {
    span: SourceSpan,
    summary_key: SummaryKey,
}

impl ResolvedInvocation {
    /// Create a precise resolved invocation entry.
    pub(crate) fn new(span: SourceSpan, summary_key: SummaryKey) -> Self {
        Self { span, summary_key }
    }

    /// Return the invocation span covered by this precise resolution.
    pub(crate) fn span(&self) -> SourceSpan {
        self.span
    }

    /// Return the resolved callee summary key.
    pub(crate) fn summary_key(&self) -> &SummaryKey {
        &self.summary_key
    }
}

/// Kind of a call-like MASM instruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnalysisInvocationKind {
    /// An `exec` instruction.
    Exec,
    /// A `call` instruction.
    Call,
    /// A `syscall` instruction.
    SysCall,
    /// A `procref` instruction.
    ProcRef,
}

/// Syntax-level kind of invocation target.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AnalysisInvocationTarget {
    /// A target written as a MAST root digest.
    MastRoot(String),
    /// A target written as a local symbol name.
    Symbol(String),
    /// A target written as a path.
    Path(String),
}

/// A normalized local access in MASM.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AnalysisLocalAccess {
    /// Address-of access to a local slot.
    Address {
        /// Local index used by the instruction.
        index: u16,
        /// Source span of the access.
        span: SourceSpan,
    },
    /// Load from a local slot or word.
    Load {
        /// Local index used by the instruction.
        index: u16,
        /// Layout of the loaded local region.
        lane: AnalysisLocalLane,
        /// Source span of the access.
        span: SourceSpan,
    },
    /// Store into a local slot or word.
    Store {
        /// Local index used by the instruction.
        index: u16,
        /// Layout of the stored local region.
        lane: AnalysisLocalLane,
        /// Source span of the access.
        span: SourceSpan,
    },
}

impl AnalysisLocalAccess {
    /// Return the local index touched by this access.
    pub fn index(&self) -> u16 {
        match self {
            Self::Address { index, .. } | Self::Load { index, .. } | Self::Store { index, .. } => {
                *index
            }
        }
    }

    /// Return the source span of this local access.
    pub fn span(&self) -> SourceSpan {
        match self {
            Self::Address { span, .. } | Self::Load { span, .. } | Self::Store { span, .. } => {
                *span
            }
        }
    }

    /// Return the local lane layout when this access loads or stores data.
    pub fn lane(&self) -> Option<AnalysisLocalLane> {
        match self {
            Self::Address { .. } => None,
            Self::Load { lane, .. } | Self::Store { lane, .. } => Some(*lane),
        }
    }
}

/// Layout used for local-slot accesses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnalysisLocalLane {
    /// A single felt local.
    Element,
    /// A word-local using big-endian layout.
    WordBigEndian,
    /// A word-local using little-endian layout.
    WordLittleEndian,
}

/// Build a normalized frontend body from a MASM syntax block.
pub(crate) fn build_body(
    module: &Module,
    source_manager: Arc<dyn SourceManager>,
    block: &Block,
) -> AnalysisBody {
    let resolver = module.resolver(source_manager);
    AnalysisBody::new(
        block.span(),
        block.iter().map(|op| build_op(module, &resolver, op)).collect(),
    )
}

fn build_op(module: &Module, resolver: &LocalSymbolResolver, op: &Op) -> AnalysisOp {
    match op {
        Op::If {
            span,
            then_blk,
            else_blk,
        } => AnalysisOp::If {
            span: *span,
            then_body: build_body(module, resolver.source_manager(), then_blk),
            else_body: build_body(module, resolver.source_manager(), else_blk),
        },
        Op::While { span, body } => AnalysisOp::While {
            span: *span,
            body: build_body(module, resolver.source_manager(), body),
        },
        Op::Repeat { span, count, body } => AnalysisOp::Repeat {
            span: *span,
            count: count.clone(),
            body: build_body(module, resolver.source_manager(), body),
        },
        Op::Inst(instruction) => AnalysisOp::Inst(build_instruction(module, resolver, instruction)),
    }
}

fn build_instruction(
    module: &Module,
    resolver: &LocalSymbolResolver,
    instruction: &Span<Instruction>,
) -> AnalysisInstruction {
    AnalysisInstruction::new(
        instruction.span(),
        instruction.inner().clone(),
        classify_invocation(module, resolver, instruction),
        classify_local_access(instruction),
    )
}

fn classify_invocation(
    _module: &Module,
    _resolver: &LocalSymbolResolver,
    instruction: &Span<Instruction>,
) -> Option<AnalysisInvocation> {
    let (kind, target) = match instruction.inner() {
        Instruction::Exec(target) => (AnalysisInvocationKind::Exec, target),
        Instruction::Call(target) => (AnalysisInvocationKind::Call, target),
        Instruction::SysCall(target) => (AnalysisInvocationKind::SysCall, target),
        Instruction::ProcRef(target) => (AnalysisInvocationKind::ProcRef, target),
        _ => return None,
    };

    Some(AnalysisInvocation::new(
        kind,
        invocation_target(target),
        target.span(),
    ))
}

/// Collect precise invocation resolutions for one block when a backend has richer call-target
/// knowledge than the minimal frontend contract requires.
pub(crate) fn collect_precise_invocations(
    module: &Module,
    source_manager: Arc<dyn SourceManager>,
    block: &Block,
    known_targets: &KnownTargets,
) -> Vec<ResolvedInvocation> {
    let resolver = module.resolver(source_manager);
    let mut resolved = Vec::new();
    collect_block_precise_invocations(module, &resolver, block, known_targets, &mut resolved);
    resolved
}

fn invocation_target(target: &InvocationTarget) -> AnalysisInvocationTarget {
    match target {
        InvocationTarget::MastRoot(word) => AnalysisInvocationTarget::MastRoot(word.to_string()),
        InvocationTarget::Symbol(symbol) => AnalysisInvocationTarget::Symbol(symbol.to_string()),
        InvocationTarget::Path(path) => AnalysisInvocationTarget::Path(path.to_string()),
    }
}

fn resolve_invocation_target(
    module: &Module,
    resolver: &LocalSymbolResolver,
    target: &InvocationTarget,
    known_targets: &KnownTargets,
) -> Option<SummaryKey> {
    match target {
        InvocationTarget::MastRoot(_) => None,
        InvocationTarget::Symbol(symbol) => {
            match resolver.resolve(Span::new(symbol.span(), symbol.as_str())) {
                Ok(resolution) => {
                    resolved_summary_key(module, symbol.as_str(), resolution, true, known_targets)
                }
                Err(SymbolResolutionError::UndefinedSymbol { .. }) => None,
                Err(_) => None,
            }
        }
        InvocationTarget::Path(path) => {
            let target_text = path.to_string();
            match resolver.resolve_path(path.as_deref()) {
                Ok(resolution) => {
                    resolved_summary_key(
                        module,
                        &target_text,
                        resolution,
                        false,
                        known_targets,
                    )
                }
                Err(SymbolResolutionError::UndefinedSymbol { .. }) => {
                    Some(summary_key_for_path(&target_text))
                }
                Err(_) => None,
            }
        }
    }
}

fn collect_block_precise_invocations(
    module: &Module,
    resolver: &LocalSymbolResolver,
    block: &Block,
    known_targets: &KnownTargets,
    resolved: &mut Vec<ResolvedInvocation>,
) {
    for op in block.iter() {
        match op {
            Op::If {
                then_blk, else_blk, ..
            } => {
                collect_block_precise_invocations(
                    module,
                    resolver,
                    then_blk,
                    known_targets,
                    resolved,
                );
                collect_block_precise_invocations(
                    module,
                    resolver,
                    else_blk,
                    known_targets,
                    resolved,
                );
            }
            Op::While { body, .. } | Op::Repeat { body, .. } => {
                collect_block_precise_invocations(module, resolver, body, known_targets, resolved);
            }
            Op::Inst(instruction) => collect_instruction_precise_invocation(
                module,
                resolver,
                instruction,
                known_targets,
                resolved,
            ),
        }
    }
}

fn collect_instruction_precise_invocation(
    module: &Module,
    resolver: &LocalSymbolResolver,
    instruction: &Span<Instruction>,
    known_targets: &KnownTargets,
    resolved: &mut Vec<ResolvedInvocation>,
) {
    let target = match instruction.inner() {
        Instruction::Exec(target)
        | Instruction::Call(target)
        | Instruction::SysCall(target)
        | Instruction::ProcRef(target) => target,
        _ => return,
    };

    if let Some(summary_key) = resolve_invocation_target(module, resolver, target, known_targets) {
        resolved.push(ResolvedInvocation::new(target.span(), summary_key));
    }
}

fn resolved_summary_key(
    module: &Module,
    target_text: &str,
    resolution: SymbolResolution,
    is_symbol_target: bool,
    known_targets: &KnownTargets,
) -> Option<SummaryKey> {
    match resolution {
        SymbolResolution::Local(index) => local_procedure_summary_key(module, target_text, index),
        SymbolResolution::External(path) => {
            let canonical_path = path.to_string();
            let module_path = canonical_path.trim_start_matches("::");
            let summary_key = summary_key_for_path(&canonical_path);

            if is_symbol_target {
                // A bare imported symbol like `exec.plus` lacks item-kind metadata from
                // `assembly-syntax`, so the precise frontend stays conservative unless the
                // canonical callee is already part of the loaded workspace (tracked in
                // `KnownTargets`).
                known_targets
                    .contains_procedure(summary_key.as_str())
                    .then_some(summary_key)
            } else if known_targets.contains_module(module_path) {
                None
            } else {
                Some(summary_key)
            }
        }
        SymbolResolution::Exact { path, .. } => {
            let canonical_path = path.to_string();
            let module_path = canonical_path.trim_start_matches("::");
            let summary_key = summary_key_for_path(&canonical_path);

            if is_symbol_target {
                known_targets
                    .contains_procedure(summary_key.as_str())
                    .then_some(summary_key)
            } else if known_targets.contains_module(module_path) {
                None
            } else {
                Some(summary_key)
            }
        }
        SymbolResolution::Module { .. } | SymbolResolution::MastRoot(_) => None,
    }
}

fn summary_key_for_path(path: &str) -> SummaryKey {
    SummaryKey::new(path.trim_start_matches("::"))
}

fn local_procedure_summary_key(
    module: &Module,
    target_text: &str,
    index: Span<ItemIndex>,
) -> Option<SummaryKey> {
    match module.get(index.into_inner()) {
        Some(Export::Procedure(_)) => Some(summary_key_for_name(module_path(module), target_text)),
        Some(Export::Constant(_) | Export::Type(_) | Export::Alias(_)) | None => None,
    }
}

fn classify_local_access(instruction: &Span<Instruction>) -> Option<AnalysisLocalAccess> {
    let span = instruction.span();

    match instruction.inner() {
        Instruction::Locaddr(index) => Some(AnalysisLocalAccess::Address {
            index: index.expect_value(),
            span,
        }),
        Instruction::LocLoad(index) => Some(AnalysisLocalAccess::Load {
            index: index.expect_value(),
            lane: AnalysisLocalLane::Element,
            span,
        }),
        Instruction::LocLoadWBe(index) => Some(AnalysisLocalAccess::Load {
            index: index.expect_value(),
            lane: AnalysisLocalLane::WordBigEndian,
            span,
        }),
        Instruction::LocLoadWLe(index) => Some(AnalysisLocalAccess::Load {
            index: index.expect_value(),
            lane: AnalysisLocalLane::WordLittleEndian,
            span,
        }),
        Instruction::LocStore(index) => Some(AnalysisLocalAccess::Store {
            index: index.expect_value(),
            lane: AnalysisLocalLane::Element,
            span,
        }),
        Instruction::LocStoreWBe(index) => Some(AnalysisLocalAccess::Store {
            index: index.expect_value(),
            lane: AnalysisLocalLane::WordBigEndian,
            span,
        }),
        Instruction::LocStoreWLe(index) => Some(AnalysisLocalAccess::Store {
            index: index.expect_value(),
            lane: AnalysisLocalLane::WordLittleEndian,
            span,
        }),
        _ => None,
    }
}

fn module_path(module: &Module) -> &str {
    module.path().as_ref()
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use masm_decompiler::frontend::testing::workspace_from_modules;
    use miden_assembly_syntax::{
        ast::{GlobalItemIndex, ItemIndex, ModuleIndex, Path},
        Word,
    };
    use miden_debug_types::{SourceSpan, Span};
    use std::collections::HashSet;

    use super::*;

    #[test]
    fn resolved_summary_key_from_mast_root_is_none() {
        let workspace =
            workspace_from_modules(&[("math::word_ops", "proc foo\n    push.1\nend\n")]);
        let module = workspace.modules().next().expect("module").module();
        let resolution = SymbolResolution::MastRoot(Span::new(SourceSpan::UNKNOWN, Word::empty()));

        let known_targets = KnownTargets::new(HashSet::new(), HashSet::new());

        assert!(
            resolved_summary_key(module, "foo", resolution, true, &known_targets).is_none(),
            "mast root resolutions should not yield summary keys"
        );
    }

    #[test]
    fn resolved_summary_key_filters_unknown_local_symbols() {
        let workspace =
            workspace_from_modules(&[("math::word_ops", "proc foo\n    push.1\nend\n")]);
        let module = workspace.modules().next().expect("module").module();
        let resolution =
            SymbolResolution::Local(Span::new(SourceSpan::UNKNOWN, ItemIndex::new(0)));
        let known_targets = KnownTargets::new(HashSet::new(), HashSet::new());

        assert!(
            resolved_summary_key(module, "unused_constant", resolution, true, &known_targets)
                .is_none(),
            "local non-procedure symbols should not produce summary keys"
        );
    }

    #[test]
    fn resolved_summary_key_filters_unknown_external_aliases() {
        let workspace =
            workspace_from_modules(&[("math::word_ops", "proc foo\n    push.1\nend\n")]);
        let module = workspace.modules().next().expect("module").module();
        let path_buf =
            miden_assembly_syntax::ast::PathBuf::new("std::math::u64").expect("valid path");
        let canonical_path = <Path as AsRef<str>>::as_ref(path_buf.as_ref()).to_string();
        let arc_path = Arc::from(Path::new(&canonical_path));
        let resolution = SymbolResolution::External(Span::new(SourceSpan::UNKNOWN, arc_path));
        let known_targets = KnownTargets::new(HashSet::new(), HashSet::new());

        assert!(
            resolved_summary_key(module, "u64", resolution, true, &known_targets).is_none(),
            "unknown external aliases should not produce summary keys"
        );
    }

    #[test]
    fn resolved_summary_key_filters_two_segment_external_symbol_aliases() {
        let workspace =
            workspace_from_modules(&[("math::word_ops", "proc foo\n    push.1\nend\n")]);
        let module = workspace.modules().next().expect("module").module();
        let path_buf = miden_assembly_syntax::ast::PathBuf::new("math::add").expect("valid path");
        let canonical_path = <Path as AsRef<str>>::as_ref(path_buf.as_ref()).to_string();
        let arc_path = Arc::from(Path::new(&canonical_path));
        let resolution = SymbolResolution::External(Span::new(SourceSpan::UNKNOWN, arc_path));
        let known_targets = KnownTargets::new(HashSet::new(), HashSet::new());

        assert!(
            resolved_summary_key(module, "plus", resolution, true, &known_targets).is_none(),
            "two-segment external symbol aliases should stay opaque without linker metadata"
        );
    }

    #[test]
    fn resolved_summary_key_preserves_known_external_procedure_aliases() {
        let modules = &[
            ("math::helpers", "pub proc add\n    push.1\nend\n"),
            (
                "app::main",
                "use ::math::helpers::add->plus\nproc caller\n    exec.plus\nend\n",
            ),
        ];
        let workspace = workspace_from_modules(modules);
        let module = workspace
            .modules()
            .find(|program| {
                <Path as AsRef<str>>::as_ref(program.module().path()) == "app::main"
            })
            .expect("main")
            .module();
        let path_buf =
            miden_assembly_syntax::ast::PathBuf::new("math::helpers::add").expect("valid path");
        let canonical_path = <Path as AsRef<str>>::as_ref(path_buf.as_ref()).to_string();
        let arc_path = Arc::from(Path::new(&canonical_path));
        let resolution = SymbolResolution::External(Span::new(SourceSpan::UNKNOWN, arc_path));
        let (procedures, module_paths) = workspace.modules().fold(
            (HashSet::new(), HashSet::new()),
            |(mut procedures, mut module_paths), program| {
                let module_path = <Path as AsRef<str>>::as_ref(program.module().path()).to_string();
                module_paths.insert(module_path.clone());
                procedures.extend(program.procedures().map(|procedure| {
                    summary_key_for_name(&module_path, procedure.name().as_str())
                        .as_str()
                        .to_string()
                }));
                (procedures, module_paths)
            },
        );
        let known_targets = KnownTargets::new(procedures, module_paths);

        assert_eq!(
            resolved_summary_key(module, "plus", resolution, true, &known_targets)
                .as_ref()
                .map(SummaryKey::as_str),
            Some("math::helpers::add"),
            "external symbol aliases backed by workspace-known procedures should preserve \
             summary keys"
        );
    }

    #[test]
    fn resolved_summary_key_filters_exact_symbol_targets_not_known_as_procedures() {
        let workspace =
            workspace_from_modules(&[("math::word_ops", "proc foo\n    push.1\nend\n")]);
        let module = workspace.modules().next().expect("module").module();
        let path_buf =
            miden_assembly_syntax::ast::PathBuf::new("pkg::math::CONST").expect("valid path");
        let canonical_path = <Path as AsRef<str>>::as_ref(path_buf.as_ref()).to_string();
        let arc_path = Arc::from(Path::new(&canonical_path));
        let resolution = SymbolResolution::Exact {
            gid: GlobalItemIndex {
                module: ModuleIndex::const_new(0),
                index: ItemIndex::const_new(0),
            },
            path: Span::new(SourceSpan::UNKNOWN, arc_path),
        };
        let known_targets = KnownTargets::new(HashSet::new(), HashSet::new());

        assert!(
            resolved_summary_key(module, "x", resolution, true, &known_targets).is_none(),
            "exact symbol targets should only resolve when the frontend knows they are procedures"
        );
    }
}
