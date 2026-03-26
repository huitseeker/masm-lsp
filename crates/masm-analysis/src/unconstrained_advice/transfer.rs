//! Explicit transfer helpers for unconstrained-advice analyses.

use masm_decompiler::{
    ir::{BinOp, Expr, Intrinsic, LocalAccessKind, LoopPhi, Stmt, UnOp, Var},
    SymbolPath,
    types::VarKey,
};

use super::{
    domain::AdviceFact,
    state::EqZeroWitness,
    summary::AdviceSummaryMap,
};
pub(crate) use super::state::Env;

/// Maximum number of loop-approximation passes.
pub(crate) const MAX_LOOP_PASSES: usize = 32;

/// Seed input variables using the same numbering scheme as the lifting pass.
pub(crate) fn seed_input_env(input_count: usize) -> Env {
    let mut env = Env::default();
    for depth in 0..input_count {
        let input_position = input_count - 1 - depth;
        let var = Var::new((depth as u64).into(), depth);
        env.set_var_fact(&var, AdviceFact::from_input(input_position));
    }
    env
}

/// Preserve alias and zero-test metadata across a fresh assignment.
pub(crate) fn assign_expr_metadata(dest: &Var, expr: &Expr, env: &mut Env) {
    if let Some(identity) = expr_identity(expr, env) {
        env.set_var_identity(dest, identity);
    } else {
        env.clear_var_identity(dest);
    }
    env.set_var_zero_test(dest, eq_zero_witness_for_expr(expr, env));
}

/// Preserve metadata across a phi only when both sides agree exactly.
pub(crate) fn assign_phi_metadata(
    dest: &Var,
    lhs_var: &Var,
    lhs_env: &Env,
    rhs_var: &Var,
    rhs_env: &Env,
    env: &mut Env,
) {
    let lhs_identity = lhs_env.identity_for_var(lhs_var);
    let rhs_identity = rhs_env.identity_for_var(rhs_var);
    if lhs_identity == rhs_identity {
        env.set_var_identity(dest, lhs_identity);
    } else {
        env.clear_var_identity(dest);
    }

    let lhs_witness = lhs_env.zero_test_for_var(lhs_var);
    let rhs_witness = rhs_env.zero_test_for_var(rhs_var);
    if lhs_witness.is_some() && lhs_witness == rhs_witness {
        env.set_var_zero_test(dest, lhs_witness);
    } else {
        env.set_var_zero_test(dest, None);
    }
}

/// Join one loop-body evaluation back into the current abstract loop state.
pub(crate) fn join_loop_head_env(
    loop_env: &Env,
    entry_env: &Env,
    body_env: &Env,
    phis: &[LoopPhi],
) -> Env {
    let mut next_env = loop_env.join(body_env);
    for phi in phis {
        let merged = entry_env
            .fact_for_var(&phi.init)
            .join(&body_env.fact_for_var(&phi.step));
        next_env.set_var_fact(&phi.dest, merged);
        assign_phi_metadata(
            &phi.dest,
            &phi.init,
            entry_env,
            &phi.step,
            body_env,
            &mut next_env,
        );
    }

    next_env
}

/// Refine branch environments using an exact `eq.0` witness when available.
pub(crate) fn refine_if_envs(cond: &Expr, env: &Env) -> (Env, Env) {
    let then_env = env.clone();
    let mut else_env = env.clone();
    if let Some(witness) = eq_zero_witness_for_expr(cond, env) {
        else_env.mark_identity_nonzero(witness.value_identity);
    }
    (then_env, else_env)
}

/// Refine the environment after `assertz` proves an `eq.0` witness is zero.
pub(crate) fn refine_nonzero_from_intrinsic(intrinsic: &Intrinsic, env: &mut Env) {
    if intrinsic_base_name(&intrinsic.name) != "assertz" {
        return;
    }
    let Some(arg) = intrinsic.args.first() else {
        return;
    };
    let Some(witness) = env.zero_test_for_var(arg) else {
        return;
    };
    env.mark_identity_nonzero(witness.value_identity);
}

/// Return the exact alias identity of an expression, when it is a simple copy.
pub(crate) fn expr_identity(expr: &Expr, env: &Env) -> Option<VarKey> {
    match expr {
        Expr::Var(var) => Some(env.identity_for_var(var)),
        _ => None,
    }
}

/// Return the exact `eq.0` witness carried by an expression, if any.
pub(crate) fn eq_zero_witness_for_expr(expr: &Expr, env: &Env) -> Option<EqZeroWitness> {
    match expr {
        Expr::Var(var) => env.zero_test_for_var(var),
        Expr::Binary(BinOp::Eq, lhs, rhs) => {
            zero_comparison_var(lhs, rhs).map(|var| EqZeroWitness {
                value_identity: env.identity_for_var(var),
            })
        }
        _ => None,
    }
}

/// Return true when the expression is proven non-zero by the best-effort refinement.
pub(crate) fn expr_is_proven_nonzero(expr: &Expr, env: &Env) -> bool {
    match expr {
        Expr::Constant(constant) => !constant.is_zero(),
        Expr::Var(var) => env.is_var_nonzero(var),
        _ => false,
    }
}

/// Compute the provenance fact for an expression result.
pub(crate) fn expr_output_fact(expr: &Expr, env: &Env) -> AdviceFact {
    match expr {
        Expr::Var(var) => env.fact_for_var(var),
        Expr::Ternary {
            then_expr,
            else_expr,
            ..
        } => expr_output_fact(then_expr, env).join(&expr_output_fact(else_expr, env)),
        Expr::Unary(op, inner) => match op {
            UnOp::Neg | UnOp::Inv | UnOp::Pow2 => expr_output_fact(inner, env),
            UnOp::Not
            | UnOp::U32Cast
            | UnOp::U32Test
            | UnOp::U32Not
            | UnOp::U32Clz
            | UnOp::U32Ctz
            | UnOp::U32Clo
            | UnOp::U32Cto => AdviceFact::bottom(),
        },
        Expr::Binary(op, lhs, rhs) => match op {
            BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div => {
                expr_output_fact(lhs, env).join(&expr_output_fact(rhs, env))
            }
            BinOp::And
            | BinOp::Or
            | BinOp::Xor
            | BinOp::Eq
            | BinOp::Neq
            | BinOp::Lt
            | BinOp::Lte
            | BinOp::Gt
            | BinOp::Gte
            | BinOp::U32And
            | BinOp::U32Or
            | BinOp::U32Xor
            | BinOp::U32Shl
            | BinOp::U32Shr
            | BinOp::U32Rotr
            | BinOp::U32Lt
            | BinOp::U32Lte
            | BinOp::U32Gt
            | BinOp::U32Gte
            | BinOp::U32WrappingAdd
            | BinOp::U32WrappingSub
            | BinOp::U32WrappingMul
            | BinOp::U32Exp => AdviceFact::bottom(),
        },
        Expr::EqW { .. } | Expr::True | Expr::False | Expr::Constant(_) => AdviceFact::bottom(),
    }
}

/// Apply the common provenance transfer semantics of one intrinsic statement.
pub(crate) fn apply_intrinsic_effect(
    span: miden_debug_types::SourceSpan,
    intrinsic: &Intrinsic,
    env: &mut Env,
) {
    if intrinsic.name.starts_with("adv_push.") {
        for result in &intrinsic.results {
            env.set_var_fact(result, AdviceFact::from_source(span));
            env.clear_var_metadata(result);
        }
        return;
    }

    match intrinsic_base_name(&intrinsic.name) {
        "u32assert" | "u32assert2" | "u32assertw" => {
            for arg in &intrinsic.args {
                env.sanitize_var(arg);
            }
        }
        "u32split" => {
            for result in &intrinsic.results {
                env.set_var_fact(result, AdviceFact::bottom());
                env.clear_var_metadata(result);
            }
        }
        "is_odd" => {
            for result in &intrinsic.results {
                env.set_var_fact(result, AdviceFact::bottom());
                env.clear_var_metadata(result);
            }
        }
        "adv_pipe" => {
            apply_adv_pipe_effect(span, intrinsic, env);
        }
        "mem_stream" | "sdepth" => {
            for result in &intrinsic.results {
                env.set_var_fact(result, AdviceFact::bottom());
                env.clear_var_metadata(result);
            }
        }
        name if name.starts_with("locaddr") => {
            for result in &intrinsic.results {
                env.set_var_fact(result, AdviceFact::bottom());
                env.clear_var_metadata(result);
            }
        }
        _ if intrinsic_requires_u32_precondition(&intrinsic.name) => {
            for result in &intrinsic.results {
                env.set_var_fact(result, AdviceFact::bottom());
                env.clear_var_metadata(result);
            }
        }
        _ => {
            let joined =
                AdviceFact::join_all(intrinsic.args.iter().map(|arg| env.fact_for_var(arg)));
            for result in &intrinsic.results {
                env.set_var_fact(result, joined.clone());
                env.clear_var_metadata(result);
            }
        }
    }
}

/// Store a scalar local and preserve exact alias metadata when possible.
pub(crate) fn apply_local_store(values: &[Var], index: u32, env: &mut Env) {
    let fact = AdviceFact::join_all(values.iter().map(|var| env.fact_for_var(var)));
    env.set_local_fact(index, fact);
    let identity = single_var(values).map(|var| env.identity_for_var(var));
    let witness = single_var(values).and_then(|var| env.zero_test_for_var(var));
    env.set_local_identity(index, identity);
    env.set_local_zero_test(index, witness);
}

/// Store a local word slot-by-slot.
pub(crate) fn apply_local_store_word(
    kind: LocalAccessKind,
    values: &[Var],
    index: u32,
    env: &mut Env,
) {
    for (offset, value) in values.iter().enumerate() {
        let slot = match kind {
            // Canonical slot order follows ascending local-memory addresses.
            LocalAccessKind::WordBe => index + (values.len().saturating_sub(1) - offset) as u32,
            LocalAccessKind::WordLe => index + offset as u32,
            LocalAccessKind::Element => index,
        };
        env.set_local_fact(slot, env.fact_for_var(value));
        env.set_local_identity(slot, None);
        env.set_local_zero_test(slot, None);
    }
}

/// Load a scalar local and restore exact alias metadata when available.
pub(crate) fn apply_local_load_scalar(outputs: &[Var], index: u32, env: &mut Env) {
    let fact = env.fact_for_local(index);
    let identity = env.identity_for_local(index);
    let witness = env.zero_test_for_local(index);
    for output in outputs {
        env.set_var_fact(output, fact.clone());
        if let Some(identity) = identity.clone() {
            env.set_var_identity(output, identity);
        } else {
            env.clear_var_identity(output);
        }
        env.set_var_zero_test(output, witness.clone());
    }
}

/// Load a local word slot-by-slot, preserving stack order.
pub(crate) fn apply_local_load_word(
    kind: LocalAccessKind,
    outputs: &[Var],
    index: u32,
    env: &mut Env,
) {
    for (offset, output) in outputs.iter().enumerate() {
        let slot = match kind {
            LocalAccessKind::WordBe => index + offset as u32,
            LocalAccessKind::WordLe => index + (outputs.len().saturating_sub(1) - offset) as u32,
            LocalAccessKind::Element => index,
        };
        env.set_var_fact(output, env.fact_for_local(slot));
        env.clear_var_metadata(output);
    }
}

/// Apply interprocedural summary substitution at one direct call site.
///
/// This is the explicit summary-application step of the Phase 1 abstract interpreter: resolve the
/// callee summary, substitute caller argument facts into summarized outputs, and write the results
/// back into the caller state.
pub(crate) fn apply_callee_summary(
    env: &mut Env,
    target: &str,
    args: &[Var],
    results: &[Var],
    callee_summaries: &AdviceSummaryMap,
) {
    let Some(summary) = callee_summaries.get(&SymbolPath::new(target.to_string())) else {
        clear_call_results(env, results);
        return;
    };
    if summary.is_opaque() {
        clear_call_results(env, results);
        return;
    }

    let arg_facts = args
        .iter()
        .map(|arg| env.fact_for_var(arg))
        .collect::<Vec<_>>();
    for (result, summary_fact) in results.iter().zip(summary.outputs().iter()) {
        env.set_var_fact(result, substitute_output_fact(summary_fact, &arg_facts));
        env.clear_var_metadata(result);
    }
    for result in results.iter().skip(summary.output_count()) {
        env.set_var_fact(result, AdviceFact::bottom());
        env.clear_var_metadata(result);
    }
}

/// Return the variable compared against zero in an `eq.0`-shaped expression.
pub(crate) fn zero_comparison_var<'a>(lhs: &'a Expr, rhs: &'a Expr) -> Option<&'a Var> {
    match (lhs, rhs) {
        (Expr::Var(var), Expr::Constant(constant)) if constant.is_zero() => Some(var),
        (Expr::Constant(constant), Expr::Var(var)) if constant.is_zero() => Some(var),
        _ => None,
    }
}

/// Return the statement span for a structured statement.
pub(crate) fn stmt_span(stmt: &Stmt) -> miden_debug_types::SourceSpan {
    match stmt {
        Stmt::Assign { span, .. }
        | Stmt::MemLoad { span, .. }
        | Stmt::MemStore { span, .. }
        | Stmt::AdvLoad { span, .. }
        | Stmt::AdvStore { span, .. }
        | Stmt::LocalLoad { span, .. }
        | Stmt::LocalStore { span, .. }
        | Stmt::LocalStoreW { span, .. }
        | Stmt::Call { span, .. }
        | Stmt::Exec { span, .. }
        | Stmt::SysCall { span, .. }
        | Stmt::DynCall { span, .. }
        | Stmt::Intrinsic { span, .. }
        | Stmt::Repeat { span, .. }
        | Stmt::If { span, .. }
        | Stmt::While { span, .. }
        | Stmt::Return { span, .. } => *span,
    }
}

/// Return the base intrinsic name before immediates or `.err=*` suffixes.
pub(crate) fn intrinsic_base_name(name: &str) -> &str {
    name.split_once('.').map_or(name, |(base, _)| base)
}

/// Return true if an intrinsic requires caller-side `U32` preconditions.
pub(crate) fn intrinsic_requires_u32_precondition(name: &str) -> bool {
    if !name.starts_with("u32") {
        return false;
    }

    !matches!(
        intrinsic_base_name(name),
        "u32assert" | "u32assert2" | "u32assertw" | "u32cast" | "u32split" | "u32test" | "u32testw"
    )
}

/// Return the sole variable in the slice, if there is exactly one.
pub(crate) fn single_var(vars: &[Var]) -> Option<&Var> {
    match vars {
        [var] => Some(var),
        _ => None,
    }
}

/// Apply the custom transfer semantics of `adv_pipe`.
fn apply_adv_pipe_effect(
    span: miden_debug_types::SourceSpan,
    intrinsic: &Intrinsic,
    env: &mut Env,
) {
    if intrinsic.args.len() == 13 && intrinsic.results.len() == 13 {
        env.set_var_fact(&intrinsic.results[0], AdviceFact::bottom());
        env.clear_var_metadata(&intrinsic.results[0]);

        for (offset, result) in intrinsic.results[1..5].iter().enumerate() {
            let preserved_input = &intrinsic.args[11 - offset];
            env.set_var_fact(result, env.fact_for_var(preserved_input));
            env.clear_var_metadata(result);
        }

        for result in &intrinsic.results[5..] {
            env.set_var_fact(result, AdviceFact::from_source(span));
            env.clear_var_metadata(result);
        }
        return;
    }

    for result in &intrinsic.results {
        env.set_var_fact(result, AdviceFact::from_source(span));
        env.clear_var_metadata(result);
    }
}

/// Clear call outputs when the callee is missing or opaque.
fn clear_call_results(env: &mut Env, results: &[Var]) {
    for result in results {
        env.set_var_fact(result, AdviceFact::bottom());
        env.clear_var_metadata(result);
    }
}

/// Substitute caller argument facts into one summarized callee output.
fn substitute_output_fact(summary_fact: &AdviceFact, arg_facts: &[AdviceFact]) -> AdviceFact {
    let mut substituted = AdviceFact::bottom();
    substituted.source_spans = summary_fact.source_spans.clone();
    for input_index in &summary_fact.from_inputs {
        if let Some(arg_fact) = arg_facts.get(*input_index) {
            substituted = substituted.join(arg_fact);
        }
    }
    substituted
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::{apply_callee_summary, AdviceFact, Env};
    use crate::unconstrained_advice::summary::AdviceSummary;
    use masm_decompiler::{ir::Var, SymbolPath};
    use miden_debug_types::SourceSpan;

    /// Return a small synthetic SSA variable for transfer tests.
    fn test_var(index: u8) -> Var {
        Var::new(u64::from(index).into(), usize::from(index))
    }

    #[test]
    fn callee_summary_substitutes_argument_provenance() {
        let arg = test_var(0);
        let result = test_var(1);
        let mut env = Env::default();
        let arg_fact = AdviceFact::from_source(SourceSpan::UNKNOWN).join(&AdviceFact::from_input(0));
        env.set_var_fact(&arg, arg_fact.clone());

        let summaries = HashMap::from([(
            SymbolPath::new("callee".to_string()),
            AdviceSummary::new(vec![AdviceFact::from_input(0)]),
        )]);

        apply_callee_summary(
            &mut env,
            "callee",
            &[arg],
            std::slice::from_ref(&result),
            &summaries,
        );

        assert_eq!(env.fact_for_var(&result), arg_fact);
    }

    #[test]
    fn opaque_callee_summary_clears_result_facts() {
        let arg = test_var(0);
        let result = test_var(1);
        let mut env = Env::default();
        env.set_var_fact(&arg, AdviceFact::from_input(0));
        env.set_var_fact(&result, AdviceFact::from_input(1));

        let summaries = HashMap::from([(
            SymbolPath::new("callee".to_string()),
            AdviceSummary::opaque_with_arity(1),
        )]);

        apply_callee_summary(
            &mut env,
            "callee",
            &[arg],
            std::slice::from_ref(&result),
            &summaries,
        );

        assert_eq!(env.fact_for_var(&result), AdviceFact::bottom());
    }
}
