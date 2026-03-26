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
    u32_domain::U32Validity,
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
    env.set_var_u32_validity(dest, expr_u32_validity(expr, env));
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

    env.set_var_u32_validity(
        dest,
        lhs_env
            .u32_validity_for_var(lhs_var)
            .join(rhs_env.u32_validity_for_var(rhs_var)),
    );
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

/// Compute the `u32` validity of an expression result.
pub(crate) fn expr_u32_validity(expr: &Expr, env: &Env) -> U32Validity {
    match expr {
        Expr::Var(var) => env.u32_validity_for_var(var),
        Expr::Ternary {
            then_expr,
            else_expr,
            ..
        } => expr_u32_validity(then_expr, env).join(expr_u32_validity(else_expr, env)),
        Expr::Unary(op, inner) => match op {
            UnOp::U32Cast
            | UnOp::U32Test
            | UnOp::U32Not
            | UnOp::U32Clz
            | UnOp::U32Ctz
            | UnOp::U32Clo
            | UnOp::U32Cto => U32Validity::ProvenU32,
            UnOp::Neg | UnOp::Inv | UnOp::Pow2 | UnOp::Not => {
                let _ = inner;
                U32Validity::Unknown
            }
        },
        Expr::Binary(op, lhs, rhs) => match op {
            BinOp::U32And
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
            | BinOp::U32WrappingMul => U32Validity::ProvenU32,
            BinOp::U32Exp => U32Validity::Unknown,
            BinOp::Add
            | BinOp::Sub
            | BinOp::Mul
            | BinOp::Div
            | BinOp::And
            | BinOp::Or
            | BinOp::Xor
            | BinOp::Eq
            | BinOp::Neq
            | BinOp::Lt
            | BinOp::Lte
            | BinOp::Gt
            | BinOp::Gte => {
                let _ = (lhs, rhs);
                U32Validity::Unknown
            }
        },
        Expr::EqW { .. } | Expr::True | Expr::False | Expr::Constant(_) => U32Validity::Unknown,
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
                env.set_var_u32_validity(result, U32Validity::ProvenU32);
            }
        }
        "u32testw" => {
            if let Some((flag, preserved)) = intrinsic.results.split_first() {
                env.set_var_fact(flag, AdviceFact::bottom());
                env.clear_var_metadata(flag);
                env.set_var_u32_validity(flag, U32Validity::ProvenU32);
                for (result, arg) in preserved.iter().zip(intrinsic.args.iter()) {
                    env.set_var_fact(result, env.fact_for_var(arg));
                    env.set_var_u32_validity(result, env.u32_validity_for_var(arg));
                    env.set_var_identity(result, env.identity_for_var(arg));
                    env.set_var_zero_test(result, env.zero_test_for_var(arg));
                }
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
                env.set_var_u32_validity(result, U32Validity::ProvenU32);
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
    let validity = single_var(values)
        .map(|var| env.u32_validity_for_var(var))
        .unwrap_or(U32Validity::Unknown);
    env.set_local_u32_validity(index, validity);
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
        env.set_local_u32_validity(slot, env.u32_validity_for_var(value));
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
        env.set_var_u32_validity(output, env.u32_validity_for_local(index));
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
        env.set_var_u32_validity(output, env.u32_validity_for_local(slot));
        env.clear_var_metadata(output);
    }
}

/// Apply interprocedural summary substitution at one direct call site.
///
/// This is the explicit summary-application step of the Phase 1 abstract interpreter: resolve the
/// callee summary, substitute caller argument facts into summarized outputs, and write the results
/// back into the caller state.
///
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
    let caller_arg_u32_validity = args
        .iter()
        .map(|arg| env.u32_validity_for_var(arg))
        .collect::<Vec<_>>();
    for ((arg, summary_u32), caller_u32) in args
        .iter()
        .zip(summary.u32_inputs().iter())
        .zip(caller_arg_u32_validity.iter().copied())
    {
        env.set_var_u32_validity(
            arg,
            if summary_u32.is_proven() || caller_u32.is_proven() {
                U32Validity::ProvenU32
            } else {
                U32Validity::Unknown
            },
        );
    }
    for (((result, summary_fact), summary_u32), forwarded_input) in results
        .iter()
        .zip(summary.outputs().iter())
        .zip(summary.u32_outputs().iter())
        .zip(summary.forwarded_inputs().iter())
    {
        env.set_var_fact(result, substitute_output_fact(summary_fact, &arg_facts));
        let forwarded_arg = forwarded_input.and_then(|input_index| args.get(input_index));
        let forwarded_validity = forwarded_arg
            .map(|arg| env.u32_validity_for_var(arg))
            .unwrap_or(U32Validity::Unknown);
        env.set_var_u32_validity(
            result,
            if summary_u32.is_proven() || forwarded_validity.is_proven() {
                U32Validity::ProvenU32
            } else {
                U32Validity::Unknown
            },
        );
        if let Some(arg) = forwarded_arg {
            env.set_var_identity(result, env.identity_for_var(arg));
            env.set_var_zero_test(result, env.zero_test_for_var(arg));
        } else {
            env.clear_var_metadata(result);
        }
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
            env.set_var_u32_validity(result, env.u32_validity_for_var(preserved_input));
            env.set_var_identity(result, env.identity_for_var(preserved_input));
            env.set_var_zero_test(result, env.zero_test_for_var(preserved_input));
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

    use super::{apply_callee_summary, apply_intrinsic_effect, AdviceFact, Env};
    use crate::unconstrained_advice::{
        state::AdvicePlace,
        summary::AdviceSummary,
        u32_domain::U32Validity,
    };
    use masm_decompiler::{
        ir::{BinOp, Expr, Intrinsic, Var},
        SymbolPath,
    };
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

    #[test]
    fn u32assert_marks_its_operand_as_proven_u32() {
        let arg = test_var(0);
        let mut env = Env::default();
        let fact = AdviceFact::from_input(0);
        env.set_var_fact(&arg, fact.clone());

        let intrinsic = Intrinsic {
            name: "u32assert".to_string(),
            args: vec![arg],
            results: Vec::new(),
        };

        apply_intrinsic_effect(SourceSpan::UNKNOWN, &intrinsic, &mut env);

        assert_eq!(env.fact_for_var(&intrinsic.args[0]), fact);
        assert_eq!(
            env.place_u32_validity(&AdvicePlace::var(&intrinsic.args[0])),
            U32Validity::ProvenU32
        );
    }

    #[test]
    fn u32cast_assignment_marks_the_result_as_proven_u32() {
        let input = test_var(0);
        let result = test_var(1);
        let mut env = Env::default();
        env.set_var_fact(&input, AdviceFact::from_input(0));

        super::assign_expr_metadata(&result, &Expr::Unary(masm_decompiler::ir::UnOp::U32Cast, Box::new(Expr::Var(input))), &mut env);

        assert_eq!(
            env.place_u32_validity(&AdvicePlace::var(&result)),
            U32Validity::ProvenU32
        );
    }

    #[test]
    fn u32test_assignment_marks_the_boolean_result_as_proven_u32() {
        let input = test_var(0);
        let result = test_var(1);
        let other = test_var(2);
        let mut env = Env::default();
        env.set_var_fact(&input, AdviceFact::from_input(0));

        super::assign_expr_metadata(
            &result,
            &Expr::Unary(masm_decompiler::ir::UnOp::U32Test, Box::new(Expr::Var(input))),
            &mut env,
        );

        assert_eq!(
            env.place_u32_validity(&AdvicePlace::var(&result)),
            U32Validity::ProvenU32
        );
        assert_eq!(
            super::super::u32::expr_u32_sink_fact(
                &Expr::Binary(
                    BinOp::U32WrappingAdd,
                    Box::new(Expr::Var(result)),
                    Box::new(Expr::Var(other)),
                ),
                &env,
            ),
            AdviceFact::bottom()
        );
    }

    #[test]
    fn local_round_trip_preserves_u32_validity() {
        let input = test_var(0);
        let output = test_var(1);
        let mut env = Env::default();
        env.set_var_u32_validity(&input, U32Validity::ProvenU32);

        super::apply_local_store(std::slice::from_ref(&input), 0, &mut env);
        super::apply_local_load_scalar(std::slice::from_ref(&output), 0, &mut env);

        assert_eq!(
            env.place_u32_validity(&AdvicePlace::local(0)),
            U32Validity::ProvenU32
        );
        assert_eq!(
            env.place_u32_validity(&AdvicePlace::var(&output)),
            U32Validity::ProvenU32
        );
    }

    #[test]
    fn adv_pipe_preserves_u32_validity_for_forwarded_inputs() {
        let mut env = Env::default();
        let args = (0u8..13).map(test_var).collect::<Vec<_>>();
        let results = (13u8..26).map(test_var).collect::<Vec<_>>();
        let preserved_input = args[11].clone();
        env.set_var_fact(&preserved_input, AdviceFact::from_input(0));
        env.set_var_u32_validity(&preserved_input, U32Validity::ProvenU32);

        let intrinsic = Intrinsic {
            name: "adv_pipe".to_string(),
            args,
            results: results.clone(),
        };

        apply_intrinsic_effect(SourceSpan::UNKNOWN, &intrinsic, &mut env);

        assert_eq!(env.fact_for_var(&results[1]), AdviceFact::from_input(0));
        assert_eq!(
            env.place_u32_validity(&AdvicePlace::var(&results[1])),
            U32Validity::ProvenU32
        );
        assert_eq!(env.identity_for_var(&results[1]), env.identity_for_var(&preserved_input));
    }

    #[test]
    fn u32split_marks_both_limbs_as_proven_u32() {
        let input = test_var(0);
        let lo = test_var(1);
        let hi = test_var(2);
        let mut env = Env::default();

        let intrinsic = Intrinsic {
            name: "u32split".to_string(),
            args: vec![input],
            results: vec![lo.clone(), hi.clone()],
        };

        apply_intrinsic_effect(SourceSpan::UNKNOWN, &intrinsic, &mut env);

        assert_eq!(
            env.place_u32_validity(&AdvicePlace::var(&lo)),
            U32Validity::ProvenU32
        );
        assert_eq!(
            env.place_u32_validity(&AdvicePlace::var(&hi)),
            U32Validity::ProvenU32
        );
    }

    #[test]
    fn u32testw_marks_the_boolean_result_as_proven_u32() {
        let mut env = Env::default();
        let args = (0u8..4).map(test_var).collect::<Vec<_>>();
        let results = (4u8..9).map(test_var).collect::<Vec<_>>();
        env.set_var_fact(&args[0], AdviceFact::from_input(0));

        let intrinsic = Intrinsic {
            name: "u32testw".to_string(),
            args,
            results: results.clone(),
        };

        apply_intrinsic_effect(SourceSpan::UNKNOWN, &intrinsic, &mut env);

        assert_eq!(
            env.place_u32_validity(&AdvicePlace::var(&results[0])),
            U32Validity::ProvenU32
        );
        assert_eq!(
            super::super::u32::intrinsic_u32_sink_fact(
                &Intrinsic {
                    name: "u32overflowing_add".to_string(),
                    args: vec![results[0].clone()],
                    results: Vec::new(),
                },
                &env,
            ),
            AdviceFact::bottom()
        );
    }

    #[test]
    fn phi_join_keeps_u32_validity_only_when_both_inputs_are_proven() {
        let lhs = test_var(0);
        let rhs = test_var(1);
        let dest = test_var(2);
        let mut lhs_env = Env::default();
        let rhs_env = Env::default();
        let mut joined = Env::default();
        lhs_env.set_var_u32_validity(&lhs, U32Validity::ProvenU32);

        super::assign_phi_metadata(&dest, &lhs, &lhs_env, &rhs, &rhs_env, &mut joined);

        assert_eq!(
            joined.place_u32_validity(&AdvicePlace::var(&dest)),
            U32Validity::Unknown
        );
    }

    #[test]
    fn ordinary_arithmetic_does_not_infer_u32_validity() {
        let lhs = test_var(0);
        let rhs = test_var(1);
        let result = test_var(2);
        let mut env = Env::default();
        env.set_var_u32_validity(&lhs, U32Validity::ProvenU32);
        env.set_var_u32_validity(&rhs, U32Validity::ProvenU32);

        super::assign_expr_metadata(
            &result,
            &Expr::Binary(BinOp::Add, Box::new(Expr::Var(lhs)), Box::new(Expr::Var(rhs))),
            &mut env,
        );

        assert_eq!(
            env.place_u32_validity(&AdvicePlace::var(&result)),
            U32Validity::Unknown
        );
    }

    #[test]
    fn u32exp_result_is_not_assumed_to_be_u32() {
        let base = test_var(0);
        let exponent = test_var(1);
        let result = test_var(2);
        let mut env = Env::default();
        env.set_var_u32_validity(&base, U32Validity::ProvenU32);
        env.set_var_u32_validity(&exponent, U32Validity::ProvenU32);

        super::assign_expr_metadata(
            &result,
            &Expr::Binary(
                BinOp::U32Exp,
                Box::new(Expr::Var(base)),
                Box::new(Expr::Var(exponent)),
            ),
            &mut env,
        );

        assert_eq!(
            env.place_u32_validity(&AdvicePlace::var(&result)),
            U32Validity::Unknown
        );
    }

    #[test]
    fn call_summary_preserves_u32_validity_from_the_callee_summary() {
        let arg = test_var(0);
        let result = test_var(1);
        let mut env = Env::default();

        let summaries = HashMap::from([(
            SymbolPath::new("callee".to_string()),
            AdviceSummary::with_u32_outputs(
                vec![AdviceFact::from_input(0)],
                vec![U32Validity::ProvenU32],
            ),
        )]);

        apply_callee_summary(
            &mut env,
            "callee",
            &[arg],
            std::slice::from_ref(&result),
            &summaries,
        );

        assert_eq!(
            env.place_u32_validity(&AdvicePlace::var(&result)),
            U32Validity::ProvenU32
        );
    }

    #[test]
    fn call_summary_applies_input_u32_postconditions() {
        let arg = test_var(0);
        let mut env = Env::default();

        let summaries = HashMap::from([(
            SymbolPath::new("callee".to_string()),
            AdviceSummary::with_u32_postconditions(
                Vec::new(),
                Vec::new(),
                vec![U32Validity::ProvenU32],
            ),
        )]);

        apply_callee_summary(&mut env, "callee", std::slice::from_ref(&arg), &[], &summaries);

        assert_eq!(
            env.place_u32_validity(&AdvicePlace::var(&arg)),
            U32Validity::ProvenU32
        );
    }

    #[test]
    fn call_summary_does_not_clear_a_caller_proof_when_postcondition_is_unknown() {
        let arg = test_var(0);
        let mut env = Env::default();
        env.set_var_u32_validity(&arg, U32Validity::ProvenU32);

        let summaries = HashMap::from([(
            SymbolPath::new("callee".to_string()),
            AdviceSummary::with_u32_postconditions(Vec::new(), Vec::new(), vec![U32Validity::Unknown]),
        )]);

        apply_callee_summary(&mut env, "callee", std::slice::from_ref(&arg), &[], &summaries);

        assert_eq!(
            env.place_u32_validity(&AdvicePlace::var(&arg)),
            U32Validity::ProvenU32
        );
    }

    #[test]
    fn call_summary_preserves_forwarded_caller_u32_proof() {
        let arg = test_var(0);
        let result = test_var(1);
        let mut env = Env::default();
        env.set_var_u32_validity(&arg, U32Validity::ProvenU32);

        let summaries = HashMap::from([(
            SymbolPath::new("callee".to_string()),
            AdviceSummary::with_forwarding(
                vec![AdviceFact::from_input(0)],
                vec![U32Validity::Unknown],
                vec![Some(0)],
                Vec::new(),
            ),
        )]);

        apply_callee_summary(
            &mut env,
            "callee",
            std::slice::from_ref(&arg),
            std::slice::from_ref(&result),
            &summaries,
        );

        assert_eq!(
            env.place_u32_validity(&AdvicePlace::var(&result)),
            U32Validity::ProvenU32
        );
    }

    #[test]
    fn call_summary_can_forward_a_proof_across_aliased_formals() {
        let validated = test_var(0);
        let forwarded = test_var(1);
        let result = test_var(2);
        let mut env = Env::default();
        env.set_var_fact(&validated, AdviceFact::from_input(0));
        env.set_var_fact(&forwarded, AdviceFact::from_input(0));
        env.set_var_identity(&forwarded, env.identity_for_var(&validated));

        let summaries = HashMap::from([(
            SymbolPath::new("callee".to_string()),
            AdviceSummary::with_forwarding(
                vec![AdviceFact::from_input(1)],
                vec![U32Validity::Unknown],
                vec![Some(1)],
                vec![U32Validity::ProvenU32, U32Validity::Unknown],
            ),
        )]);

        apply_callee_summary(
            &mut env,
            "callee",
            &[validated, forwarded],
            std::slice::from_ref(&result),
            &summaries,
        );

        assert_eq!(
            env.place_u32_validity(&AdvicePlace::var(&result)),
            U32Validity::ProvenU32
        );
    }

    #[test]
    fn forwarded_result_keeps_alias_identity_for_later_sanitization() {
        let arg = test_var(0);
        let sibling = test_var(1);
        let result = test_var(2);
        let mut env = Env::default();
        env.set_var_fact(&arg, AdviceFact::from_input(0));
        env.set_var_fact(&sibling, AdviceFact::from_input(0));
        env.set_var_identity(&sibling, env.identity_for_var(&arg));

        let summaries = HashMap::from([(
            SymbolPath::new("callee".to_string()),
            AdviceSummary::with_forwarding(
                vec![AdviceFact::from_input(0)],
                vec![U32Validity::Unknown],
                vec![Some(0)],
                Vec::new(),
            ),
        )]);

        apply_callee_summary(
            &mut env,
            "callee",
            std::slice::from_ref(&arg),
            std::slice::from_ref(&result),
            &summaries,
        );
        env.sanitize_var(&result);

        assert_eq!(
            env.place_u32_validity(&AdvicePlace::var(&sibling)),
            U32Validity::ProvenU32
        );
    }
}
