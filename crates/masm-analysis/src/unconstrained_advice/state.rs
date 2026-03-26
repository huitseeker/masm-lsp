//! Explicit abstract state used by unconstrained-advice analyses.

use std::collections::{HashMap, HashSet};

use masm_decompiler::{
    ir::Var,
    types::VarKey,
};

use crate::abstract_interp::JoinSemiLattice;

use super::domain::AdviceFact;

/// Analysis-visible storage location tracked by the unconstrained-advice state.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) enum AdvicePlace {
    /// An SSA value produced by the lifted decompiler IR.
    Var(VarKey),
    /// A local slot in the current procedure frame.
    Local(u32),
}

impl AdvicePlace {
    /// Build a place referring to a lifted SSA value.
    pub(crate) fn var(var: &Var) -> Self {
        Self::Var(VarKey::from_var(var))
    }

    /// Build a place referring to a local slot.
    pub(crate) fn local(slot: u32) -> Self {
        Self::Local(slot)
    }
}

/// Exact witness that a boolean value was computed as `x == 0`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct EqZeroWitness {
    /// Alias identity of the value compared against zero.
    pub(crate) value_identity: VarKey,
}

/// Abstract machine state at one program point for advice analyses.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub(crate) struct AdviceState {
    facts: HashMap<AdvicePlace, AdviceFact>,
    aliases: HashMap<VarKey, VarKey>,
    local_aliases: HashMap<u32, VarKey>,
    zero_tests: HashMap<VarKey, EqZeroWitness>,
    local_zero_tests: HashMap<u32, EqZeroWitness>,
    nonzero_identities: HashSet<VarKey>,
}

impl AdviceState {
    /// Read the current fact for a tracked place.
    pub(crate) fn fact_for_place(&self, place: &AdvicePlace) -> AdviceFact {
        self.facts
            .get(place)
            .cloned()
            .unwrap_or_else(AdviceFact::bottom)
    }

    /// Set the current fact for a tracked place.
    pub(crate) fn set_place_fact(&mut self, place: AdvicePlace, fact: AdviceFact) {
        self.facts.insert(place, fact);
    }

    /// Read the current fact for a variable.
    pub(crate) fn fact_for_var(&self, var: &Var) -> AdviceFact {
        self.fact_for_place(&AdvicePlace::var(var))
    }

    /// Set the current fact for a variable.
    pub(crate) fn set_var_fact(&mut self, var: &Var, fact: AdviceFact) {
        self.set_place_fact(AdvicePlace::var(var), fact);
    }

    /// Return the alias identity for a variable.
    pub(crate) fn identity_for_var(&self, var: &Var) -> VarKey {
        let key = VarKey::from_var(var);
        self.aliases.get(&key).cloned().unwrap_or(key)
    }

    /// Return the alias identity for a local slot, if known.
    pub(crate) fn identity_for_local(&self, slot: u32) -> Option<VarKey> {
        self.local_aliases.get(&slot).cloned()
    }

    /// Set the alias identity for a variable.
    pub(crate) fn set_var_identity(&mut self, var: &Var, identity: VarKey) {
        let key = VarKey::from_var(var);
        if identity == key {
            self.aliases.remove(&key);
        } else {
            self.aliases.insert(key, identity);
        }
    }

    /// Clear any alias identity for a variable.
    pub(crate) fn clear_var_identity(&mut self, var: &Var) {
        self.aliases.remove(&VarKey::from_var(var));
    }

    /// Set the alias identity for a local slot.
    pub(crate) fn set_local_identity(&mut self, slot: u32, identity: Option<VarKey>) {
        match identity {
            Some(identity) => {
                self.local_aliases.insert(slot, identity);
            }
            None => {
                self.local_aliases.remove(&slot);
            }
        }
    }

    /// Return the zero-test witness for a variable, if any.
    pub(crate) fn zero_test_for_var(&self, var: &Var) -> Option<EqZeroWitness> {
        self.zero_tests.get(&VarKey::from_var(var)).cloned()
    }

    /// Return the zero-test witness for a local slot, if any.
    pub(crate) fn zero_test_for_local(&self, slot: u32) -> Option<EqZeroWitness> {
        self.local_zero_tests.get(&slot).cloned()
    }

    /// Set the zero-test witness for a variable.
    pub(crate) fn set_var_zero_test(&mut self, var: &Var, witness: Option<EqZeroWitness>) {
        match witness {
            Some(witness) => {
                self.zero_tests.insert(VarKey::from_var(var), witness);
            }
            None => {
                self.zero_tests.remove(&VarKey::from_var(var));
            }
        }
    }

    /// Set the zero-test witness for a local slot.
    pub(crate) fn set_local_zero_test(&mut self, slot: u32, witness: Option<EqZeroWitness>) {
        match witness {
            Some(witness) => {
                self.local_zero_tests.insert(slot, witness);
            }
            None => {
                self.local_zero_tests.remove(&slot);
            }
        }
    }

    /// Return true if the variable is proven non-zero on the current path.
    pub(crate) fn is_var_nonzero(&self, var: &Var) -> bool {
        self.nonzero_identities
            .contains(&self.identity_for_var(var))
    }

    /// Mark the given alias identity as non-zero on the current path.
    pub(crate) fn mark_identity_nonzero(&mut self, identity: VarKey) {
        self.nonzero_identities.insert(identity);
    }

    /// Clear all best-effort metadata for a variable definition.
    pub(crate) fn clear_var_metadata(&mut self, var: &Var) {
        self.clear_var_identity(var);
        self.set_var_zero_test(var, None);
    }

    /// Sanitize a variable from this point onward.
    pub(crate) fn sanitize_var(&mut self, var: &Var) {
        self.set_var_fact(var, AdviceFact::bottom());
    }

    /// Read the current fact for a local slot.
    pub(crate) fn fact_for_local(&self, slot: u32) -> AdviceFact {
        self.fact_for_place(&AdvicePlace::local(slot))
    }

    /// Set the current fact for a local slot.
    pub(crate) fn set_local_fact(&mut self, slot: u32, fact: AdviceFact) {
        self.set_place_fact(AdvicePlace::local(slot), fact);
    }

    /// Join two abstract states conservatively.
    pub(crate) fn join(&self, other: &Self) -> Self {
        let mut joined = self.clone();
        for (place, fact) in &other.facts {
            let current = joined
                .facts
                .get(place)
                .cloned()
                .unwrap_or_else(AdviceFact::bottom);
            joined.facts.insert(place.clone(), current.join(fact));
        }
        joined.aliases = agreeing_entries(&self.aliases, &other.aliases);
        joined.local_aliases = agreeing_entries(&self.local_aliases, &other.local_aliases);
        joined.zero_tests = agreeing_entries(&self.zero_tests, &other.zero_tests);
        joined.local_zero_tests = agreeing_entries(&self.local_zero_tests, &other.local_zero_tests);
        joined.nonzero_identities = self
            .nonzero_identities
            .intersection(&other.nonzero_identities)
            .cloned()
            .collect();
        joined
    }
}

impl JoinSemiLattice for AdviceState {
    fn join_assign(&mut self, other: &Self) -> bool {
        let joined = self.join(other);
        let changed = *self != joined;
        *self = joined;
        changed
    }
}

/// Temporary compatibility alias while the rest of the pass migrates to `AdviceState`.
pub(crate) type Env = AdviceState;

/// Retain only entries that are present in both maps with the same value.
fn agreeing_entries<K, V>(lhs: &HashMap<K, V>, rhs: &HashMap<K, V>) -> HashMap<K, V>
where
    K: Clone + Eq + std::hash::Hash,
    V: Clone + Eq,
{
    lhs.iter()
        .filter_map(|(key, value)| {
            rhs.get(key)
                .filter(|other| *other == value)
                .map(|_| (key.clone(), value.clone()))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{AdvicePlace, AdviceState};
    use crate::{
        abstract_interp::JoinSemiLattice,
        unconstrained_advice::domain::AdviceFact,
    };
    use masm_decompiler::ir::Var;

    /// Return a small synthetic SSA variable for advice-state tests.
    fn test_var(index: u8) -> Var {
        Var::new(u64::from(index).into(), usize::from(index))
    }

    #[test]
    fn advice_place_constructors_produce_distinct_places() {
        let var = test_var(0);

        assert_ne!(AdvicePlace::var(&var), AdvicePlace::local(0));
    }

    #[test]
    fn advice_state_join_assign_reports_changes_and_merges_facts() {
        let mut lhs = AdviceState::default();
        let lhs_var = test_var(0);
        lhs.set_var_fact(&lhs_var, AdviceFact::from_input(0));

        let mut rhs = AdviceState::default();
        let rhs_var = test_var(1);
        rhs.set_var_fact(&lhs_var, AdviceFact::from_input(1));
        rhs.set_var_fact(&rhs_var, AdviceFact::from_input(2));

        assert!(lhs.join_assign(&rhs));
        assert_eq!(
            lhs.fact_for_var(&lhs_var),
            AdviceFact::from_input(0).join(&AdviceFact::from_input(1))
        );
        assert_eq!(lhs.fact_for_var(&rhs_var), AdviceFact::from_input(2));
        assert!(!lhs.join_assign(&rhs));
    }
}
