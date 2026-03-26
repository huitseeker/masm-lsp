//! Explicit abstract state used by unconstrained-advice analyses.

use std::collections::{HashMap, HashSet};

use masm_decompiler::{
    ir::Var,
    types::VarKey,
};

use crate::abstract_interp::JoinSemiLattice;

use super::{domain::AdviceFact, u32_domain::U32Validity};

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
    u32_validity: HashMap<AdvicePlace, U32Validity>,
    u32_valid_identities: HashSet<VarKey>,
    aliases: HashMap<VarKey, VarKey>,
    local_aliases: HashMap<u32, VarKey>,
    zero_tests: HashMap<VarKey, EqZeroWitness>,
    local_zero_tests: HashMap<u32, EqZeroWitness>,
    nonzero_identities: HashSet<VarKey>,
}

impl AdviceState {
    /// Return the alias identity currently attached to a tracked place, if any.
    fn identity_for_place(&self, place: &AdvicePlace) -> Option<VarKey> {
        match place {
            AdvicePlace::Var(key) => Some(self.aliases.get(key).cloned().unwrap_or_else(|| key.clone())),
            AdvicePlace::Local(slot) => self.local_aliases.get(slot).cloned(),
        }
    }

    /// Return all currently tracked places that alias the given runtime value.
    fn aliased_places(&self, identity: &VarKey) -> HashSet<AdvicePlace> {
        self.facts
            .keys()
            .cloned()
            .chain(self.u32_validity.keys().cloned())
            .chain(self.aliases.keys().cloned().map(AdvicePlace::Var))
            .chain(self.local_aliases.keys().copied().map(AdvicePlace::Local))
            .filter(|place| self.identity_for_place(place).as_ref() == Some(identity))
            .collect()
    }

    /// Set a `u32` validity fact for every currently tracked alias of the identity.
    fn set_identity_u32_validity(&mut self, identity: &VarKey, validity: U32Validity) {
        for place in self.aliased_places(identity) {
            self.set_place_u32_validity(place, validity);
        }
        if validity.is_proven() {
            self.u32_valid_identities.insert(identity.clone());
        } else {
            self.u32_valid_identities.remove(identity);
        }
    }

    /// Refresh the cached proof bit for one alias identity after a state mutation.
    fn refresh_u32_identity_cache(&mut self, identity: &VarKey) {
        if self
            .aliased_places(identity)
            .iter()
            .any(|place| self.u32_validity.get(place).copied().unwrap_or(U32Validity::Unknown).is_proven())
        {
            self.u32_valid_identities.insert(identity.clone());
        } else {
            self.u32_valid_identities.remove(identity);
        }
    }

    /// Rebuild the alias-identity cache from the current place-level facts.
    fn rebuild_u32_identity_cache(&mut self) {
        self.u32_valid_identities.clear();
        let identities = self
            .u32_validity
            .keys()
            .filter_map(|place| self.identity_for_place(place))
            .collect::<HashSet<_>>();
        for identity in identities {
            self.refresh_u32_identity_cache(&identity);
        }
    }

    /// Read the current fact for a tracked place.
    pub(crate) fn fact_for_place(&self, place: &AdvicePlace) -> AdviceFact {
        self.facts
            .get(place)
            .cloned()
            .unwrap_or_else(AdviceFact::bottom)
    }

    /// Set the current fact for a tracked place.
    pub(crate) fn set_place_fact(&mut self, place: AdvicePlace, fact: AdviceFact) {
        let identity = self.identity_for_place(&place);
        self.u32_validity.remove(&place);
        if let Some(identity) = identity {
            self.refresh_u32_identity_cache(&identity);
        }
        self.facts.insert(place, fact);
    }

    /// Set the current `u32` validity fact for a tracked place.
    pub(crate) fn set_place_u32_validity(&mut self, place: AdvicePlace, validity: U32Validity) {
        let identity = self.identity_for_place(&place);
        if validity.is_proven() {
            self.u32_validity.insert(place, validity);
        } else {
            self.u32_validity.remove(&place);
        }
        if let Some(identity) = identity {
            self.refresh_u32_identity_cache(&identity);
        }
    }

    /// Read the current `u32` validity fact for a tracked place.
    pub(crate) fn u32_validity_for_place(&self, place: &AdvicePlace) -> U32Validity {
        let direct = self
            .u32_validity
            .get(place)
            .copied()
            .unwrap_or(U32Validity::Unknown);
        if direct.is_proven() {
            return direct;
        }
        self.identity_for_place(place)
            .filter(|identity| self.u32_valid_identities.contains(identity))
            .map(|_| U32Validity::ProvenU32)
            .unwrap_or(U32Validity::Unknown)
    }

    /// Read the current fact for a variable.
    pub(crate) fn fact_for_var(&self, var: &Var) -> AdviceFact {
        self.fact_for_place(&AdvicePlace::var(var))
    }

    /// Set the current fact for a variable.
    pub(crate) fn set_var_fact(&mut self, var: &Var, fact: AdviceFact) {
        self.set_place_fact(AdvicePlace::var(var), fact);
    }

    /// Read the current `u32` validity fact for a variable.
    pub(crate) fn u32_validity_for_var(&self, var: &Var) -> U32Validity {
        self.u32_validity_for_place(&AdvicePlace::var(var))
    }

    /// Set the current `u32` validity fact for a variable.
    pub(crate) fn set_var_u32_validity(&mut self, var: &Var, validity: U32Validity) {
        self.set_place_u32_validity(AdvicePlace::var(var), validity);
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
        let old_identity = self.identity_for_var(var);
        if identity == key {
            self.aliases.remove(&key);
        } else {
            self.aliases.insert(key, identity);
        }
        self.refresh_u32_identity_cache(&old_identity);
        self.refresh_u32_identity_cache(&self.identity_for_var(var));
    }

    /// Clear any alias identity for a variable.
    pub(crate) fn clear_var_identity(&mut self, var: &Var) {
        let old_identity = self.identity_for_var(var);
        self.aliases.remove(&VarKey::from_var(var));
        self.refresh_u32_identity_cache(&old_identity);
        self.refresh_u32_identity_cache(&self.identity_for_var(var));
    }

    /// Set the alias identity for a local slot.
    pub(crate) fn set_local_identity(&mut self, slot: u32, identity: Option<VarKey>) {
        let old_identity = self.identity_for_local(slot);
        match identity {
            Some(identity) => {
                self.local_aliases.insert(slot, identity);
            }
            None => {
                self.local_aliases.remove(&slot);
            }
        }
        if let Some(identity) = old_identity {
            self.refresh_u32_identity_cache(&identity);
        }
        if let Some(identity) = self.identity_for_local(slot) {
            self.refresh_u32_identity_cache(&identity);
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
        let identity = self.identity_for_var(var);
        self.set_identity_u32_validity(&identity, U32Validity::ProvenU32);
    }

    /// Read the current fact for a local slot.
    pub(crate) fn fact_for_local(&self, slot: u32) -> AdviceFact {
        self.fact_for_place(&AdvicePlace::local(slot))
    }

    /// Read the current `u32` validity fact for a local slot.
    pub(crate) fn u32_validity_for_local(&self, slot: u32) -> U32Validity {
        self.u32_validity_for_place(&AdvicePlace::local(slot))
    }

    /// Set the current fact for a local slot.
    pub(crate) fn set_local_fact(&mut self, slot: u32, fact: AdviceFact) {
        self.set_place_fact(AdvicePlace::local(slot), fact);
    }

    /// Set the current `u32` validity fact for a local slot.
    pub(crate) fn set_local_u32_validity(&mut self, slot: u32, validity: U32Validity) {
        self.set_place_u32_validity(AdvicePlace::local(slot), validity);
    }

    /// Return the current `u32` validity fact for a tracked place.
    #[cfg(test)]
    pub(crate) fn place_u32_validity(&self, place: &AdvicePlace) -> U32Validity {
        self.u32_validity_for_place(place)
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
        let proven_identities = self
            .u32_valid_identities
            .intersection(&other.u32_valid_identities)
            .cloned()
            .collect::<HashSet<_>>();
        let tracked_places = joined
            .facts
            .keys()
            .cloned()
            .chain(self.u32_validity.keys().cloned())
            .chain(other.u32_validity.keys().cloned())
            .chain(joined.aliases.keys().cloned().map(AdvicePlace::Var))
            .chain(joined.local_aliases.keys().copied().map(AdvicePlace::Local))
            .collect::<HashSet<_>>();
        joined.u32_validity.clear();
        joined.u32_valid_identities.clear();
        for place in tracked_places {
            let direct_validity = self
                .u32_validity
                .get(&place)
                .copied()
                .unwrap_or(U32Validity::Unknown)
                .join(
                    other
                        .u32_validity
                        .get(&place)
                        .copied()
                        .unwrap_or(U32Validity::Unknown),
                );
            let identity_validity = joined
                .identity_for_place(&place)
                .filter(|identity| proven_identities.contains(identity))
                .map(|_| U32Validity::ProvenU32)
                .unwrap_or(U32Validity::Unknown);
            let validity = if direct_validity.is_proven() || identity_validity.is_proven() {
                U32Validity::ProvenU32
            } else {
                U32Validity::Unknown
            };
            joined.set_place_u32_validity(place, validity);
        }
        joined.rebuild_u32_identity_cache();
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
        unconstrained_advice::{domain::AdviceFact, u32_domain::U32Validity},
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

    #[test]
    fn join_drops_u32_proof_when_only_one_path_validates() {
        let mut lhs = AdviceState::default();
        let rhs = AdviceState::default();
        lhs.set_place_u32_validity(AdvicePlace::local(0), U32Validity::ProvenU32);

        let joined = lhs.join(&rhs);

        assert!(joined.u32_validity.is_empty());
    }

    #[test]
    fn overwriting_a_place_clears_stale_u32_proof() {
        let var = test_var(0);
        let mut state = AdviceState::default();
        state.set_place_u32_validity(AdvicePlace::var(&var), U32Validity::ProvenU32);
        state.set_place_u32_validity(AdvicePlace::local(0), U32Validity::ProvenU32);

        state.set_var_fact(&var, AdviceFact::from_input(0));
        state.set_local_fact(0, AdviceFact::from_input(1));

        assert_eq!(
            state.place_u32_validity(&AdvicePlace::var(&var)),
            U32Validity::Unknown
        );
        assert_eq!(
            state.place_u32_validity(&AdvicePlace::local(0)),
            U32Validity::Unknown
        );
    }

    #[test]
    fn sanitizing_one_alias_marks_all_current_aliases_as_proven_u32() {
        let source = test_var(0);
        let copy = test_var(1);
        let mut state = AdviceState::default();
        let identity = state.identity_for_var(&source);
        state.set_var_fact(&source, AdviceFact::from_input(0));
        state.set_var_fact(&copy, AdviceFact::from_input(0));
        state.set_var_identity(&copy, identity.clone());
        state.set_local_fact(0, AdviceFact::from_input(0));
        state.set_local_identity(0, Some(identity));

        state.sanitize_var(&source);

        assert_eq!(
            state.place_u32_validity(&AdvicePlace::var(&source)),
            U32Validity::ProvenU32
        );
        assert_eq!(
            state.place_u32_validity(&AdvicePlace::var(&copy)),
            U32Validity::ProvenU32
        );
        assert_eq!(
            state.place_u32_validity(&AdvicePlace::local(0)),
            U32Validity::ProvenU32
        );
    }

    #[test]
    fn join_preserves_u32_proof_when_each_path_validates_a_different_alias() {
        let source = test_var(0);
        let alias = test_var(1);
        let mut lhs = AdviceState::default();
        let mut rhs = AdviceState::default();
        let identity = lhs.identity_for_var(&source);

        for state in [&mut lhs, &mut rhs] {
            state.set_var_fact(&source, AdviceFact::from_input(0));
            state.set_var_fact(&alias, AdviceFact::from_input(0));
            state.set_var_identity(&alias, identity.clone());
        }

        lhs.set_var_u32_validity(&source, U32Validity::ProvenU32);
        rhs.set_var_u32_validity(&alias, U32Validity::ProvenU32);

        let joined = lhs.join(&rhs);

        assert_eq!(
            joined.place_u32_validity(&AdvicePlace::var(&source)),
            U32Validity::ProvenU32
        );
        assert_eq!(
            joined.place_u32_validity(&AdvicePlace::var(&alias)),
            U32Validity::ProvenU32
        );
    }

    #[test]
    fn join_preserves_direct_local_u32_proof_without_alias_identity() {
        let mut lhs = AdviceState::default();
        let mut rhs = AdviceState::default();
        lhs.set_local_u32_validity(0, U32Validity::ProvenU32);
        rhs.set_local_u32_validity(0, U32Validity::ProvenU32);

        let joined = lhs.join(&rhs);

        assert_eq!(
            joined.place_u32_validity(&AdvicePlace::local(0)),
            U32Validity::ProvenU32
        );
    }
}
