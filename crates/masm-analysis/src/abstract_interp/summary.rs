//! Reusable interprocedural summary primitives for MASM analyses.

/// Precision status attached to one procedure summary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SummaryStatus {
    /// The summary was computed from an analysis-visible procedure body.
    Known,
    /// The summary is intentionally opaque or incomplete.
    Opaque,
}

/// Interprocedural summary together with its precision status.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Summary<T> {
    value: T,
    status: SummaryStatus,
}

impl<T> Summary<T> {
    /// Build a known summary value.
    pub fn known(value: T) -> Self {
        Self {
            value,
            status: SummaryStatus::Known,
        }
    }

    /// Build an opaque summary value.
    pub fn opaque(value: T) -> Self {
        Self {
            value,
            status: SummaryStatus::Opaque,
        }
    }

    /// Return the summary precision status.
    pub fn status(&self) -> SummaryStatus {
        self.status
    }

    /// Return `true` when this summary is known.
    pub fn is_known(&self) -> bool {
        self.status == SummaryStatus::Known
    }

    /// Return `true` when this summary is opaque.
    pub fn is_opaque(&self) -> bool {
        self.status == SummaryStatus::Opaque
    }

    /// Return the summarized value.
    pub fn value(&self) -> &T {
        &self.value
    }

    /// Consume the summary and return the summarized value.
    pub fn into_value(self) -> T {
        self.value
    }
}

#[cfg(test)]
mod tests {
    use super::{Summary, SummaryStatus};

    #[test]
    fn known_summary_reports_known_status() {
        let summary = Summary::known(vec![1u8, 2, 3]);

        assert!(summary.is_known());
        assert!(!summary.is_opaque());
        assert_eq!(summary.status(), SummaryStatus::Known);
        assert_eq!(summary.value(), &[1, 2, 3]);
    }

    #[test]
    fn opaque_summary_reports_opaque_status() {
        let summary = Summary::opaque(vec![0u8; 2]);

        assert!(!summary.is_known());
        assert!(summary.is_opaque());
        assert_eq!(summary.status(), SummaryStatus::Opaque);
        assert_eq!(summary.into_value(), vec![0u8; 2]);
    }
}
