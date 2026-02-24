//! Metadata filtering with dot-notation field access.
//!
//! Filters operate on `serde_json::Value` metadata attached to documents.
//! Dot-notation (e.g. `"tags.entity"`) resolves nested fields.

use serde_json::Value;

/// A predicate over document metadata.
#[derive(Debug, Clone)]
pub enum Filter {
    /// Field equals value.
    Eq { field: String, value: Value },
    /// Field does not equal value.
    ///
    /// Returns `true` when the field is missing or `None`, following SQL-like
    /// NULL semantics where a missing value is considered "not equal" to any value.
    Ne { field: String, value: Value },
    /// Field value is in the given set.
    In { field: String, values: Vec<Value> },
    /// Field value is greater than the given value.
    Gt { field: String, value: Value },
    /// Field value is less than the given value.
    Lt { field: String, value: Value },
    /// Field value is greater than or equal to the given value.
    Gte { field: String, value: Value },
    /// Field value is less than or equal to the given value.
    Lte { field: String, value: Value },
    /// All sub-filters must match.
    And(Vec<Filter>),
    /// At least one sub-filter must match.
    Or(Vec<Filter>),
    /// Negation.
    Not(Box<Filter>),
    /// Field exists (is not null/missing).
    Exists(String),
}

impl Filter {
    // Convenience constructors.

    pub fn eq(field: impl Into<String>, value: impl Into<Value>) -> Self {
        Self::Eq {
            field: field.into(),
            value: value.into(),
        }
    }

    pub fn ne(field: impl Into<String>, value: impl Into<Value>) -> Self {
        Self::Ne {
            field: field.into(),
            value: value.into(),
        }
    }

    pub fn and(filters: Vec<Filter>) -> Self {
        Self::And(filters)
    }

    pub fn or(filters: Vec<Filter>) -> Self {
        Self::Or(filters)
    }

    #[allow(clippy::should_implement_trait)]
    pub fn not(filter: Filter) -> Self {
        Self::Not(Box::new(filter))
    }

    pub fn exists(field: impl Into<String>) -> Self {
        Self::Exists(field.into())
    }

    pub fn is_in(field: impl Into<String>, values: Vec<Value>) -> Self {
        Self::In {
            field: field.into(),
            values,
        }
    }

    pub fn gt(field: impl Into<String>, value: impl Into<Value>) -> Self {
        Self::Gt {
            field: field.into(),
            value: value.into(),
        }
    }

    pub fn lt(field: impl Into<String>, value: impl Into<Value>) -> Self {
        Self::Lt {
            field: field.into(),
            value: value.into(),
        }
    }

    pub fn gte(field: impl Into<String>, value: impl Into<Value>) -> Self {
        Self::Gte {
            field: field.into(),
            value: value.into(),
        }
    }

    pub fn lte(field: impl Into<String>, value: impl Into<Value>) -> Self {
        Self::Lte {
            field: field.into(),
            value: value.into(),
        }
    }

    /// Test whether `metadata` satisfies this filter.
    ///
    /// Returns `false` when metadata is `None`.
    pub fn matches(&self, metadata: Option<&Value>) -> bool {
        let Some(root) = metadata else {
            return false;
        };
        self.eval(root)
    }

    fn eval(&self, root: &Value) -> bool {
        match self {
            Filter::Eq { field, value } => {
                resolve(root, field).map(|v| v == value).unwrap_or(false)
            }
            Filter::Ne { field, value } => resolve(root, field).map(|v| v != value).unwrap_or(true),
            Filter::In { field, values } => resolve(root, field)
                .map(|v| values.contains(v))
                .unwrap_or(false),
            Filter::Gt { field, value } => resolve(root, field)
                .and_then(|v| json_cmp(v, value))
                .map(|ord| ord == std::cmp::Ordering::Greater)
                .unwrap_or(false),
            Filter::Lt { field, value } => resolve(root, field)
                .and_then(|v| json_cmp(v, value))
                .map(|ord| ord == std::cmp::Ordering::Less)
                .unwrap_or(false),
            Filter::Gte { field, value } => resolve(root, field)
                .and_then(|v| json_cmp(v, value))
                .map(|ord| ord != std::cmp::Ordering::Less)
                .unwrap_or(false),
            Filter::Lte { field, value } => resolve(root, field)
                .and_then(|v| json_cmp(v, value))
                .map(|ord| ord != std::cmp::Ordering::Greater)
                .unwrap_or(false),
            Filter::And(filters) => filters.iter().all(|f| f.eval(root)),
            Filter::Or(filters) => filters.iter().any(|f| f.eval(root)),
            Filter::Not(inner) => !inner.eval(root),
            Filter::Exists(field) => resolve(root, field).is_some(),
        }
    }
}

/// Compare two JSON values. Supports numbers (coerced to f64) and strings.
/// Returns `None` for incompatible types or NaN.
fn json_cmp(a: &Value, b: &Value) -> Option<std::cmp::Ordering> {
    match (a, b) {
        (Value::Number(na), Value::Number(nb)) => {
            let fa = na.as_f64()?;
            let fb = nb.as_f64()?;
            fa.partial_cmp(&fb)
        }
        (Value::String(sa), Value::String(sb)) => Some(sa.cmp(sb)),
        _ => None,
    }
}

/// Resolve a dot-notation path against a JSON value.
///
/// `"a.b.c"` walks `root["a"]["b"]["c"]`.
fn resolve<'a>(root: &'a Value, path: &str) -> Option<&'a Value> {
    let mut current = root;
    for segment in path.split('.') {
        match current {
            Value::Object(map) => {
                current = map.get(segment)?;
            }
            _ => return None,
        }
    }
    Some(current)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn meta() -> Value {
        json!({
            "scope": "workspace",
            "tags": {
                "entity": "owner",
                "priority": 5
            },
            "active": true
        })
    }

    #[test]
    fn eq_top_level() {
        let f = Filter::eq("scope", "workspace");
        assert!(f.matches(Some(&meta())));
    }

    #[test]
    fn eq_nested_dot_notation() {
        let f = Filter::eq("tags.entity", "owner");
        assert!(f.matches(Some(&meta())));
    }

    #[test]
    fn eq_nested_number() {
        let f = Filter::eq("tags.priority", json!(5));
        assert!(f.matches(Some(&meta())));
    }

    #[test]
    fn eq_mismatch() {
        let f = Filter::eq("scope", "session");
        assert!(!f.matches(Some(&meta())));
    }

    #[test]
    fn ne_basic() {
        let f = Filter::ne("scope", "session");
        assert!(f.matches(Some(&meta())));
    }

    #[test]
    fn ne_same_value() {
        let f = Filter::ne("scope", "workspace");
        assert!(!f.matches(Some(&meta())));
    }

    #[test]
    fn ne_missing_field() {
        // Missing field != anything → true (field absent means "not equal")
        let f = Filter::ne("nonexistent", "value");
        assert!(f.matches(Some(&meta())));
    }

    #[test]
    fn and_both_true() {
        let f = Filter::and(vec![
            Filter::eq("scope", "workspace"),
            Filter::eq("active", true),
        ]);
        assert!(f.matches(Some(&meta())));
    }

    #[test]
    fn and_one_false() {
        let f = Filter::and(vec![
            Filter::eq("scope", "workspace"),
            Filter::eq("active", false),
        ]);
        assert!(!f.matches(Some(&meta())));
    }

    #[test]
    fn or_one_true() {
        let f = Filter::or(vec![
            Filter::eq("scope", "session"),
            Filter::eq("scope", "workspace"),
        ]);
        assert!(f.matches(Some(&meta())));
    }

    #[test]
    fn or_none_true() {
        let f = Filter::or(vec![
            Filter::eq("scope", "session"),
            Filter::eq("scope", "global"),
        ]);
        assert!(!f.matches(Some(&meta())));
    }

    #[test]
    fn not_filter() {
        let f = Filter::not(Filter::eq("scope", "session"));
        assert!(f.matches(Some(&meta())));

        let f2 = Filter::not(Filter::eq("scope", "workspace"));
        assert!(!f2.matches(Some(&meta())));
    }

    #[test]
    fn exists_present() {
        let f = Filter::exists("tags.entity");
        assert!(f.matches(Some(&meta())));
    }

    #[test]
    fn exists_missing() {
        let f = Filter::exists("tags.nonexistent");
        assert!(!f.matches(Some(&meta())));
    }

    #[test]
    fn none_metadata_always_false() {
        let f = Filter::eq("scope", "workspace");
        assert!(!f.matches(None));
    }

    #[test]
    fn empty_and_is_true() {
        let f = Filter::and(vec![]);
        assert!(f.matches(Some(&meta())));
    }

    #[test]
    fn empty_or_is_false() {
        let f = Filter::or(vec![]);
        assert!(!f.matches(Some(&meta())));
    }

    // ── In ──────────────────────────────────────────────────────────

    #[test]
    fn in_match() {
        let f = Filter::is_in("scope", vec![json!("workspace"), json!("global")]);
        assert!(f.matches(Some(&meta())));
    }

    #[test]
    fn in_no_match() {
        let f = Filter::is_in("scope", vec![json!("session"), json!("global")]);
        assert!(!f.matches(Some(&meta())));
    }

    #[test]
    fn in_missing_field() {
        let f = Filter::is_in("nonexistent", vec![json!("a")]);
        assert!(!f.matches(Some(&meta())));
    }

    // ── Gt / Lt / Gte / Lte ────────────────────────────────────────

    #[test]
    fn gt_numeric() {
        let f = Filter::gt("tags.priority", json!(4));
        assert!(f.matches(Some(&meta()))); // 5 > 4
    }

    #[test]
    fn gt_equal_is_false() {
        let f = Filter::gt("tags.priority", json!(5));
        assert!(!f.matches(Some(&meta()))); // 5 > 5 → false
    }

    #[test]
    fn lt_numeric() {
        let f = Filter::lt("tags.priority", json!(6));
        assert!(f.matches(Some(&meta()))); // 5 < 6
    }

    #[test]
    fn lt_equal_is_false() {
        let f = Filter::lt("tags.priority", json!(5));
        assert!(!f.matches(Some(&meta()))); // 5 < 5 → false
    }

    #[test]
    fn gte_equal() {
        let f = Filter::gte("tags.priority", json!(5));
        assert!(f.matches(Some(&meta()))); // 5 >= 5
    }

    #[test]
    fn gte_below() {
        let f = Filter::gte("tags.priority", json!(6));
        assert!(!f.matches(Some(&meta()))); // 5 >= 6 → false
    }

    #[test]
    fn lte_equal() {
        let f = Filter::lte("tags.priority", json!(5));
        assert!(f.matches(Some(&meta()))); // 5 <= 5
    }

    #[test]
    fn lte_above() {
        let f = Filter::lte("tags.priority", json!(4));
        assert!(!f.matches(Some(&meta()))); // 5 <= 4 → false
    }

    #[test]
    fn string_lexicographic_comparison() {
        let f = Filter::gt("scope", json!("session"));
        assert!(f.matches(Some(&meta()))); // "workspace" > "session"

        let f2 = Filter::lt("scope", json!("z"));
        assert!(f2.matches(Some(&meta()))); // "workspace" < "z"
    }

    #[test]
    fn comparison_incompatible_types() {
        // Comparing number to string → None → false
        let f = Filter::gt("tags.priority", json!("hello"));
        assert!(!f.matches(Some(&meta())));
    }

    #[test]
    fn comparison_none_metadata() {
        let f = Filter::gt("tags.priority", json!(1));
        assert!(!f.matches(None));
    }
}
