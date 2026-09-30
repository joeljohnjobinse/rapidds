from dataclasses import dataclass


@dataclass(frozen=True)
class CleaningPolicy:
    """Controls which suggestion actions auto_clean may execute."""
    name: str
    max_risk: str
    min_confidence: str
    allow_drop_columns: bool = False
    allow_standardize_case: bool = False
    allow_remove_outliers: bool = False
    allow_datetime_features: bool = False

    @classmethod
    def from_name(cls, name="conservative"):
        name = str(name).lower()
        policies = {
            "conservative": cls("conservative", "low", "high"),
            "balanced": cls("balanced", "medium", "high", allow_drop_columns=True, allow_standardize_case=True),
            "aggressive": cls("aggressive", "high", "medium", allow_drop_columns=True,
                               allow_standardize_case=True, allow_remove_outliers=True,
                               allow_datetime_features=True),
        }
        if name not in policies:
            raise ValueError("Unknown policy. Choose conservative, balanced, or aggressive.")
        return policies[name]


_LEVEL = {"low": 1, "medium": 2, "high": 3}


def action_allowed(suggestion, policy):
    """Return whether a suggestion is safe enough for the selected policy."""
    if _LEVEL.get(suggestion.get("risk", "high"), 3) > _LEVEL[policy.max_risk]:
        return False
    if _LEVEL.get(suggestion.get("confidence", "low"), 1) < _LEVEL[policy.min_confidence]:
        return False
    action = suggestion.get("action")
    if action == "drop_column":
        return policy.allow_drop_columns
    if action == "standardize_case":
        return policy.allow_standardize_case
    if action in {"remove_outliers", "clip_outliers"}:
        return policy.allow_remove_outliers
    if action == "add_datetime_features":
        return policy.allow_datetime_features
    # High-risk investigative suggestions should never silently mutate data.
    return action in {
        "drop_duplicates", "impute", "strip_whitespace", "convert_numeric",
        "drop_constant", "drop_low_information"
    }
