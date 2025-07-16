import pandas as pd
from itertools import combinations
import numpy as np


def generate_multi_patterns(df, max_predicates=2, bin_numerical=True, bins=3):
    """
    Generate multi-predicate patterns by combining simple feature-value conditions.
    
    Creates conjunctions of simple patterns (feature-value pairs) up to a specified 
    maximum number of predicates. Handles conflicts where multiple values are specified 
    for the same feature.
    
    Args:
        df: DataFrame containing the data
        max_predicates: Maximum number of conditions to combine (default: 2)
        bin_numerical: Whether to bin numerical features (default: True)
        bins: Number of bins for numerical features (default: 3)
        
    Returns:
        List of dictionaries representing multi-predicate patterns
    """
    # First generate all simple patterns (single feature-value pairs)
    simple_patterns = generate_simple_patterns(df, bin_numerical, bins)
    patterns = []
    
    # Generate combinations of different sizes (from 1 to max_predicates)
    for r in range(1, max_predicates + 1):
        for combo in combinations(simple_patterns, r):
            merged = {}
            conflict = False

            # Merge all predicates in the combination
            for p in combo:
                for k, v in p.items():
                    # Check for conflicting values for same feature
                    if k in merged and merged[k] != v:
                        conflict = True
                        break
                    merged[k] = v
                if conflict:
                    break
            # Add only conflict-free patterns
            if not conflict:
                patterns.append(merged)
    return patterns


def generate_simple_patterns(df, target_col, bin_numerical=True, bins=3):
    """
    Generate all possible simple patterns (single feature-value conditions).
    
    For numerical features, bins them into quantile-based intervals when requested.
    
    Args:
        df: DataFrame containing the data
        bin_numerical: Whether to bin numerical features (default: True)
        bins: Number of bins for numerical features (default: 3)
        
    Returns:
        List of dictionaries representing single-feature patterns
    """
    ignore_columns = ['influence', target_col]
    patterns = []
    feature_cols = [c for c in df.columns if c not in ignore_columns]
    patterns = []
    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(df[col]) and bin_numerical:
            binned = pd.qcut(df[col], bins, duplicates="drop")
            for interval in binned.unique():
                patterns.append({col: interval})
        else:
            for val in df[col].unique():
                patterns.append({col: val})
    return patterns


def pattern_support(df, pattern):
    """
    Calculate the support for a given pattern in the dataset.
    
    Args:
        df: DataFrame containing the data
        pattern: Dictionary of feature-value conditions
        
    Returns:
        subset: DataFrame rows matching the pattern
        support_ratio: Fraction of dataset matching the pattern
    """
    mask = np.ones(len(df), dtype=bool)
    for col, val in pattern.items():
        if isinstance(val, pd.Interval):
            mask &= df[col].between(val.left, val.right, inclusive="left")
        else:
            mask &= df[col] == val
    return df[mask], mask.sum() / len(df)


def evaluate_patterns(df, target_col, min_support=0.01, top_k=5, max_predicates=1):
    """
    Evaluate and rank patterns by their interestingness score.
    
    Interestingness = Responsibility / Support
    Responsibility = Sum(influence) of matching rows / Total absolute influence
    
    Args:
        df: DataFrame with 'influence' column and features
        target_col: Target column name (unused in current implementation)
        min_support: Minimum support threshold (default: 0.01)
        top_k: Number of top patterns to return (default: 5)
        max_predicates: Maximum predicates in patterns (default: 1)
        
    Returns:
        DataFrame with top_k patterns and their metrics
    """
    if max_predicates == 1:
        patterns = generate_simple_patterns(df, target_col)
    else:
        patterns = generate_multi_patterns(df, max_predicates=max_predicates)

    results = []
    total_influence = abs(df["influence"].sum())

    for pattern in patterns:
        subset, support = pattern_support(df, pattern)
        if support < min_support:
            continue

        responsibility = subset["influence"].sum() / total_influence
        interestingness = responsibility / support if support > 0 else 0

        keys = list(pattern.keys())
        vals = list(pattern.values())

        row = {
            "pattern_col_1": keys[0] if len(keys) > 0 else None,
            "pattern_val_1": str(vals[0]) if len(vals) > 0 else None,
            "pattern_col_2": keys[1] if len(keys) > 1 else None,
            "pattern_val_2": str(vals[1]) if len(vals) > 1 else None,
            "support": round(support, 4),
            "responsibility": round(responsibility, 4),
            "interestingness": round(interestingness, 4),
        }
        results.append(row)
    results = sorted(results, key=lambda x: -x["interestingness"])
    return pd.DataFrame(results[:top_k])


"""unused function"""


def evaluate_given_pattern(df, pattern):
    subset, support = pattern_support(df, pattern)
    total_influence = df["influence"].sum()
    responsibility = subset["influence"].sum() / total_influence
    interestingness = responsibility / support if support > 0 else 0
    results = {
        "pattern": pattern,
        "support": support,
        "responsibility": responsibility,
        "interestingness": interestingness,
    }
    return results
