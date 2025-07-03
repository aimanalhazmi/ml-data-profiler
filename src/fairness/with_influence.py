import pandas as pd
from itertools import combinations
import numpy as np


def check_fairness(df: pd.DataFrame, influence_f):
    pass


def generate_multi_patterns(df, max_predicates=2, bin_numerical=True, bins=3):
    simple_patterns = generate_simple_patterns(df, bin_numerical, bins)
    patterns = []

    for r in range(1, max_predicates + 1):
        for combo in combinations(simple_patterns, r):
            merged = {}
            conflict = False
            for p in combo:
                for k, v in p.items():
                    if k in merged and merged[k] != v:
                        conflict = True
                        break
                    merged[k] = v
                if conflict:
                    break
            if not conflict:
                patterns.append(merged)
    return patterns

def generate_simple_patterns(df, bin_numerical=True, bins=3):
    patterns = []
    for col in df.columns[:-1]:  # exclude 'influence'
        if pd.api.types.is_numeric_dtype(df[col]) and bin_numerical:
            binned = pd.qcut(df[col], bins, duplicates='drop')
            for interval in binned.unique():
                patterns.append({col: interval})
        else:
            for val in df[col].unique():
                patterns.append({col: val})
    return patterns

def pattern_support(df, pattern):
    mask = np.ones(len(df), dtype=bool)
    for col, val in pattern.items():
        if isinstance(val, pd.Interval):
            mask &= df[col].between(val.left, val.right, inclusive='left')
        else:
            mask &= df[col] == val
    return df[mask], mask.sum() / len(df)

def evaluate_patterns(df, min_support=0.01, top_k=5, max_predicates=1):
    """ top-k interestingness single/multi pattern"""
    if max_predicates == 1:
        patterns = generate_simple_patterns(df)
    else:
        patterns = generate_multi_patterns(df, max_predicates=max_predicates)

    results = []
    total_influence = df["influence"].sum()

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
            "support": support,
            "responsibility": responsibility,
            "interestingness": interestingness,
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
    results ={
        "pattern": pattern,
        "support": support,
        "responsibility": responsibility,
        "interestingness": interestingness,
    }
    return results

"""unused function"""
def pattern_to_readable(pattern, columns):
    readable = []
    for col_idx, val in pattern.items():
        col_name = columns[col_idx] if isinstance(col_idx, int) else col_idx
        if isinstance(val, pd.Interval):
            readable.append(f"{col_name} ∈ {val}")
        else:
            readable.append(f"{col_name} = {val}")
    return readable


def print_pattern_table(df_patterns):
    print(f"{'Pattern':<60} | {'Support':<8} | {'Resp.':<8} | {'Interest.':<10}")

    for _, row in df_patterns.iterrows():
        cond1 = f"{row['pattern_col_1']} = {row['pattern_val_1']}"
        cond2 = (
            f" AND {row['pattern_col_2']} = {row['pattern_val_2']}"
            if pd.notna(row['pattern_col_2'])
            else ""
        )
        pattern_str = (cond1 + cond2)[:60]

        print(f"{pattern_str:<60} | {row['support']:<8.3f} | {row['responsibility']:<8.3f} | {row['interestingness']:<10.3f}")
