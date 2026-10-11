#!/usr/bin/env python3
"""Compare a rerun with the published records, column by column.

    python3 scripts/compare_results.py results results-rerun

For every results CSV present in both directories, rows are matched on the columns that identify a
run (size, seed, method, ...), and every deterministic column is compared exactly as written.
Elapsed time is excluded: it is the only column that depends on the machine and its load.
timed.csv is reported but not compared, because its stopping rule is a clock.  A file present in
only one directory is listed, so a partial rerun (one experiment, or --only some methods) can be
checked against the matching rows.  Exits non-zero if any compared value differs.

Standard library only.
"""
import csv
import os
import sys

KEYS = {
    "tuning.csv": ["method", "config", "param"],
    "tuning_runs.csv": ["method", "config", "n", "seed"],
    "compare.csv": ["n", "seed", "method"],
    "agreement.csv": ["n", "seed", "method"],
    "budgetmode.csv": ["n", "seed", "method", "currency"],
    "budgetmode_tuned.csv": ["n", "seed", "method", "currency"],
    "evalcost.csv": ["n", "method", "seed"],
    "evalcost_tuned.csv": ["n", "method", "seed"],
    "converge.csv": ["n", "method", "seed", "work_budget"],
    "crossover.csv": ["n", "seed", "config", "arm"],
}
CLOCK_COLUMNS = {"elapsed_ms"}
CLOCK_FILES = {"timed.csv"}


def load(path, key):
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    return {tuple(r[k] for k in key): r for r in rows}


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    a_dir, b_dir = sys.argv[1], sys.argv[2]
    names = sorted(set(os.listdir(a_dir)) | set(os.listdir(b_dir)))
    differing = 0
    for name in names:
        if not name.endswith(".csv"):
            continue
        in_a = os.path.exists(os.path.join(a_dir, name))
        in_b = os.path.exists(os.path.join(b_dir, name))
        if not (in_a and in_b):
            print(f"  only in {a_dir if in_a else b_dir}: {name}")
            continue
        if name in CLOCK_FILES:
            print(f"  {name}: not compared (its runs stop on a wall-clock limit)")
            continue
        key = KEYS.get(name)
        if key is None:
            print(f"  {name}: no key defined, skipped")
            continue
        a, b = load(os.path.join(a_dir, name), key), load(os.path.join(b_dir, name), key)
        common = sorted(set(a) & set(b))
        bad = 0
        for k in common:
            cols = [c for c in a[k] if c in b[k] and c not in CLOCK_COLUMNS]
            diffs = [c for c in cols if a[k][c] != b[k][c]]
            if diffs:
                bad += 1
                if bad <= 5:
                    print(f"    DIFF {name} {k}: " +
                          ", ".join(f"{c} {a[k][c]} -> {b[k][c]}" for c in diffs[:4]))
        differing += bad
        print(f"  {name}: {len(common)} matching runs compared, {bad} differ"
              f" ({len(a) - len(common)} only in {a_dir}, {len(b) - len(common)} only in {b_dir})")
    print("identical in every compared deterministic column" if differing == 0
          else f"{differing} run(s) differ")
    return 1 if differing else 0


if __name__ == "__main__":
    sys.exit(main())
