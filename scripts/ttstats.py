"""Statistics and LaTeX emitters for the experiment analyses.

Standard library only: no numpy, no scipy, no pandas.  The point is that `python3 analyse.py` works
on any machine that can run the experiments, and that every number in the paper can be traced to a
line of a results file by reading one short script.
"""

from __future__ import annotations

import csv
import itertools
import math
import os
import time
from collections import defaultdict

# ---------------------------------------------------------------------------------------------
#  Reading
# ---------------------------------------------------------------------------------------------


def read(path):
    """Read a results CSV into a list of dicts, dropping incomplete trailing rows."""
    with open(path, newline="") as f:
        rows = [r for r in csv.DictReader(f) if all(v is not None for v in r.values())]
    return rows


def num(row, key, default=0.0):
    try:
        return float(row[key])
    except (TypeError, ValueError, KeyError):
        return default


def group(rows, *keys):
    out = defaultdict(list)
    for r in rows:
        out[tuple(r[k] for k in keys)].append(r)
    return out


# ---------------------------------------------------------------------------------------------
#  Descriptive statistics
# ---------------------------------------------------------------------------------------------


def median(xs):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    m = len(xs) // 2
    return xs[m] if len(xs) % 2 else 0.5 * (xs[m - 1] + xs[m])


def quantile(xs, q):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    if len(xs) == 1:
        return xs[0]
    pos = q * (len(xs) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (pos - lo) * (xs[hi] - xs[lo])


def iqr(xs):
    return quantile(xs, 0.75) - quantile(xs, 0.25)


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def stdev(xs):
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


# ---------------------------------------------------------------------------------------------
#  Non-parametric tests
# ---------------------------------------------------------------------------------------------


def _midranks(values):
    """Ranks 1..n of `values`, ties sharing their mean rank; also the sizes of the tie groups."""
    n = len(values)
    order = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    groups = []
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        r = 0.5 * ((i + 1) + (j + 1))
        for k in range(i, j + 1):
            ranks[order[k]] = r
        groups.append(j - i + 1)
        i = j + 1
    return ranks, groups


def wilcoxon_signed_rank(a, b):
    """Two-sided Wilcoxon signed-rank test for paired samples.

    Zero differences are dropped (Wilcoxon's convention).  Tied absolute differences receive their
    mean rank.  When at most 25 non-zero differences remain the p-value is EXACT, including under
    ties: the null distribution of W+ is the distribution of the sum of the actual (mid)ranks under
    independent random signs, enumerated by dynamic programming over twice the ranks (which are
    integers).  Beyond 25 a normal approximation with the tie-corrected variance and a continuity
    correction is used.  Returns (statistic, p), the statistic being min(W+, W-).
    """
    diffs = [x - y for x, y in zip(a, b) if x != y]
    n = len(diffs)
    if n == 0:
        return 0.0, 1.0
    ranks, groups = _midranks([abs(d) for d in diffs])
    wplus = sum(ranks[i] for i in range(n) if diffs[i] > 0)
    wminus = sum(ranks[i] for i in range(n) if diffs[i] < 0)
    w = min(wplus, wminus)

    if n <= 25:
        twice = [int(round(2 * r)) for r in ranks]
        counts = {0: 1}
        for r in twice:
            nxt = defaultdict(int)
            for s_, c in counts.items():
                nxt[s_] += c
                nxt[s_ + r] += c
            counts = nxt
        total = 2 ** n
        w2 = int(round(2 * w))
        le = sum(c for s_, c in counts.items() if s_ <= w2)
        return w, min(1.0, 2.0 * le / total)

    mu = n * (n + 1) / 4.0
    var = n * (n + 1) * (2 * n + 1) / 24.0 - sum(g ** 3 - g for g in groups) / 48.0
    if var <= 0:
        return w, 1.0
    z = (w - mu + 0.5) / math.sqrt(var)
    return w, min(1.0, 2.0 * normal_cdf(z))


def paired_superiority(a, b):
    """The paired effect size: the share of pairs in which a[i] > b[i], ties counted a half.

    Unlike the Vargha--Delaney statistic, which compares every value of one sample with every value
    of the other, this uses the pairing the test uses.  0.5 is no effect, 1.0 means a exceeded b on
    every pair."""
    pairs = list(zip(a, b))
    if not pairs:
        return float("nan")
    return sum(1.0 if x > y else 0.5 if x == y else 0.0 for x, y in pairs) / len(pairs)


def sign_test(a, b):
    """Two-sided exact sign test for paired samples."""
    pos = sum(1 for x, y in zip(a, b) if x > y)
    neg = sum(1 for x, y in zip(a, b) if x < y)
    n = pos + neg
    if n == 0:
        return 0, 1.0
    k = min(pos, neg)
    p = 0.0
    for i in range(k + 1):
        p += math.comb(n, i)
    p = min(1.0, 2.0 * p / (2 ** n))
    return k, p


def normal_cdf(z):
    return 0.5 * math.erfc(-z / math.sqrt(2.0))


def vargha_delaney(a, b):
    """A-hat_12: the probability that a random draw from `a` exceeds one from `b`, ties counted a
    half.  0.5 is no effect; the usual thresholds are 0.56 small, 0.64 medium, 0.71 large."""
    if not a or not b:
        return float("nan")
    greater = sum(1 for x in a for y in b if x > y)
    equal = sum(1 for x in a for y in b if x == y)
    return (greater + 0.5 * equal) / (len(a) * len(b))


def friedman(blocks):
    """Friedman test.  `blocks` is a list of lists: one list per block (instance), each holding one
    measurement per treatment, in a fixed treatment order.  Ties within a block share their mean
    rank and the statistic is divided by the standard tie correction
    1 - sum(t^3 - t) / (n k (k^2 - 1)).  Returns (chi2, df, p, mean_ranks)."""
    if not blocks:
        return 0.0, 0, 1.0, []
    k = len(blocks[0])
    n = len(blocks)
    rank_sums = [0.0] * k
    tie_sum = 0.0
    for row in blocks:
        ranks, groups = _midranks(row)
        tie_sum += sum(g ** 3 - g for g in groups)
        for i in range(k):
            rank_sums[i] += ranks[i]
    mean_ranks = [s_ / n for s_ in rank_sums]
    chi2 = 12.0 * n / (k * (k + 1)) * sum((r - (k + 1) / 2.0) ** 2 for r in mean_ranks)
    correction = 1.0 - tie_sum / (n * k * (k * k - 1))
    if correction > 0:
        chi2 /= correction
    df = k - 1
    return chi2, df, chi2_sf(chi2, df), mean_ranks


def chi2_sf(x, df):
    """Upper tail of the chi-squared distribution, by the regularised incomplete gamma function."""
    if x <= 0:
        return 1.0
    return gammaincc(df / 2.0, x / 2.0)


def gammaincc(a, x):
    """Regularised upper incomplete gamma Q(a, x), by series or continued fraction."""
    if x < a + 1.0:
        # series for P(a, x)
        ap, s, d = a, 1.0 / a, 1.0 / a
        for _ in range(500):
            ap += 1
            d *= x / ap
            s += d
            if abs(d) < abs(s) * 1e-14:
                break
        return 1.0 - s * math.exp(-x + a * math.log(x) - math.lgamma(a))
    # continued fraction for Q(a, x)
    tiny = 1e-300
    b, c, d = x + 1.0 - a, 1.0 / tiny, 1.0 / (x + 1.0 - a)
    h = d
    for i in range(1, 500):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-14:
            break
    return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


def holm(pairs):
    """Holm step-down correction.  `pairs` is a list of (label, p).  Returns the same list with
    adjusted p-values, in the original order."""
    ordered = sorted(range(len(pairs)), key=lambda i: pairs[i][1])
    m = len(pairs)
    adjusted = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(ordered):
        val = min(1.0, (m - rank) * pairs[idx][1])
        running = max(running, val)
        adjusted[idx] = running
    return [(pairs[i][0], adjusted[i]) for i in range(m)]


def bootstrap_ci(xs, stat=median, reps=2000, level=0.95, seed=12345):
    """A percentile bootstrap interval, with a deterministic generator so the number in the paper
    does not move between runs."""
    if not xs:
        return (float("nan"), float("nan"))
    state = seed & 0xFFFFFFFF
    n = len(xs)
    vals = []
    for _ in range(reps):
        sample = []
        for _ in range(n):
            state = (1103515245 * state + 12345) & 0x7FFFFFFF
            sample.append(xs[state % n])
        vals.append(stat(sample))
    vals.sort()
    lo = vals[int((1 - level) / 2 * reps)]
    hi = vals[min(reps - 1, int((1 + level) / 2 * reps))]
    return (lo, hi)


# ---------------------------------------------------------------------------------------------
#  Emitters
# ---------------------------------------------------------------------------------------------


def fmt(x, digits=2):
    if x != x:
        return "--"
    if abs(x) >= 1e6:
        return f"{x:.3g}"
    if float(x).is_integer() and abs(x) < 1e6:
        return f"{int(x)}"
    return f"{x:.{digits}f}"


def fmt_p(p):
    if p != p:
        return "--"
    if p < 0.001:
        return "$<$0.001"
    return f"{p:.3f}"


def write_table(path, caption, label, header, rows, align=None, note=None, small=False):
    """A booktabs table, written as a standalone file to be \\input into the paper."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    k = len(header)
    align = align or ("l" + "r" * (k - 1))
    with open(path, "w") as f:
        f.write("% generated -- do not edit\n")
        f.write("\\begin{table}[tbp]\n\\centering\n")
        f.write(f"\\caption{{{caption}}}\n\\label{{{label}}}\n")
        if small:
            f.write("\\small\n")
        f.write(f"\\begin{{tabular}}{{{align}}}\n\\toprule\n")
        f.write(" & ".join(header) + " \\\\\n\\midrule\n")
        for r in rows:
            f.write(" & ".join(str(c) for c in r) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
        if note:
            f.write(f"\\\\[2pt]\n\\begin{{minipage}}{{\\linewidth}}\\footnotesize {note}\\end{{minipage}}\n")
        f.write("\\end{table}\n")


def write_dat(path, columns, rows):
    """A whitespace-separated table for pgfplots, with a header line of column names."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write(" ".join(columns) + "\n")
        for r in rows:
            f.write(" ".join(str(c) for c in r) + "\n")


_DIGIT_WORDS = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four",
                "5": "Five", "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine"}


def macro_name(name):
    """A LaTeX control sequence may contain letters only, so a fact key built from an instance size
    or a thread count has to have its digits spelled out.  Doing it here, in one place, means a
    caller can key facts by whatever is natural and never produce a name that silently splits into
    a macro plus some stray digits in the middle of a sentence."""
    return "".join(_DIGIT_WORDS.get(ch, ch) for ch in name if ch.isalnum())


def reset_outputs(*dirs, keep=("macro-fallbacks.tex",)):
    """Delete the generated tables and figure data before regenerating them.

    Without this, a table or a figure whose results file has been removed -- because the experiment
    that produced it was found to be measuring the wrong thing and is being re-run -- survives on
    disk and keeps appearing in the paper, sourced from data that no longer exists.  Everything in
    these directories is generated except the placeholder file, so clearing them is safe and the
    paper falls back to "results pending" for whatever is not regenerated.
    """
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for name in os.listdir(d):
            if name in keep:
                continue
            path = os.path.join(d, name)
            if os.path.isfile(path):
                os.remove(path)


def complete(path, min_age_seconds=None):
    """Is this results file finished being written?

    An experiment appends to its CSV as it runs, so an analysis started while one is in flight would
    build a table from a design that is only half executed -- a table that looks finished and is not.
    Treat a file as complete only when it exists and has not been touched recently.  The threshold can
    be overridden with TT_MIN_AGE, and set to 0 while iterating on the analysis itself.
    """
    if not os.path.exists(path):
        return False
    if min_age_seconds is None:
        min_age_seconds = float(os.environ.get("TT_MIN_AGE", "120"))
    age = time.time() - os.path.getmtime(path)
    if age < min_age_seconds:
        print(f"  skipping {os.path.basename(path)}: written {age:.0f}s ago, still running?")
        return False
    return True


def write_macros(path, macros):
    """Numbers quoted in the prose, as LaTeX macros, so that no value is ever typed by hand."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write("% generated -- do not edit\n")
        for name, value in sorted(macros.items()):
            f.write(f"\\newcommand{{\\{macro_name(name)}}}{{{value}}}\n")
