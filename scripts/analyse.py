#!/usr/bin/env python3
"""Tables, figure data and inline numbers for the controlled method comparison.

Conventions used throughout (and stated in the article):

* The *reported objective* f is the validator's value of the objective; it is the endpoint of every
  test, every median and every ranking below.
* *Feasible* means accepted by the validator (no domain, availability or load-cap breach), no
  unplaced class, and H = 0.  It is reported in its own column and never folded into f.
* Tests are paired two-sided Wilcoxon signed-rank tests on f (exact, ties included), Holm-corrected
  within the family named beside them.  The effect size is the paired superiority share: the share of
  paired instances on which the method's f exceeded the reference method's, ties counted a half.
"""

import argparse
import collections
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ttstats as t  # noqa: E402

METHODS = ["ga", "ma", "ma-adaptive", "sls", "sa", "rrhc"]
POP = ("ga", "ma", "ma-adaptive")
TRJ = ("sls", "sa", "rrhc")
LABEL = {
    "ga": "genetic",
    "ma": "memetic",
    "ma-adaptive": "memetic, adaptive",
    "sls": "local search",
    "sa": "simulated annealing",
    "rrhc": "restart hill climbing",
}
SHORTLABEL = {"ga": "GA", "ma": "MA", "ma-adaptive": "MA-ad", "sls": "SLS", "sa": "SA",
              "rrhc": "RRHC"}


def f_of(r):
    return t.num(r, "objective")


def feasible(r):
    return (r.get("clean", "1") == "1" and t.num(r, "unplaced") == 0 and t.num(r, "hard") == 0)


def lexkey(r):
    """The internal lexicographic order, used only for the robustness check and for tuning."""
    return (0 if r.get("clean", "1") == "1" else 1, t.num(r, "unplaced"), t.num(r, "hard"), f_of(r))


def sign_lex(a, b):
    return (lexkey(a) > lexkey(b)) - (lexkey(a) < lexkey(b))


def kname(m):
    return m.replace("-", "")


def main():
    ap = argparse.ArgumentParser()
    # Defaults are resolved against this application's directory, not the working directory, so the
    # script does the same thing whether it is run from code/, code/scripts/ or anywhere else.
    app = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--results", default=os.path.join(app, "results"))
    ap.add_argument("--out", default=None,
                    help="where tables/ and figures/ are written; defaults to the article directory "
                         "above this one when it holds a main.tex, and otherwise to this directory")
    args = ap.parse_args()
    # The published records never change, so the "still being written?" guard of ttstats.complete()
    # is pointless for them -- and in a fresh clone every file is seconds old.  It stays on for any
    # other results directory, which may belong to a run still in progress.
    if (os.path.abspath(args.results) == os.path.join(app, "results")
            and "TT_MIN_AGE" not in os.environ):
        os.environ["TT_MIN_AGE"] = "0"
    if args.out is None:
        parent = os.path.dirname(app)
        args.out = parent if os.path.exists(os.path.join(parent, "main.tex")) else app
    R, TAB, FIG = args.results, os.path.join(args.out, "tables"), os.path.join(args.out, "figures")
    os.makedirs(TAB, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    t.reset_outputs(TAB, FIG, keep=())
    facts = {}
    bestParams = {}

    # --- G1: legality at twice the budget ---------------------------------------------------------
    path = os.path.join(R, "agreement.csv")
    if t.complete(path):
        rows = t.read(path)
        facts["agreementRuns"] = len(rows)
        facts["agreementClean"] = sum(1 for r in rows if r["clean"] == "1")
        facts["agreementMethods"] = len({r["method"] for r in rows})
        facts["agreementPerMethod"] = min(collections.Counter(r["method"] for r in rows).values())
        facts["agreementBreaches"] = sum(int(r["domain_breaches"]) + int(r["availability_breaches"])
                                         + int(r["cap_breaches"]) for r in rows)
        facts["agreementUnplaced"] = sum(1 for r in rows if t.num(r, "unplaced") > 0)

    # --- G2: tuning ---------------------------------------------------------------------------------
    path = os.path.join(R, "tuning.csv")
    if t.complete(path):
        rows = t.read(path)
        rank = {}
        params = collections.defaultdict(dict)
        for r in rows:
            m, ci = r["method"], int(r["config"])
            params[(m, ci)][r["param"]] = r["value"]
            rank[(m, ci)] = t.num(r, "mean_rank")
        best = {}
        for (m, ci), v in sorted(rank.items(), key=lambda kv: (kv[0][0], kv[0][1])):
            if m not in best or v < best[m][1] - 1e-12:
                best[m] = (ci, v)

        def show(v):
            x = float(v)
            if abs(x - round(x)) < 1e-9 and abs(x) < 1e7:
                return str(int(round(x)))
            return t.fmt(x, 4).rstrip("0").rstrip(".")

        SHORT = {"population": "pop", "tournament": "tour", "elite": "elite",
                 "crossover_rate": "xo", "mutation_rate": "mut", "ls_probability": "lsp",
                 "ls_candidates": "lsc", "ls_history": "lsh", "ls_engine_dlas": "lsdlas",
                 "lahc_length": "hist", "engine_dlas": "dlas", "stagnation_moves": "stag",
                 "kick_max": "kick", "chain_depth": "chain", "lns_max": "lns",
                 "bandit_decay": "decay", "sa_t0_factor": "T0", "sa_cooling": "cool",
                 "sa_reheat_after": "reheat", "restart_after": "restart"}
        trows = []
        for m in METHODS:
            if m not in best:
                continue
            ci, v = best[m]
            p = params[(m, ci)]
            cell = " ".join(f"{SHORT.get(k, k)}={show(x)}" for k, x in sorted(p.items()))
            trows.append([LABEL[m], len(p), ci, t.fmt(rank[(m, 0)], 1), t.fmt(v, 1),
                          "\\texttt{\\footnotesize " + cell + "}"])
            facts["tuneWinner" + kname(m)] = str(ci)
            bestParams[m] = {k: float(x) for k, x in p.items()}
        t.write_table(
            os.path.join(TAB, "tuning.tex"),
            "Selected configurations. \\emph{dims}: parameters sampled for the method; "
            "\\emph{config}: selected index (0 = defaults); \\emph{rank}: mean rank over the ten "
            "tuning instances, of the defaults and of the selection (13 candidates).",
            "tab:tuning",
            ["method", "dims", "config", "rank$_0$", "rank", "selected configuration"],
            trows, align="lrrrrp{0.47\\textwidth}", small=True)
        percfg = {m: len({r["config"] for r in rows if r["method"] == m})
                  for m in {r["method"] for r in rows}}
        facts["tuningConfigs"] = min(percfg.values())
        facts["tuningDrawn"] = min(percfg.values()) - 1
        facts["tuningConfigsTotal"] = sum(percfg.values())
        facts["tuningSeeds"] = str(max(int(r["runs"]) for r in rows) // 2)
        dims = {m: len(params[(m, best[m][0])]) for m in best}
        facts["tuningDimsMin"] = min(dims.values())
        facts["tuningDimsMax"] = max(dims.values())
        facts["tuningAtReference"] = str(sum(1 for m in best if best[m][0] == 0))
        atref = [LABEL[m] for m in METHODS if m in best and best[m][0] == 0]
        facts["tuningAtReferenceNames"] = ", ".join(atref) if atref else "none"
        tr = os.path.join(R, "tuning_runs.csv")
        if t.complete(tr):
            facts["tuningRuns"] = len(t.read(tr))

    # --- G3: the main comparison --------------------------------------------------------------------
    path = os.path.join(R, "compare.csv")
    if t.complete(path):
        rows = t.read(path)
        by = collections.defaultdict(list)
        for r in rows:
            by[(int(r["n"]), r["method"])].append(r)
        sizes = sorted({k[0] for k in by})
        crows, dat = [], collections.defaultdict(list)
        lexDisagree = 0
        lexCells = 0
        iters = {}
        orders = []
        for n in sizes:
            base = {r["seed"]: r for r in by.get((n, "sls"), [])}
            tests, effect = [], {}
            for m in METHODS:
                v = by.get((n, m))
                if not v or m == "sls":
                    continue
                b = {r["seed"]: r for r in v}
                ks = sorted(set(base) & set(b), key=int)
                _, p = t.wilcoxon_signed_rank([f_of(b[k]) for k in ks], [f_of(base[k]) for k in ks])
                tests.append((m, p))
                effect[m] = t.paired_superiority([f_of(b[k]) for k in ks],
                                                 [f_of(base[k]) for k in ks])
                # Robustness: does the internal lexicographic order agree in sign on every pair
                # with the f-based order?
                lexCells += 1
                if any((sign_lex(b[k], base[k]) > 0) != (f_of(b[k]) > f_of(base[k])) for k in ks):
                    lexDisagree += 1
            adj = dict(t.holm(tests))
            for m in METHODS:
                v = by.get((n, m))
                if not v:
                    continue
                obj = [f_of(r) for r in v]
                hard = [t.num(r, "hard") for r in v]
                soft = [t.num(r, "soft") for r in v]
                feas = sum(1 for r in v if feasible(r))
                gens = [t.num(r, "generations") for r in v]
                offs = [t.num(r, "evaluations") for r in v]
                crows.append([n, LABEL[m], f"{feas}/{len(v)}",
                              t.fmt(t.median(gens), 0) if m in POP else "--",
                              t.fmt(t.median(offs), 0) if m in POP else "--",
                              t.fmt(t.median(hard), 1), t.fmt(t.median(soft), 0),
                              t.fmt(t.median(obj), 0), t.fmt(t.iqr(obj), 0),
                              t.fmt_p(adj[m]) if m in adj else "--",
                              t.fmt(effect[m], 2) if m in effect else "--"])
                if m in POP:
                    facts[f"gens{kname(m)}{n}"] = t.fmt(t.median(gens), 0)
                    facts[f"offRange{kname(m)}{n}"] = f"{int(min(offs))}--{int(max(offs))}"
                    bp = bestParams.get(m, {})
                    perGen = int(bp.get("population", 30)) - int(bp.get("elite", 2))
                    facts[f"perGen{kname(m)}"] = str(perGen)
                    facts[f"noFullGen{kname(m)}{n}"] = str(sum(1 for x in offs if x < perGen))
                    facts[f"fullGen{kname(m)}{n}"] = str(sum(1 for x in offs if x >= perGen))
                    facts[f"singleIter{kname(m)}{n}"] = str(sum(1 for x in gens if x <= 1))
                    iters[m] = iters.get(m, []) + gens
                    facts[f"fromZero{kname(m)}{n}"] = str(sum(1 for r in v
                                                            if t.num(r, "best_generation") == 0))
                if m in adj:
                    facts[f"cmpP{n}{kname(m)}"] = t.fmt_p(adj[m])
                    facts[f"cmpPS{n}{kname(m)}"] = t.fmt(effect[m], 2)
                dat[m].append([n, round(t.median(obj), 1), round(t.quantile(obj, 0.25), 1),
                               round(t.quantile(obj, 0.75), 1)])
            medsN = {m: t.median([f_of(r) for r in by[(n, m)]]) for m in METHODS if (n, m) in by}
            orders.append(tuple(sorted(medsN, key=lambda m: medsN[m])))
            facts[f"cmpRejected{n}"] = str(sum(1 for _, p in adj.items() if p < 0.05))
            facts[f"cmpPSmin{n}"] = t.fmt(min(effect.values()), 2)
        for m, g in iters.items():
            facts[f"iterMin{kname(m)}"] = str(int(min(g)))
            facts[f"iterMax{kname(m)}"] = str(int(max(g)))
        facts["cmpOrderSame"] = "yes" if len(set(orders)) == 1 else "no"
        facts["cmpOrder"] = "; ".join(LABEL[m] for m in orders[0])
        facts["lexDisagreeCells"] = str(lexDisagree)
        facts["lexCells"] = str(lexCells)
        allRej = sum(int(facts[f"cmpRejected{n}"]) for n in sizes)
        facts["cmpRejectedAll"] = str(allRej)
        facts["cmpTestsAll"] = str(5 * len(sizes))
        facts["cmpPSminAll"] = t.fmt(min(float(facts[f"cmpPSmin{n}"]) for n in sizes), 2)
        t.write_table(
            os.path.join(TAB, "compare.tex"),
            "The main comparison, $W=6000n$ bucket recomputations, ten instances per cell. "
            "\\emph{feasible}: legal, complete and $H=0$; \\emph{iter.}: median generation "
            "iterations, a final partial one included; \\emph{off.}: median offspring; $H$, "
            "\\emph{soft}, $f$: medians; IQR of $f$; $p$: Wilcoxon signed-rank on $f$ against the "
            "local search, Holm-corrected over the five comparisons of the size; $\\hat P$: share "
            "of instances on which the method's $f$ exceeded the local search's, ties counted a half.",
            "tab:compare",
            ["$n$", "method", "feasible", "iter.", "off.", "$H$", "soft", "$f$", "IQR", "$p$",
             "$\\hat P$"],
            crows, align="rlrrrrrrrrr", small=True)
        for m, v in dat.items():
            t.write_dat(os.path.join(FIG, f"compare-{m}.dat"), ["n", "objective", "q25", "q75"], v)

        # Feasibility over all sizes.
        for m in METHODS:
            allv = [r for k, v in by.items() if k[1] == m for r in v]
            if allv:
                facts["feas" + kname(m)] = str(sum(1 for r in allv if feasible(r)))
                facts["runs" + kname(m)] = str(len(allv))
                infeasSizes = sorted({int(r["n"]) for r in allv if not feasible(r)})
                facts["infeasSizes" + kname(m)] = (", ".join(str(x) for x in infeasSizes)
                                                    if infeasSizes else "none")
                feasSizes = sorted({int(r["n"]) for r in allv if feasible(r)})
                facts["feasSizes" + kname(m)] = (", ".join(str(x) for x in feasSizes)
                                                  if feasSizes else "none")
        facts["cmpUnclean"] = str(sum(1 for r in rows if r.get("clean") != "1"))
        facts["cmpAgree"] = str(sum(1 for r in rows if r.get("evaluator_agrees") == "1"))
        facts["cmpUnplaced"] = str(sum(1 for r in rows if t.num(r, "unplaced") > 0))
        facts["cmpRunsTotal"] = str(len(rows))
        seeds = max(len(v) for v in by.values())
        facts["cmpSeeds"] = str(seeds)
        facts["cmpFloorP"] = t.fmt_p(min(1.0, (2.0 / (2 ** seeds)) * 5))

        # The share of f that the hard terms account for in infeasible rows: an upper bound
        # sum_i beta_i Pi_i^2 <= 150 H^2.
        hardShare = [150.0 * t.num(r, "hard") ** 2 / f_of(r) for r in rows
                     if t.num(r, "hard") > 0 and f_of(r) > 0]
        if hardShare:
            facts["hardShareMax"] = t.fmt(100 * max(hardShare), 1)
            facts["hardShareMedian"] = t.fmt(100 * t.median(hardShare), 2)

        # Budget accounting: actual work over nominal.
        over = {}
        for m in METHODS:
            v = [t.num(r, "work") / t.num(r, "budget") - 1 for r in rows
                 if r["method"] == m and t.num(r, "budget") > 0]
            if v:
                over[m] = max(v)
                facts["overshoot" + kname(m)] = t.fmt(100 * max(v), 2)
        if over:
            facts["overshootMax"] = t.fmt(100 * max(over.values()), 2)
            facts["overshootMaxMethod"] = LABEL[max(over, key=over.get)]
            under = min(t.num(r, "work") / t.num(r, "budget") for r in rows
                        if t.num(r, "budget") > 0)
            facts["budgetSpentMin"] = t.fmt(100 * under, 1)
        # Realised refinement: offspring scored versus offspring refined, over all main runs.
        for m in ("ma", "ma-adaptive"):
            v = [r for r in rows if r["method"] == m]
            if v:
                facts["childrenTotal" + kname(m)] = str(int(sum(t.num(r, "evaluations") for r in v)))
                facts["refinedTotal" + kname(m)] = str(int(sum(t.num(r, "refined_children")
                                                               for r in v)))
        # Actual initial-refinement work over the budget (nominal allocation one fifth).
        for m in ("ma", "ma-adaptive"):
            v = [t.num(r, "initial_refine_work") / t.num(r, "budget") for r in rows
                 if r["method"] == m and t.num(r, "budget") > 0]
            if v:
                facts["initShareMax" + kname(m)] = t.fmt(100 * max(v), 2)
        # Construction share of a population method's budget.
        for m in POP:
            v = [t.num(r, "construction_work") / t.num(r, "work") for r in rows
                 if r["method"] == m and t.num(r, "work") > 0]
            if v:
                facts["constrShare" + kname(m)] = t.fmt(100 * max(v), 1)

        # Friedman across methods, and all-pairs Holm post-hoc within each size.
        crossClose = []
        for n in sizes:
            present = [m for m in METHODS if (n, m) in by]
            seedsN = set.intersection(*[{r["seed"] for r in by[(n, m)]} for m in present])
            get = {(m, r["seed"]): r for m in present for r in by[(n, m)]}
            blocks = [[f_of(get[(m, s)]) for m in present] for s in sorted(seedsN, key=int)]
            chi2, df, p, ranks = t.friedman(blocks)
            facts[f"fried{n}Chi"] = t.fmt(chi2, 1)
            facts[f"fried{n}P"] = t.fmt_p(p)
            for m, rk in zip(present, ranks):
                facts["rank" + kname(m).capitalize() + str(n)] = t.fmt(rk, 1)
            pairs = []
            for i, a in enumerate(present):
                for b in present[i + 1:]:
                    _, pp = t.wilcoxon_signed_rank([f_of(get[(a, s)]) for s in seedsN],
                                                   [f_of(get[(b, s)]) for s in seedsN])
                    pairs.append(((a, b), pp))
            adj = t.holm(pairs)
            notsep = [(a, b) for (a, b), pp in adj if pp >= 0.05]
            facts[f"postSep{n}"] = str(len(adj) - len(notsep))
            facts[f"postPairs{n}"] = str(len(adj))
            facts[f"postNotSep{n}"] = ("none" if not notsep else
                                      "; ".join(f"{LABEL[a]} and {LABEL[b]}" for a, b in notsep))
            # Cross-family: is every population method worse than every trajectory method on the
            # median, and which cross pair is closest?
            meds = {m: t.median([f_of(r) for r in by[(n, m)]]) for m in present}
            for a in present:
                for b in present:
                    if a in POP and b in TRJ:
                        crossClose.append((meds[a] / max(1.0, meds[b]), n, a, b))
            facts[f"popBeatsTrj{n}"] = str(sum(1 for a in present for b in present
                                               if a in POP and b in TRJ and meds[a] < meds[b]))
            s = meds.get("sls")
            if s:
                facts[f"popGap{n}"] = t.fmt(min(meds[m] for m in POP if m in meds) / s, 0)
                if "sa" in meds:
                    facts[f"saGap{n}"] = t.fmt(meds["sa"] / s, 1)
            if (n, "sa") in by and (n, "rrhc") in by:
                for (a, b), pp in adj:
                    if {a, b} == {"sa", "rrhc"}:
                        facts[f"saRrhcP{n}"] = t.fmt_p(pp)
        if crossClose:
            g, n0, a, b = min(crossClose)
            facts["crossCloseRatio"] = t.fmt(g, 2)
            facts["crossCloseN"] = str(n0)
            facts["crossClosePop"] = LABEL[a]
            facts["crossCloseTrj"] = LABEL[b]
            facts["popBeatsTrjAll"] = str(sum(int(facts[f"popBeatsTrj{n}"]) for n in sizes))

    # --- G4: two budget currencies --------------------------------------------------------------------
    def currency_table(rows, fname, label, caption):
        by = collections.defaultdict(list)
        for r in rows:
            by[(int(r["n"]), r["currency"], r["method"])].append(r)
        brows, order = [], {}
        for n in sorted({k[0] for k in by}):
            for cur in ("work", "evaluations"):
                meds = {m: t.median([f_of(r) for r in by[(n, cur, m)]])
                        for m in METHODS if by.get((n, cur, m))}
                # Competition ranks: tied medians share the smaller rank.
                rk = {m: 1 + sum(1 for x in meds if meds[x] < meds[m]) for m in meds}
                order[(n, cur)] = tuple(rk.get(m) for m in METHODS)
                for m in METHODS:
                    v = by.get((n, cur, m))
                    if not v:
                        continue
                    brows.append([n, cur, LABEL[m], rk[m],
                                  f"{sum(1 for r in v if feasible(r))}/{len(v)}",
                                  t.fmt(meds[m], 0),
                                  t.fmt(t.median([t.num(r, "work") for r in v]) / 1e6, 2),
                                  t.fmt(t.median([t.num(r, "evaluations") for r in v]) / 1e3, 1),
                                  t.fmt(t.median([t.num(r, "elapsed_ms") for r in v]) / 1000.0, 1)])
        if fname:
            t.write_table(os.path.join(TAB, fname), caption, label,
                          ["$n$", "currency", "method", "rank", "feasible", "$f$", "work (M)",
                           "evals (k)", "s"], brows, align="rlrrrrrrr", small=True)
        return by, order

    path = os.path.join(R, "budgetmode.csv")
    if t.complete(path):
        rows = t.read(path)
        by, order = currency_table(
            rows, "budgetmode.tex", "tab:budgetmode",
            "Reference configurations under two budget currencies, ten instances per cell. One "
            "\\emph{evaluation} is one offspring or one trajectory candidate; the evaluation budget "
            "is the number of candidates the local search used under the work budget. "
            "\\emph{rank} orders the medians of $f$ within the cell; work, evaluations and seconds "
            "are medians.")
        sizes = sorted({int(r["n"]) for r in rows})
        same = []
        for n in sizes:
            tag = "Small" if n == sizes[0] else "Large"
            facts["subsidyN" + tag] = str(n)
            ref = t.median([t.num(r, "work") for r in by[(n, "evaluations", "sls")]])
            for m in ("ga", "ma", "ma-adaptive"):
                v = by.get((n, "evaluations", m))
                if v and ref:
                    facts["subsidy" + kname(m) + tag] = t.fmt(
                        t.median([t.num(r, "work") for r in v]) / ref, 0)
            s = order[(n, "work")] == order[(n, "evaluations")]
            same.append(s)
            facts["orderSame" + tag] = "identical" if s else "different"
            # completeness of both currencies
            done = all(t.num(r, "evaluations") >= t.num(r, "budget")
                       for r in rows if r["currency"] == "evaluations" and int(r["n"]) == n)
            facts["evalBudgetSpent" + tag] = "yes" if done else "no"
            # how much did f move between currencies, per population method
            for m in POP:
                a = t.median([f_of(r) for r in by[(n, "work", m)]])
                b = t.median([f_of(r) for r in by[(n, "evaluations", m)]])
                facts["currencyGain" + kname(m) + tag] = t.fmt(100 * (1 - b / a), 1)
            # The closest cross-family pair under the evaluation budget (population methods with
            # their subsidy against the trajectory methods).
            mE = {m: t.median([f_of(r) for r in by[(n, "evaluations", m)]]) for m in METHODS}
            g = min((mE[p_] / mE[q_], p_, q_) for p_ in POP for q_ in TRJ)
            facts["bmClose" + tag] = t.fmt(g[0], 1)
            facts["bmClosePop" + tag] = LABEL[g[1]]
            facts["bmCloseTrj" + tag] = LABEL[g[2]]
        facts["orderSameAll"] = "yes" if all(same) else "no"
        gains = []
        for n in sizes:
            for m in POP:
                a_ = t.median([f_of(r) for r in by[(n, "work", m)]])
                b_ = t.median([f_of(r) for r in by[(n, "evaluations", m)]])
                gains.append(abs(100 * (1 - b_ / a_)))
        facts["currencyGainMaxAbs"] = t.fmt(max(gains), 1)
        facts["budgetModeSizes"] = " and ".join(str(x) for x in sizes)
        facts["budgetModeWorstWork"] = t.fmt(max(t.num(r, "work") for r in rows) / 1e6, 0)

    tpath = os.path.join(R, "budgetmode_tuned.csv")
    if t.complete(tpath):
        trows = t.read(tpath)
        rpath = os.path.join(R, "budgetmode.csv")
        rrows = t.read(rpath) if t.complete(rpath) else []
        tsizes = sorted({int(r["n"]) for r in trows})
        facts["subsidyTunedSeeds"] = str(len({r["seed"] for r in trows}))
        facts["subsidyTunedSizes"] = " and ".join(str(x) for x in tsizes)

        def subsidy(rs, n, m, seeds=None):
            ev = [r for r in rs if int(r["n"]) == n and r["currency"] == "evaluations"
                  and (seeds is None or r["seed"] in seeds)]
            v = [t.num(r, "work") for r in ev if r["method"] == m]
            ref = [t.num(r, "work") for r in ev if r["method"] == "sls"]
            done = all(t.num(r, "evaluations") >= t.num(r, "budget") for r in ev)
            if not v or not ref:
                return None, False
            return t.median(v) / t.median(ref), done

        srows, truncated = [], 0
        tseeds = {r["seed"] for r in trows}
        for n in tsizes:
            for m in POP:
                a, _ = subsidy(rrows, n, m) if rrows else (None, False)
                a3, _ = subsidy(rrows, n, m, tseeds) if rrows else (None, False)
                b, bDone = subsidy(trows, n, m)
                if b is None:
                    continue
                truncated += 0 if bDone else 1
                srows.append([n, LABEL[m], "--" if a is None else t.fmt(a, 0),
                              "--" if a3 is None else t.fmt(a3, 0),
                              t.fmt(b, 0) + ("" if bDone else "$^{\\ast}$")])
                if bDone:
                    facts["subsidyTuned" + kname(m)] = t.fmt(b, 0)
                    if a3:
                        facts["subsidyTunedOver" + kname(m)] = t.fmt(b / a3, 1)
        facts["subsidyTunedTruncated"] = str(truncated)
        if srows:
            t.write_table(
                os.path.join(TAB, "subsidytuned.tex"),
                "Work spent under the evaluation budget, as a multiple of the local search's: "
                "reference configurations (ten instances, and the same %s instances as the tuned "
                "runs) and tuned configurations." % facts["subsidyTunedSeeds"]
                + ("" if not truncated else
                   " $^{\\ast}$: runs stopped by the time limit; a lower bound."),
                "tab:subsidytuned",
                ["$n$", "method", "ref.", "ref. (same)", "tuned"],
                srows, align="rlrrr")
        # tuned order under both currencies, as a check of F1 at the tuned configurations
        byT, orderT = currency_table(trows, None, None, None)
        for n in tsizes:
            facts["orderSameTuned"] = ("identical" if orderT[(n, "work")] ==
                                       orderT[(n, "evaluations")] else "different")


    # --- G5: the cost of one evaluation ------------------------------------------------------------
    def evalcost(rows, tag):
        by = collections.defaultdict(list)
        for r in rows:
            by[(int(r["n"]), r["method"])].append(r)
        ns = sorted({k[0] for k in by})
        out = {}
        for m in METHODS:
            for n in ns:
                v = by.get((n, m))
                if not v:
                    continue
                out[(m, "total", n)] = t.median([t.num(r, "work_per_evaluation") for r in v])
                if m in POP:
                    out[(m, "score", n)] = t.median([t.num(r, "child_scoring_per_offspring")
                                                     for r in v])
                    out[(m, "h", n)] = t.median([t.num(r, "mean_changed_genes") for r in v])
                    out[(m, "par", n)] = t.median([t.num(r, "mean_parent_distance") for r in v])
        return by, ns, out

    path = os.path.join(R, "evalcost.csv")
    if t.complete(path):
        rows = t.read(path)
        by, ns, ec = evalcost(rows, "ref")
        erows = []
        for m in METHODS:
            if m in POP:
                erows.append([LABEL[m] + ", scoring"] + [t.fmt(ec.get((m, "score", n), float("nan")), 0)
                                                        for n in ns])
                erows.append([LABEL[m] + ", total"] + [t.fmt(ec.get((m, "total", n), float("nan")), 0)
                                                      for n in ns])
            else:
                erows.append([LABEL[m]] + [t.fmt(ec.get((m, "total", n), float("nan")), 1)
                                           for n in ns])
            out = [[n, round(ec.get((m, "score" if m in POP else "total", n), float("nan")), 2)]
                   for n in ns]
            t.write_dat(os.path.join(FIG, f"evalcost-{m}.dat"), ["n", "work_per_evaluation"], out)
        t.write_table(
            os.path.join(TAB, "evalcost.tex"),
            "Work per evaluation in bucket recomputations, reference configurations, $W=3000n$, "
            "medians of six instances. \\emph{scoring}: scoring one child; \\emph{total}: all work "
            "(construction, scoring, refinement) per offspring. Trajectory rows: all work per "
            "candidate.",
            "tab:evalcost",
            ["method"] + [f"$n={n}$" for n in ns], erows, align="l" + "r" * len(ns), small=True)
        n0, n1 = ns[0], ns[-1]
        facts["evalCostNmin"], facts["evalCostNmax"] = n0, n1
        facts["evalCostSeeds"] = str(max(len(v) for v in by.values()))
        facts["gaCostSmall"] = t.fmt(ec[("ga", "score", n0)], 0)
        facts["gaCostLarge"] = t.fmt(ec[("ga", "score", n1)], 0)
        facts["slsCostSmall"] = t.fmt(ec[("sls", "total", n0)], 1)
        facts["slsCostLarge"] = t.fmt(ec[("sls", "total", n1)], 1)
        sl = [ec[("sls", "total", n)] for n in ns]
        facts["slsCostMin"] = t.fmt(min(sl), 0)
        facts["slsCostMax"] = t.fmt(max(sl), 0)
        facts["costRatioSmall"] = t.fmt(ec[("ga", "score", n0)] / ec[("sls", "total", n0)], 0)
        facts["costRatioLarge"] = t.fmt(ec[("ga", "score", n1)] / ec[("sls", "total", n1)], 0)
        # Fraction of genes a child differs in, relative to n, and log-log slope of scoring cost.
        for m in POP:
            if any((m, "score", n) not in ec for n in ns):
                continue
            xs = [math.log(n) for n in ns]
            ys = [math.log(ec[(m, "score", n)]) for n in ns]
            mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
            slope = (sum((x - mx) * (y - my) for x, y in zip(xs, ys))
                     / sum((x - mx) ** 2 for x in xs))
            facts["costSlope" + kname(m)] = t.fmt(slope, 2)
            facts["hFrac" + kname(m) + "Small"] = t.fmt(ec[(m, "h", n0)] / n0, 2)
            facts["hFrac" + kname(m) + "Large"] = t.fmt(ec[(m, "h", n1)] / n1, 2)
            facts["parFrac" + kname(m) + "Large"] = t.fmt(ec[(m, "par", n1)] / n1, 2)
            facts["perGene" + kname(m) + "Large"] = t.fmt(ec[(m, "score", n1)] / ec[(m, "h", n1)], 1)
        xs = [math.log(n) for n in ns]
        ys = [math.log(ec[("sls", "total", n)]) for n in ns]
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        facts["costSlopesls"] = t.fmt(sum((x - mx) * (y - my) for x, y in zip(xs, ys))
                                      / sum((x - mx) ** 2 for x in xs), 2)
        # The same projection at the reference configuration (rho = 0.3, c_LS = 2000), which the
        # completed equal-evaluation runs of the currency experiment can check.
        for n in ns:
            sls = ec.get(("sls", "total", n))
            v = by.get((n, "ma"))
            if v and sls:
                wc = t.median([(t.num(r, "initial_refine_work") + t.num(r, "child_refine_work"))
                               / max(1.0, t.num(r, "refine_candidates")) for r in v])
                marg = ec[("ma", "score", n)] + 0.3 * 2000 * wc
                facts[f"projRefRatioma{n}"] = t.fmt(marg / sls, 0)
        for n in (400, 1600):
            if ("ma", "total", n) in ec:
                facts[f"maTotalPerOff{n}"] = t.fmt(ec[("ma", "total", n)], 0)
                facts[f"maScorePerOff{n}"] = t.fmt(ec[("ma", "score", n)], 0)

    path = os.path.join(R, "evalcost_tuned.csv")
    if t.complete(path):
        rows = t.read(path)
        by, ns, ec = evalcost(rows, "tuned")
        big = ns[-1]
        facts["tunedCostN"] = big
        facts["tunedCostga"] = t.fmt(ec[("ga", "score", big)], 0)
        facts["tunedCostsls"] = t.fmt(ec[("sls", "total", big)], 1)
        facts["tunedRatioga"] = t.fmt(ec[("ga", "score", big)] / ec[("sls", "total", big)], 0)
        facts["tunedCostmaTotal"] = t.fmt(ec[("ma", "total", big)], 0)
        facts["tunedCostmaScore"] = t.fmt(ec[("ma", "score", big)], 0)
        facts["tunedHFracga"] = t.fmt(ec[("ga", "h", big)] / big, 2)
        facts["tunedParFracga"] = t.fmt(ec[("ga", "par", big)] / big, 2)
        xs = [math.log(n) for n in ns]
        ys = [math.log(ec[("ga", "score", n)]) for n in ns]
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        facts["tunedCostSlopega"] = t.fmt(sum((x - mx) * (y - my) for x, y in zip(xs, ys))
                                          / sum((x - mx) ** 2 for x in xs), 2)
        # Projected marginal work of one more evaluation at the tuned memetic configurations:
        # scoring one child plus rho refinements of c_LS candidates at the measured work per
        # refinement candidate.  A projection from measured per-unit costs, not a completed
        # equal-evaluation run (which at these configurations would take hours per run).
        prow = []
        for n in ns:
            sls = ec.get(("sls", "total", n))
            for m in ("ma", "ma-adaptive"):
                v = by.get((n, m))
                if not v or m not in bestParams or not sls:
                    continue
                rho = bestParams[m].get("ls_probability", 0.3)
                cls = bestParams[m].get("ls_candidates", 2000)
                wc = t.median([(t.num(r, "initial_refine_work") + t.num(r, "child_refine_work"))
                               / max(1.0, t.num(r, "refine_candidates")) for r in v])
                marg = ec[(m, "score", n)] + rho * cls * wc
                facts[f"projWork{kname(m)}{n}"] = t.fmt(marg, 0)
                facts[f"projRatio{kname(m)}{n}"] = t.fmt(marg / sls, 0)
                facts[f"projOrder{kname(m)}{n}"] = str(int(round(math.log10(marg / sls))))
                facts[f"projWcand{kname(m)}{n}"] = t.fmt(wc, 1)
                facts[f"projRho{kname(m)}"] = t.fmt(rho, 2)
                facts[f"projCls{kname(m)}"] = t.fmt(cls, 0)

    # --- the clock: ms per million units of work, from the main comparison -----------------------
    path = os.path.join(R, "compare.csv")
    if t.complete(path):
        rows = t.read(path)
        by = collections.defaultdict(list)
        for r in rows:
            w = t.num(r, "work")
            if w > 0:
                by[(int(r["n"]), r["method"])].append(t.num(r, "elapsed_ms") / (w / 1e6))
        ns = sorted({k[0] for k in by})
        wrows, spread = [], {}
        for m in METHODS:
            wrows.append([LABEL[m]] + [t.fmt(t.median(by[(n, m)]), 0) if (n, m) in by else "--"
                                       for n in ns])
        for n in ns:
            vals = [t.median(by[(n, m)]) for m in METHODS if (n, m) in by]
            spread[n] = max(vals) / min(vals)
        wrows.append(["spread"] + [t.fmt(spread[n], 1) + "$\\times$" for n in ns])
        last = ns[-1]
        facts["clockSpreadMax"] = t.fmt(max(spread.values()), 1)
        facts["clockSpreadMaxUa"] = facts["clockSpreadMax"].replace(".", "{,}")
        facts["clockSpreadLargeN"] = t.fmt(spread[last], 1)
        facts["clockSlsOverGa"] = t.fmt(t.median(by[(last, "sls")]) / t.median(by[(last, "ga")]), 1)
        cheapest = min(METHODS, key=lambda m: t.median(by[(last, m)]))
        dearest = max(METHODS, key=lambda m: t.median(by[(last, m)]))
        facts["clockCheapest"] = LABEL[cheapest]
        facts["clockDearest"] = LABEL[dearest]
        t.write_table(
            os.path.join(TAB, "clock.tex"),
            "Milliseconds of single-threaded elapsed time per million bucket recomputations in the "
            "main comparison (medians of ten runs, whole run including construction). "
            "\\emph{spread}: largest over smallest entry of the column.",
            "tab:clock", ["method"] + [f"$n={n}$" for n in ns],
            wrows, align="l" + "r" * len(ns), small=True)

    # --- G6: budget response -----------------------------------------------------------------------
    path = os.path.join(R, "converge.csv")
    if t.complete(path):
        rows = t.read(path)
        by = collections.defaultdict(lambda: collections.defaultdict(list))
        for r in rows:
            by[r["method"]][int(r["work_budget"])].append(r)
        for m in by:
            out = []
            for b in sorted(by[m]):
                v = [f_of(r) for r in by[m][b]]
                out.append([b, round(t.median(v), 1), round(t.quantile(v, 0.25), 1),
                            round(t.quantile(v, 0.75), 1),
                            sum(1 for r in by[m][b] if feasible(r))])
            t.write_dat(os.path.join(FIG, f"converge-{m}.dat"),
                        ["work", "median", "q25", "q75", "feasible"], out)
        budgets = sorted({int(r["work_budget"]) for r in rows})
        facts["convLadderSpan"] = t.fmt(budgets[-1] / budgets[0], 0)
        facts["convN"] = str(int(rows[0]["n"]))
        facts["convSeeds"] = str(len({r["seed"] for r in rows}))
        for m in sorted(by):
            vals = collections.defaultdict(set)
            for r in rows:
                if r["method"] == m:
                    vals[r["seed"]].add(round(f_of(r), 6))
            if all(len(v) == 1 for v in vals.values()):
                facts["convFlat" + kname(m).capitalize()] = "yes"
            lo, hi = budgets[0], budgets[-1]
            a = t.median([f_of(r) for r in by[m][lo]])
            b = t.median([f_of(r) for r in by[m][hi]])
            facts["convGain" + kname(m)] = t.fmt(100 * (1 - b / a), 1)

    # --- G8: equal wall-clock time, measured ---------------------------------------------------------
    path = os.path.join(R, "timed.csv")
    cpath = os.path.join(R, "compare.csv")
    if t.complete(path):
        rows = t.read(path)
        crows_ = t.read(cpath) if t.complete(cpath) else []
        by = collections.defaultdict(list)
        for r in rows:
            by[(int(r["n"]), r["method"])].append(r)
        ns = sorted({k[0] for k in by})
        trows, same = [], []
        for n in ns:
            meds = {m: t.median([f_of(r) for r in by[(n, m)]]) for m in METHODS if (n, m) in by}
            ranked = sorted(meds, key=lambda m: meds[m])
            cm = {m: t.median([f_of(r) for r in crows_ if int(r["n"]) == n and r["method"] == m])
                  for m in METHODS}
            cranked = sorted(cm, key=lambda m: cm[m]) if crows_ else []
            same.append(ranked == cranked)
            base = {r["seed"]: r for r in by[(n, "sls")]}
            tests = []
            for m in METHODS:
                if m == "sls" or (n, m) not in by:
                    continue
                b = {r["seed"]: r for r in by[(n, m)]}
                ks = sorted(set(base) & set(b), key=int)
                tests.append((m, t.wilcoxon_signed_rank([f_of(b[k]) for k in ks],
                                                        [f_of(base[k]) for k in ks])[1]))
            adj = dict(t.holm(tests))
            for m in ranked:
                v = by[(n, m)]
                trows.append([n, LABEL[m], ranked.index(m) + 1,
                              cranked.index(m) + 1 if cranked else "--",
                              f"{sum(1 for r in v if feasible(r))}/{len(v)}",
                              t.fmt(meds[m], 0),
                              t.fmt(t.median([t.num(r, "work") for r in v]) / 1e6, 2),
                              t.fmt(t.median([t.num(r, "elapsed_ms") for r in v]) / 1000, 1),
                              t.fmt_p(adj[m]) if m in adj else "--"])
            facts[f"timedOrder{n}"] = "; ".join(f"{i + 1}.~{LABEL[m]}"
                                               for i, m in enumerate(ranked))
            facts[f"timedSame{n}"] = "identical" if ranked == cranked else "different"
            facts[f"timedLimit{n}"] = t.fmt(t.median([t.num(r, "time_limit_ms") for r in v])
                                            / 1000, 1)
            facts[f"timedRejected{n}"] = str(sum(1 for p in adj.values() if p < 0.05))
            if "sa" in meds and meds["sls"] > 0:
                facts[f"timedSaOverSls{n}"] = t.fmt(meds["sa"] / meds["sls"], 1)
            wsls = t.median([t.num(r, "work") for r in by[(n, "sls")]])
            for m in METHODS:
                if (n, m) in by and m != "sls":
                    facts[f"timedWork{kname(m)}{n}"] = t.fmt(
                        t.median([t.num(r, "work") for r in by[(n, m)]]) / wsls, 1)
            facts[f"timedRejected{n}"] = str(sum(1 for p in adj.values() if p < 0.05))
        facts["timedSizes"] = ", ".join(str(n) for n in ns)
        facts["timedSameAll"] = "yes" if all(same) else "no"
        facts["timedSeeds"] = str(max(len(v) for v in by.values()))
        over = [t.num(r, "elapsed_ms") / t.num(r, "time_limit_ms") - 1 for r in rows
                if t.num(r, "time_limit_ms") > 0]
        facts["timedOverMax"] = t.fmt(100 * max(over), 0)
        t.write_table(
            os.path.join(TAB, "equaltime.tex"),
            "Equal wall-clock time, measured: every method run with a time limit equal to the time "
            "the tuned local search took for $W=6000n$ on the same instance, ten instances per "
            "size. \\emph{rank}, \\emph{rank$_W$}: order of median $f$ here and under equal work "
            "(Table~\\ref{tab:compare}); work and seconds are medians; $p$ as in "
            "Table~\\ref{tab:compare}.",
            "tab:equaltime",
            ["$n$", "method", "rank", "rank$_W$", "feasible", "$f$", "work (M)", "s", "$p$"],
            trows, align="rlrrrrrrr", small=True)

    # --- G7: the crossover-disabled control ----------------------------------------------------------
    path = os.path.join(R, "crossover.csv")
    if t.complete(path):
        rows = t.read(path)
        by = collections.defaultdict(list)
        cfgs = []
        for r in rows:
            if r["config"] not in cfgs:
                cfgs.append(r["config"])
            by[(int(r["n"]), r["config"], r["arm"])].append(r)
        tests, cells = [], []
        for n in sorted({k[0] for k in by}):
            for cfg in cfgs:
                a = {r["seed"]: r for r in by[(n, cfg, "ma")]}
                b = {r["seed"]: r for r in by[(n, cfg, "ma-nox")]}
                ks = sorted(set(a) & set(b), key=int)
                _, p = t.wilcoxon_signed_rank([f_of(a[k]) for k in ks], [f_of(b[k]) for k in ks])
                tests.append(((n, cfg), p))
                cells.append((n, cfg))
        adj = dict(t.holm(tests))
        raw = dict(tests)
        xrows = []
        maZero = maRuns = maxGens = 0
        for (n, cfg) in cells:
            for arm, label in (("ma", "with"), ("ma-nox", "without")):
                v = by[(n, cfg, arm)]
                gens = [t.num(r, "generations") for r in v]
                bg = [t.num(r, "best_generation") for r in v]
                if arm == "ma":
                    maZero += sum(1 for x in bg if x == 0)
                    maRuns += len(bg)
                    maxGens = max(maxGens, int(max(gens)))
                xrows.append([n, cfg, label, t.fmt(t.median(gens), 0),
                              f"{sum(1 for x in bg if x == 0)}/{len(bg)}",
                              t.fmt(t.median([t.num(r, "soft") for r in v]), 0),
                              t.fmt(t.median([f_of(r) for r in v]), 0),
                              t.fmt(t.median([t.num(r, "work") for r in v]) / 1e6, 2),
                              t.fmt_p(raw[(n, cfg)]) if arm == "ma-nox" else "",
                              t.fmt_p(adj[(n, cfg)]) if arm == "ma-nox" else ""])
                tag = "Small" if n == min(k[0] for k in cells) else "Large"
                if arm == "ma":
                    facts["xoverGens" + cfg.capitalize() + tag] = t.fmt(t.median(gens), 0)
                    facts["xoverOffspring" + cfg.capitalize() + tag] = t.fmt(
                        t.median([t.num(r, "offspring") for r in v]), 0)
                else:
                    facts["xoverGensNox" + cfg.capitalize() + tag] = t.fmt(t.median(gens), 0)
        facts["xoverFromZero"] = str(maZero)
        facts["xoverRuns"] = str(maRuns)
        facts["xoverFromZeroPct"] = t.fmt(100.0 * maZero / maRuns, 0)
        facts["xoverMaxGens"] = str(maxGens)
        facts["xoverSeeds"] = str(max(len(v) for v in by.values()))
        facts["xoverMinP"] = t.fmt_p(min(raw.values()))
        facts["xoverMinPadj"] = t.fmt_p(min(adj.values()))
        gen = [r for r in rows if r["config"] == "generational" and r["arm"] == "ma"]
        facts["xoverGenerationalExceptions"] = str(sum(1 for r in gen
                                                       if t.num(r, "best_generation") > 0))
        facts["xoverGenerationalRuns"] = str(len(gen))
        ctl = [r for r in rows if r["arm"] == "ma-nox"]
        facts["xoverControlFromZero"] = str(sum(1 for r in ctl if t.num(r, "best_generation") == 0))
        facts["xoverControlRuns"] = str(len(ctl))
        t.write_table(
            os.path.join(TAB, "crossover.tex"),
            "The memetic algorithm with and without recombination (crossover-disabled MA: "
            "selection, mutation, refinement and replacement kept), $W=6000n$, medians of six "
            "instances. \\emph{from gen.~0}: runs whose result came from the refined initial "
            "population; \\emph{iter.}: median generation iterations, a final partial one "
            "included. $p$: raw signed-rank on $f$; $p_H$: Holm over the four cells.",
            "tab:crossover",
            ["$n$", "config.", "xover", "iter.", "from gen.~0", "soft", "$f$", "work (M)", "$p$",
             "$p_H$"],
            xrows, align="rllrrrrrrr", small=True)

    # How well the projection from per-unit costs predicts the measured subsidy where both exist.
    for tag in ("Small", "Large"):
        n = facts.get("subsidyN" + tag)
        ok = n and f"projRefRatioma{n}" in facts and "subsidyma" + tag in facts
        if ok and facts["subsidyma" + tag] != "--" and facts[f"projRefRatioma{n}"] != "--":
            proj = float(facts[f"projRefRatioma{n}"])
            meas = float(facts["subsidyma" + tag])
            facts["projErr" + tag] = t.fmt(100 * (proj / meas - 1), 0)
    t.write_macros(os.path.join(TAB, "facts.tex"), facts)
    print(f"wrote {len(facts)} macros")


if __name__ == "__main__":
    main()
