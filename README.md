# Genetic, memetic and local search under one objective: a controlled comparison

Code, input data and per-run records for the article *Genetic, memetic and local search under one
objective: a controlled comparison for university course timetabling* (O. Zanevych, V. Kukharskyy),
prepared for *Radioelectronic and Computer Systems* (REKS). Repository:
<https://github.com/OlehZanevych/timetabling-ga-memetic-local-search-comparison-reks-2026>.

Six methods -- a generational genetic algorithm, a memetic algorithm, a memetic algorithm with an
adaptive refinement rate, a multi-neighbourhood local search, simulated annealing, and random-restart
hill climbing as a control -- on the same generated instances, assessed by the same objective computed
by the same code, configured by the same fixed-budget random configuration search, and given the same
budget of evaluator work: bucket recomputations of the shared incremental evaluator.

The work count is a deterministic proxy for evaluator work, not for all computation: it charges every
recomputation of an (entity, day) bucket, in every evaluator state a run uses (constructions,
scoring, refinements), and nothing else. Its relation to elapsed time is measured, and the comparison
is repeated under equal wall-clock limits.

`src/population.{hpp,cpp}` holds the genetic and memetic algorithms: randomised greedy initial
population (its work charged), community crossover applied with a tuned probability, per-gene
mutation, tournament selection, elitism, and for the memetic variants a bounded refinement (cheap
neighbourhoods, late acceptance or DLAS, no escape) capped by the work that remains. Individuals are
ordered by (illegal, unplaced, H, f), so an illegal child never displaces a legal individual.

## Experiments

| subcommand | what it measures |
|---|---|
| `tune` | fixed-budget random configuration search, 13 candidates per method, mean per-instance rank |
| `agreement` | legality of every method's output at twice the budget, by the independent validator |
| `compare` | the main comparison at a common work budget, with legality and work components per run |
| `budgetmode` | the same methods under a work budget and an evaluation budget (`--tuned` for the tuned configurations) |
| `evalcost` | work per evaluation against n, split into child scoring and total per offspring |
| `converge` | budget response: separate runs of the tuned configurations at a ladder of work budgets |
| `crossover` | the memetic algorithm with recombination disabled |
| `timed` | equal wall-clock time: every method limited to the time the local search took for the work budget |
| `instances` | writes every instance the experiments use (232 files) to `--data DIR` |
| `selftest` | the three implementation checks described below |

`tune` must be run before the others: they read the selected configurations from `tuning.csv` in
their `--results` directory and stop with an error if it is missing (they never fall back to the
defaults silently). An unknown option is an error, not a no-op. Each subcommand writes one CSV
with one line per run; aggregation is the analysis script's job. `--only m1,m2` restricts an
experiment to some methods, and `--seedstart`, `--minn` and `--maxn` restrict `budgetmode` and
`timed` to some seeds and sizes, so that independent parts can run as separate processes; the rows
are the same.

## Checking the implementation first

```bash
./build/experiments selftest
```

Three checks, each a few seconds. The planted schedules of generated instances (sizes 200, 800 and
3200) are re-scored by the independent validator and must be feasible, so that a residual hard cost is never the
generator's fault. An instance written in the plain-text format and read back must score exactly as
the original did, so that the format loses nothing. And the incremental evaluator is compared term
by term with the independent validator along a random walk of
placements, so that two implementations of the objective that share no code cannot be wrong in the
same way. `scripts/run_all.sh` runs it before anything else, together with

```bash
python3 scripts/check_stats.py
```

which checks the statistics in `scripts/ttstats.py` against their definitions rather than against
another library: the exact Wilcoxon signed-rank test is compared with an enumeration of its own null
distribution over several hundred random samples, with and without tied differences; the sign test,
the paired effect size and Holm's correction against values that can be computed by hand; the
chi-squared survival function against the tabulated critical values; and Friedman's statistic, with
and without its tie correction, against its closed form. `ttstats.py` uses the standard library
alone, so the analysis has no dependency to install and no version that can drift; the price is that
its tests have to be written, and this is them.

## Building

Requires a C++20 compiler and CMake 3.16 or newer for the experiments, and Python 3.8 or newer
(standard library only; tested with 3.10 and 3.11) for the scripts. Nothing else: no external libraries, no package manager, no
network access.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

## Input data

The experiments do not read files: every instance is generated in memory, deterministically, from its
size and seed. `data/` nevertheless ships **every instance any experiment uses**, gzipped, with its
planted feasible schedule, for the benefit of a solver that shares no code with this one:

| sizes | seeds | used by |
|---|---|---|
| 200, 400, 800, 1600, 3200 | 1--20 | compare (1--10), agreement (1--20 at 400, 1600), budgetmode, crossover, evalcost (1--6), converge (800), timed (800--3200); seeds 11--20 at 200, 800, 3200 are spare |
| 400, 1600 | 101--105 | tune |
| 6400 | 1--6 | evalcost |

`data/MANIFEST.tsv` lists every file with its size, seed, the experiments that use it and the SHA-256
of its uncompressed text, and

```bash
python3 scripts/check_data.py
```

regenerates all 232 files with `experiments instances` and confirms that each shipped file is
byte-for-byte what the generator produces, so the files are exactly the instances the experiments
ran on. The configurations the experiments use are the second input: `results/tuning.csv`, written
by `tune`.

## Reproducing the paper

There are two ways:

* **Recreate the tables and figures from the supplied records.** `results/` holds every per-run record
  the article uses; `python3 scripts/analyse.py` regenerates the article's tables (including
  `tables/facts.tex`, the numbers quoted in its text) and figure data from them, byte for byte, in
  seconds. No build is needed for this.
* **Rerun the study.** `scripts/run_all.sh` checks `data/`, runs the self-test and the statistics
  checks, and then runs the published design (Table 1 of the article; the default, non-quick
  branch), writing to `results-rerun/` so that the published records in `results/` are never
  overwritten. Then

  ```bash
  python3 scripts/compare_results.py results results-rerun
  ```

  matches every run by its identifying columns and compares every deterministic column exactly
  (elapsed time and the clock-bound equal-time runs are excluded). Serially the full design takes
  well over ten hours; the published runs split independent methods and seeds across two processes
  (`--only`, `--seedstart`, `--minn`, `--maxn`) and concatenated the CSV files, which gives the same
  rows. `--extended` adds the currency experiment at the tuned configurations, which the article does
  not report (it is projected to take many hours per run). `results/PROVENANCE.md` records exactly how
  each published file was produced.

`run_all.sh --quick` runs a reduced design (fewer seeds, smaller budgets) into `results-quick/`, enough
to check that everything works end to end. `--maxbudget K` caps the candidate count of every search
call; it exists for smoke tests only and was not used for any published record.

When pointed at any directory other than the published `results/`, `analyse.py` refuses to read a
results file written in the last two minutes, on the grounds that an experiment appends to its CSV
as it runs and a table built from a half-executed design looks finished without being so. Set
`TT_MIN_AGE=0` to override it. The published records are always read.

Where it writes: if the directory above the repository holds the article's `main.tex`, into that
directory's `tables/` and `figures/`; otherwise (a plain clone) into `tables/` and `figures/` inside
the repository, which `.gitignore` leaves untracked because they are derived from `results/`. Pass
`--out DIR` to choose.

Every number in the paper is produced this way. No value in any table or figure is typed by hand:
the analysis script reads the raw per-run records and writes the LaTeX directly, including the
numbers quoted in the prose, which reach the text as macros in `tables/facts.tex`.

## Layout

```
src/core/        the solver kernel, shared with the other applications in this series
  model.*        instances: entities, domains, the compressed tick axis
  state.*        the schedule, its (entity, day) bucket decomposition, incremental evaluation
  search.*       construction, acceptance, the seven neighbourhoods, the bandit, the escape
  validate.*     an INDEPENDENT scorer, sharing no code with state.cpp -- see below
  generate.*     instances built backwards around a hidden feasible schedule
  io.*           the plain-text instance format
  flow.hpp       Dinic's maximum flow, for the matching bounds
  hungarian.hpp  the assignment problem, for the permutation operator
  mask.hpp       128-bit tick masks
  rng.hpp        xoshiro256**, so that a seed reproduces a run exactly
src/             this application's experiment drivers (main.cpp) and population methods
scripts/         run_all.sh; analyse.py and ttstats.py (statistics, standard library only);
                 check_stats.py, check_data.py, compare_results.py
data/            every instance used, gzipped plain text, and MANIFEST.tsv
results/         one CSV per experiment, one line per run; PROVENANCE.md; logs/ (checks)
CMakeLists.txt   the build (C++20, -O3, one executable: build/experiments)
LICENSE          MIT
```

## Two things that make the numbers trustworthy

**An independent validator.** `src/core/validate.cpp` re-reads a schedule from scratch and counts
every penalty term and every hard filter by the most direct method the definitions allow -- sorting
and scanning, never a bitmask. It deliberately shares no code with the incremental evaluator in
`state.cpp`, so that an error in one cannot hide behind an identical error in the other. Every
schedule-quality number in the paper (f, H, soft cost, legality) comes from the validator, never from the search's own counters; work, evaluations, generations and times are instrumentation.

**Instances with a known answer.** The generator builds a valid schedule first, walking the week slot
by slot and placing each class into resources that are free at that moment and reachable in the time
available, and then reads the instance off it. So a schedule with no hard violation provably exists
for every instance, and anything the search cannot reach is a property of the search rather than of
the data. The planted schedule's own cost is reported alongside every result as a reference; it is a
feasibility witness, not a good timetable.

## Budgets and reproducibility

Every budget is counted in **bucket recomputations** or in **evaluations** (offspring, or applied
candidates of a trajectory method), except the equal-time experiment, which is counted in seconds by
design. Seeds are fixed functions of the instance, the method index and the run; no seed depends on
`std::hash`. With the same compiler, standard library and flags, the deterministic columns of every
work- or evaluation-budget run (objective, hard and soft counts, work, evaluations, generations)
reproduce exactly. Elapsed times do not, and the equal-time experiment, whose stopping rule is a
clock, is reproducible in design only. Floating-point summation order and a different standard library
can in principle change a run; the safety time limits were not reached by any work-budget run.

The published results were produced on a 2-vCPU Intel Xeon virtual machine at 2.8 GHz, Ubuntu
24.04 (Linux 6.18), g++ 13.3 with `-O3`, C++20, single-threaded runs, at most two runs at a time
(the main comparison and the equal-time experiment alone); Python 3.11 for the analysis.

## Licence

MIT; see [LICENSE](LICENSE).
