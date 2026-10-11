# How the published records were produced

Every CSV in this directory was written by `build/experiments` from the sources in `../src`,
between 4 October 2026, 18:37 UTC, and 5 October 2026, 00:38 UTC. The machine was a 2-vCPU Intel Xeon
virtual machine at 2.8 GHz running Ubuntu 24.04 (Linux 6.18). The code was compiled by g++ 13.3 with
`-O3` and C++20 (`CMakeLists.txt`). At most two runs executed at a time. The main comparison and the
equal-time runs executed alone.

Independent parts of an experiment ran as separate processes, each writing its own directory. Their
CSV files were then concatenated: the header once, then the rows. Seeds are fixed functions of
(instance, method, run), so a split run writes exactly the rows a serial run writes. On 11 October
2026 the rows of `tuning*.csv` and `agreement.csv` were put back into the order a serial run writes
them (method order ga, ma, ma-adaptive, sls, sa, rrhc). No value was changed.

| file | rows | command(s), all with `--budget 6000` | ran |
|---|---:|---|---|
| `tuning.csv`, `tuning_runs.csv` | 507, 780 | `tune --seeds 5 --only ga,ma,sls` and `tune --seeds 5 --only ma-adaptive,sa,rrhc` | 4 Oct 18:37–19:11 |
| `compare.csv` | 300 | `compare --seeds 10`, alone | 4 Oct 19:11–19:46 |
| `budgetmode.csv` (n = 400) | 120 | `budgetmode --seeds 5 --seedstart 1` and `budgetmode --seeds 10 --seedstart 6`; their n = 400 rows only | 4 Oct 19:46–20:30 |
| `budgetmode.csv` (n = 1600) | 120 | `budgetmode --seeds s --seedstart s --minn 1600 --timelimit 3600000`, for s = 1..10 | 4 Oct 21:37–23:54 |
| `agreement.csv` | 240 | `agreement --seeds 20 --only ga,ma,ma-adaptive` and `agreement --seeds 20 --only sls,sa,rrhc` | 4 Oct 20:30–20:55 |
| `converge.csv` | 480 | `converge --seeds 10` | 4 Oct, to 21:13 |
| `evalcost.csv` | 216 | `evalcost --seeds 6` | 4 Oct, to 21:19 |
| `crossover.csv` | 48 | `crossover --seeds 6` | 4 Oct, to 21:22 |
| `evalcost_tuned.csv` | 216 | `evalcost --seeds 6 --tuned` | 4 Oct, to 21:37 |
| `timed.csv` | 180 | `timed --seeds 10 --minn n --maxn n`, for n = 800, 1600, 3200, one after another, alone | 4 Oct 23:54 – 5 Oct 00:38 |

## Notes

- **The n = 1600 currency runs were restarted with a longer deadline.** The first n = 1600 attempt
  inside the two `budgetmode` processes was stopped: its equal-evaluation memetic runs were reaching
  the default 15-minute safety deadline. Those rows were discarded, and n = 1600 was rerun per seed
  with `--timelimit 3600000`.
- **No published run was truncated.** Every equal-evaluation run spent its whole evaluation budget,
  and every work-budget run spent its whole work budget. The analysis checks both.
- **Three builds of the same algorithm were used.**
  - `tune` ran on the first build of the corrected code.
  - The other experiments ran on a build that adds the `evaluator_agrees` column to `compare.csv`.
  - The n = 1600 currency runs and `timed` ran on a build that adds the `--minn` option.

  None of these additions changes any computation.
- **Source changes on 11 October do not touch any computation.** They add:
  - explicit errors for a missing `tuning.csv`, an unknown option or an unwritable results
    directory;
  - the full list of instances to `instances`.
- **Elapsed times are machine-dependent.** This covers the `elapsed_ms` columns and the whole of
  `timed.csv`, whose runs stop on a wall-clock limit. Every other column reproduces exactly with the
  same compiler and standard library.

## Verification on 11 October 2026

The sources in `../src` were built afresh. These experiments were then rerun:
- `compare` (all 300 runs);
- `crossover` (all 48);
- `converge` and both cost probes (`evalcost`, with and without `--tuned`);
- parts of `tune`, `budgetmode` and `agreement`.

`scripts/compare_results.py` compared the reruns with the files here. Every compared run was
identical in every deterministic column:

| file | runs compared |
|---|---|
| `compare.csv` | all 300 |
| `converge.csv` | all 480 |
| `crossover.csv` | all 48 |
| `evalcost.csv` | all 216 |
| `tuning_runs.csv` | 260 (ga and rrhc, all 13 configurations each) |
| `budgetmode.csv` | 24 (n = 400, seeds 1–2) |
| `agreement.csv` | 80 (ga and rrhc) |
| `evalcost_tuned.csv` | all 216 |

`tune`, `budgetmode`, `agreement` and the tuned cost probe were run with a build of the 11 October
sources. The others used a build of the sources as of 5 October.

`timed.csv` was not rerun, because its runs stop on a wall-clock limit.

The full output is in `logs/verification-2026-10-11.log`.

The repository copy was also checked on a second machine with a different compiler, g++ 11.4 on
Ubuntu 22.04 with Python 3.10:
- the self-test, the data check and the statistics checks passed;
- `analyse.py` regenerated the article's tables and figure data byte for byte;
- the 83 `compare` runs rerun there (all ga and rrhc runs up to n = 1600, and three at n = 3200) and
  all 48 `crossover` runs were identical to the records.

`logs/` also holds the output of the self-test (`selftest.log`), the statistics checks
(`check_stats.log`) and the data check (`check_data.log`). It also holds the progress log of the
original runs (`run-progress-2026-10-04.log`).

## MD5 of the published records

```
3fed06985c624827f584dace286cb7c2  agreement.csv
1097a3f5fd73cbc215e54dbb2ee14e14  budgetmode.csv
bc90f49b844c138e63e433e472f2d642  compare.csv
b7f5bf3e5e5f5121b89eec7ef8796b85  converge.csv
c127636190656b90dd5ea59b42ce8391  crossover.csv
4678c8e0964bf1ca6d6388fbad46cde6  evalcost.csv
6b6250aeff82441954cc68d96c5fd061  evalcost_tuned.csv
0c29da2f0eaee64aa48e6fef64b7936a  timed.csv
be6ae3cc6a87aab7f53daf7dcdde09e1  tuning.csv
ed30c5a56edae78462f66aab4811485a  tuning_runs.csv
```
