#!/usr/bin/env bash
# Run every experiment of this article, in the order their conclusions depend on
# one another.  --quick runs a reduced design that finishes in minutes.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -x build/experiments ]; then
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
  cmake --build build
fi
# The instance files in data/ are not read by the experiments, which generate the same instances in
# memory from (size, seed); they are shipped for other solvers.  Confirm they are exactly what the
# generator produces, then check the generator, the file format and the two independent evaluators
# against each other, and the statistics against their definitions.  A failure here would make every
# number below meaningless.
python3 scripts/check_data.py
mkdir -p results-rerun
./build/experiments selftest --results results-rerun
python3 scripts/check_stats.py

QUICK=0
EXTENDED=0
for arg in "$@"; do
  [ "$arg" = "--quick" ] && QUICK=1
  [ "$arg" = "--extended" ] && EXTENDED=1
done

# A quick run is a smoke test, not a replication.  Point analyse.py at its directory with --results
# to see the tables a reduced design produces.
# A full run writes to results-rerun/, never to results/, which holds the published records; compare
# the two with scripts/compare_results.py.
RES=results-rerun
if [ $QUICK -eq 1 ]; then RES=results-quick; fi
mkdir -p "$RES"
run() { ./build/experiments "$@" --results "$RES"; }

if [ $QUICK -eq 1 ]; then
  run tune       --seeds 2 --budget 200
  run agreement  --seeds 2 --budget 200
  run compare    --seeds 2 --budget 200
  run budgetmode --seeds 2 --budget 200
  run evalcost   --seeds 2
  run evalcost   --seeds 2 --tuned
  run converge   --seeds 2 --budget 200
  run crossover  --seeds 2 --budget 200
  run timed      --seeds 2 --budget 200
else
  # The design of the article (Table 1).  Methods are independent, so on a machine with cores to
  # spare any line can be split with --only (and --seedstart for budgetmode and timed) into
  # processes writing to separate directories whose CSV files are then concatenated; the rows are
  # the same.  Elapsed times are recorded but are not part of any work-budget result.
  run tune       --seeds 5  --budget 6000
  run agreement  --seeds 20 --budget 6000
  run compare    --seeds 10 --budget 6000
  # The equal-evaluation runs of the memetic methods at n = 1600 take about a quarter of an hour
  # each; the default 15-minute safety limit would truncate them, so it is raised.
  run budgetmode --seeds 10 --budget 6000 --timelimit 3600000
  run evalcost   --seeds 6
  run evalcost   --seeds 6 --tuned
  run converge   --seeds 10 --budget 6000
  run crossover  --seeds 6  --budget 6000
  # Equal wall-clock time: run alone, on an otherwise idle machine.
  run timed      --seeds 10 --budget 6000
fi

# Optional, NOT part of the published study: the currency experiment at the TUNED configurations.
# At those configurations one equal-evaluation run of a memetic method is projected to take many
# hours, which is why the article reports only an extrapolation.  Run it with --extended.
if [ $EXTENDED -eq 1 ] && [ $QUICK -eq 0 ]; then
  run budgetmode --seeds 3 --budget 6000 --tuned --maxn 400 --timelimit 36000000
fi

echo
echo "results written to $RES/; compare them with the published records:"
echo "  python3 scripts/compare_results.py results $RES"
echo "and build tables from them with: python3 scripts/analyse.py --results $RES --out <dir>"
