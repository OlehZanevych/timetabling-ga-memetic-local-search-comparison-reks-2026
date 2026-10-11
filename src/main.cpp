// Experiments for "Genetic, memetic and local search under one objective".
//
//   agreement  G1  every method's output checked by the independent validator, at twice the budget
//   tune       G2  random configuration search, one configuration budget per method
//   compare    G3  the main comparison, at a common work budget
//   budgetmode G4  the same methods under a work budget and under an evaluation budget
//   evalcost   G5  the work one evaluation costs, per method, against n
//   converge   G6  budget response: final quality at a ladder of work budgets
//   crossover  G7  the memetic algorithm with recombination disabled
//   timed      G8  the comparison under equal wall-clock time limits
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "core/generate.hpp"
#include "core/io.hpp"
#include "core/search.hpp"
#include "core/selftest.hpp"
#include "core/validate.hpp"
#include "population.hpp"

using namespace tt;
using Clock = std::chrono::steady_clock;

namespace {

std::string gResults = "results";

/// Stop with a message.  An experiment that cannot write its records, or that would silently run
/// at a configuration other than the one the article reports, must not produce a results file.
[[noreturn]] void die(const std::string& msg) {
  std::fprintf(stderr, "error: %s\n", msg.c_str());
  std::exit(1);
}

std::ofstream openCsv(const std::string& name, const std::string& header) {
  std::error_code ec;
  std::filesystem::create_directories(gResults, ec);
  std::ofstream f(gResults + "/" + name);
  if (!f) die("cannot write " + gResults + "/" + name);
  // Unbuffered: one run is one line and runs are seconds apart, so flushing costs nothing and a
  // long experiment can be watched, or salvaged, while it is still running.
  f << std::unitbuf;
  f << header << "\n";
  return f;
}

GeneratedInstance makeInstance(int n, uint64_t seed) {
  GenOptions go;
  go.nClasses = n;
  go.seed = seed;
  return generateInstance(go);
}

/// One run of one method under a common work budget, returning the schedule it found.
struct Outcome {
  std::vector<Placement> best;
  long long work = 0, elapsedMs = 0, evaluations = 0;
  /// Generations, for a population method; zero for a trajectory one.  Reported because a memetic
  /// algorithm that completes no generation is not a memetic algorithm, and a reader should be able
  /// to see that it completed some.
  long long generations = 0;
  /// The generation the returned schedule was found in; zero means the refined initial population
  /// was never improved on.
  long long bestGeneration = 0;
  /// Total work divided by outer evaluations (offspring, or candidates of a trajectory method).
  double workPerEvaluation = 0;
  // Components of a population method's work (zero for a trajectory method).
  long long constructionWork = 0, initialScoringWork = 0, initialRefineWork = 0;
  long long childScoringWork = 0, childRefineWork = 0, refinedChildren = 0, refineCandidates = 0;
  double childScoringPerOffspring = 0, meanChangedGenes = 0, meanParentDistance = 0;
  /// The method's own evaluator's account of the returned schedule, for the agreement check.
  double ownHard = 0, ownObjective = 0;
};

/// Does the method's own (incremental) account of its returned schedule agree with the validator?
int agrees(const Outcome& o, const Score& sc) {
  return (std::fabs(o.ownHard - sc.hard()) < 1e-6 &&
          std::fabs(o.ownObjective - sc.objective(kDefaultBeta)) <= 1e-9 * std::max(1.0, o.ownObjective))
             ? 1 : 0;
}

/// The population-method work components as CSV fields, in the order of kComponentHeader.
const char* kComponentHeader =
    "construction_work,initial_scoring_work,initial_refine_work,child_scoring_work,"
    "child_refine_work,refined_children,refine_candidates,child_scoring_per_offspring,"
    "mean_changed_genes,mean_parent_distance";
std::string components(const Outcome& o) {
  std::ostringstream s;
  s << o.constructionWork << "," << o.initialScoringWork << "," << o.initialRefineWork << ","
    << o.childScoringWork << "," << o.childRefineWork << "," << o.refinedChildren << ","
    << o.refineCandidates << "," << o.childScoringPerOffspring << "," << o.meanChangedGenes << ","
    << o.meanParentDistance;
  return s.str();
}

/// Legality fields of a validated schedule, in the order of kLegalityHeader.
const char* kLegalityHeader = "unplaced,domain_breaches,availability_breaches,cap_breaches,clean";
std::string legality(const Score& sc) {
  std::ostringstream s;
  s << sc.unplaced << "," << sc.domainBreaches << "," << sc.availabilityBreaches << ","
    << sc.capBreaches << "," << (sc.clean() ? 1 : 0);
  return s.str();
}

struct MethodConfig {
  // shared
  long long workBudget = 20000000;   ///< bucket recomputations
  int64_t timeLimitMs = 900000;
  uint64_t seed = 1;
  // population methods
  int population = 30, tournament = 3, elite = 2;
  double crossoverRate = 0.9, mutationRate = 0.03, lsProbability = 0.3;
  long long lsCandidates = 2000;
  std::string lsEngine = "lahc";   ///< acceptance rule of the memetic refinement
  int lsHistory = 50;              ///< its history length
  long long evalBudget = 0;   ///< offspring / candidate evaluations, when the currency is not work
  // local searches
  std::string engine = "lahc";
  int lahcLength = 100;
  bool bigNeighbourhoods = true, escape = true;
  // trajectory methods: the parameters each engine's own code reads.  They are here rather than in
  // the shared block because the configuration search gives every method a grid over *its own*
  // parameters; a parameter a method never reads cannot be tuned for it.
  double saT0Factor = 0.0015, saCooling = 0.995;
  long long saReheatAfter = 400000;
  long long stagnationMoves = 60000, restartAfter = 1200000, rrhcRestart = 20000;
  int kickMax = 300, chainDepth = 3, lnsMax = 40;
  double banditDecay = 0.7;
  std::string construction = "mrv";
};

Outcome runMethod(const std::string& method, const Instance& in, const MethodConfig& cfg) {
  Outcome out;
  if (method == "ga" || method == "ma" || method == "ma-adaptive" || method == "ma-nox") {
    ga::GaOptions go;
    go.population = cfg.population;
    go.tournament = cfg.tournament;
    go.elite = cfg.elite;
    go.crossoverRate = cfg.crossoverRate;
    go.mutationRate = cfg.mutationRate;
    go.workBudget = cfg.workBudget;
    go.offspringBudget = cfg.evalBudget;
    go.timeLimitMs = cfg.timeLimitMs;
    go.seed = cfg.seed;
    go.localSearch = (method != "ga");
    go.adaptiveLs = (method == "ma-adaptive");
    go.lsProbability = cfg.lsProbability;
    go.lsCandidates = cfg.lsCandidates;
    go.lsEngine = cfg.lsEngine;
    go.lsHistory = cfg.lsHistory;
    go.crossover = (method != "ma-nox");
    ga::GaResult r = ga::runGa(in, go);
    out.best = r.best.genes;
    out.ownHard = r.best.hard;
    out.ownObjective = r.best.objective;
    out.work = r.work;
    out.elapsedMs = r.elapsedMs;
    out.evaluations = r.offspring;
    out.generations = r.generations;
    out.bestGeneration = r.bestGeneration;
    out.workPerEvaluation = r.workPerOffspring;
    out.constructionWork = r.constructionWork;
    out.initialScoringWork = r.initialScoringWork;
    out.initialRefineWork = r.initialRefineWork;
    out.childScoringWork = r.childScoringWork;
    out.childRefineWork = r.childRefineWork;
    out.refinedChildren = r.refinedChildren;
    out.refineCandidates = r.refineCandidates;
    out.childScoringPerOffspring = r.childScoringPerOffspring;
    out.meanChangedGenes = r.meanChangedGenes;
    out.meanParentDistance = r.meanParentDistance;
    return out;
  }

  SearchOptions so;
  so.timeLimitMs = cfg.timeLimitMs;
  so.candidateLimit = cfg.evalBudget;
  so.workLimit = cfg.evalBudget > 0 ? 0 : cfg.workBudget;
  so.threads = 1;
  so.seed = cfg.seed;
  so.engine = cfg.engine;
  so.lahcLength = cfg.lahcLength;
  so.saT0Factor = cfg.saT0Factor;
  so.saCooling = cfg.saCooling;
  so.saReheatAfter = cfg.saReheatAfter;
  so.stagnationMoves = cfg.stagnationMoves;
  so.restartAfter = cfg.restartAfter;
  so.kickMax = cfg.kickMax;
  so.chainDepth = cfg.chainDepth;
  so.lnsMax = cfg.lnsMax;
  so.banditDecay = cfg.banditDecay;
  so.construction = cfg.construction;
  if (method == "sls") {
    // the full portfolio
  } else if (method == "sa") {
    so.engine = "sa";
  } else if (method == "rrhc") {
    so.engine = "hc";
    so.useLns = so.useRepack = so.useKopt = so.useChain = so.useKempe = false;
    so.useDeep = false;
    so.useKick = false;
    so.useFresh = true;
    so.restartAfter = cfg.rrhcRestart;
  }
  if (!cfg.bigNeighbourhoods) so.useLns = so.useRepack = so.useKopt = false;
  if (!cfg.escape) so.useDeep = so.useKick = so.useFresh = false;
  SearchResult r = solve(in, so);
  out.best = r.best;
  out.ownHard = r.hard;
  out.ownObjective = r.objective;
  out.work = r.workUnits;
  out.elapsedMs = r.elapsedMs;
  out.evaluations = r.candidates;
  out.workPerEvaluation = r.candidates ? double(r.workUnits) / r.candidates : 0;
  return out;
}

const std::vector<std::string> kAllMethods = {"ga", "ma", "ma-adaptive", "sls", "sa", "rrhc"};
/// Restriction to a subset of the methods (--only a,b,c), so that independent methods can run as
/// separate processes on separate cores; the rows produced are the same.
std::vector<std::string> gOnly;
std::vector<std::string> methods() {
  if (gOnly.empty()) return kAllMethods;
  std::vector<std::string> out;
  for (const auto& m : kAllMethods)
    if (std::find(gOnly.begin(), gOnly.end(), m) != gOnly.end()) out.push_back(m);
  return out;
}
/// A fixed per-method index, used where a seed has to depend on the method.  (An earlier version
/// hashed the name with std::hash, whose values are not specified across standard libraries.)
uint64_t methodIndex(const std::string& m) {
  for (size_t i = 0; i < kAllMethods.size(); ++i)
    if (kAllMethods[i] == m) return i + 1;
  return 99;
}

/// Per-run wall-clock safety deadline, and the largest instance size an experiment will visit.
/// Both exist for the long currency runs; see the option parsing.
int64_t gTimeLimitMs = 900000;
int gMaxN = 0;   ///< 0 means no restriction
int gMinN = 0;   ///< likewise, the smallest size visited

/// First seed an experiment visits.  Seeds are independent -- run i uses the fixed stream 1212 + i
/// against the instance generated from i -- so seeds 1..k in one process and seeds 1..j, j+1..k in
/// two produce the same rows.  It exists so that the long currency and equal-time experiments can be
/// split across cores.
int gSeedStart = 1;

/// Defined below; declared here because the legality check also runs the tuned configurations.
std::map<std::string, MethodConfig> loadTuned(long long budget);
/// As loadTuned, but stops unless results/tuning.csv holds a configuration for every method this
/// run visits: without it an experiment would silently fall back to the defaults.
std::map<std::string, MethodConfig> requireTuned(long long budget);

// =================================================================================================

void expAgreement(int seeds, long long budget) {
  // The legality check is run with the configurations the tuning chose -- the ones the comparison
  // actually uses -- and at twice the comparison's per-class work budget, since a rule violation is
  // something a method has to be given room to commit.  The main comparison records the same
  // legality fields for every one of its own runs, at every size.
  auto tuned = requireTuned(budget);
  auto f = openCsv("agreement.csv",
                   std::string("n,seed,method,validator_hard,validator_soft,validator_objective,") +
                       kLegalityHeader);
  for (int n : {400, 1600}) {
    for (const std::string& m : methods()) {
      for (int seed = 1; seed <= seeds; ++seed) {
        GeneratedInstance gi = makeInstance(n, seed);
        MethodConfig cfg = tuned.count(m) ? tuned[m] : MethodConfig{};
        cfg.workBudget = 2 * budget * n;
        cfg.seed = 3000 + seed;
        Outcome o = runMethod(m, gi.instance, cfg);
        Score sc = validate(gi.instance, o.best);
        f << n << "," << seed << "," << m << "," << sc.hard() << "," << sc.soft() << ","
          << sc.objective(kDefaultBeta) << "," << legality(sc) << "\n";
      }
    }
  }
  std::printf("agreement done\n");
}

/// The lexicographic key every reported comparison uses: an illegal schedule (one the validator
/// rejects) is worse than any legal one, then fewer unplaced classes, then smaller H, then smaller f.
struct RunKey {
  int illegal = 0, unplaced = 0;
  double hard = 0, objective = 0;
  bool operator<(const RunKey& o) const {
    if (illegal != o.illegal) return illegal < o.illegal;
    if (unplaced != o.unplaced) return unplaced < o.unplaced;
    if (hard != o.hard) return hard < o.hard;
    return objective < o.objective;
  }
  bool operator==(const RunKey& o) const {
    return illegal == o.illegal && unplaced == o.unplaced && hard == o.hard &&
           objective == o.objective;
  }
};

void expTune(int seeds, long long budget) {
  // Random configuration search with an equal configuration budget for every method: each method
  // gets the same number of candidate configurations, every one of them is run on every tuning
  // instance (there is no elimination, so this is not a race), and the grid is over *the parameters
  // that method's own code reads* -- each listed parameter reaches the code path it is named after.
  //
  // Configuration 0 of every method is the reference configuration -- the implementation's own
  // defaults -- entered unchanged, so that no arm can leave the tuning worse than it started.
  //
  // Selection.  The tuning instances come in two sizes whose objectives differ by an order of
  // magnitude, so pooling raw objectives would let the larger size decide alone.  Each candidate is
  // therefore ranked against the other candidates of the same method on each tuning instance, by
  // the lexicographic key (illegal, unplaced, H, f) with ties sharing the mean rank, and the
  // configuration with the smallest mean rank over all tuning instances is selected; a tie on the
  // mean rank goes to the lower configuration index, hence to the reference.
  auto f = openCsv("tuning.csv", "method,config,param,value,mean_rank,runs");
  auto detail = openCsv("tuning_runs.csv",
                        "method,config,n,seed,hard,soft,objective,work,unplaced,clean");
  const int nConfigs = 12;
  for (const std::string& ms : methods()) {
    const bool isPop = (ms == "ga" || ms == "ma" || ms == "ma-adaptive");
    const bool hasLs = (ms == "ma" || ms == "ma-adaptive");
    std::vector<std::vector<RunKey>> keys(nConfigs + 1);
    std::vector<std::vector<std::pair<std::string, double>>> sampledAll(nConfigs + 1);
    for (int ci = 0; ci <= nConfigs; ++ci) {
      const bool reference = (ci == 0);
      Rng rng(0xC0FFEE + ci * 131 + 7919 * methodIndex(ms));
      MethodConfig cfg;
      auto& sampled = sampledAll[ci];
      auto note = [&](const char* k, double v) { sampled.emplace_back(k, v); };
      if (isPop) {
        // Recombination is the community crossover throughout; the number of parents is not a
        // parameter of that operator and is not sampled.
        if (!reference) cfg.population = 10 + 10 * static_cast<int>(rng.below(6));         // 10..60
        if (!reference) cfg.tournament = 2 + static_cast<int>(rng.below(4));               // 2..5
        if (!reference) cfg.elite = static_cast<int>(rng.below(4));                        // 0..3
        if (!reference) cfg.crossoverRate = 0.5 + 0.1 * static_cast<int>(rng.below(6));    // 0.5..1.0
        if (!reference) cfg.mutationRate = 0.005 * (1 + static_cast<int>(rng.below(12)));  // 0.005..0.06
        note("population", cfg.population);
        note("tournament", cfg.tournament);
        note("elite", cfg.elite);
        note("crossover_rate", cfg.crossoverRate);
        note("mutation_rate", cfg.mutationRate);
      }
      if (hasLs) {
        if (!reference) cfg.lsProbability = 0.1 + 0.15 * static_cast<int>(rng.below(6));   // 0.1..0.85
        if (!reference) cfg.lsCandidates = 1000LL << static_cast<int>(rng.below(6));       // 1k..32k
        if (!reference) cfg.lsHistory = 1 << (3 + static_cast<int>(rng.below(8)));         // 8..1024
        if (!reference) cfg.lsEngine = (rng.below(2) ? "lahc" : "dlas");
        note("ls_probability", cfg.lsProbability);
        note("ls_candidates", static_cast<double>(cfg.lsCandidates));
        note("ls_history", cfg.lsHistory);
        note("ls_engine_dlas", cfg.lsEngine == "dlas" ? 1 : 0);
      }
      if (ms == "sls") {
        if (!reference) cfg.lahcLength = 1 << (3 + static_cast<int>(rng.below(8)));        // 8..1024
        if (!reference) cfg.engine = (rng.below(2) ? "lahc" : "dlas");
        if (!reference) cfg.stagnationMoves = 10000LL << static_cast<int>(rng.below(5));   // 10k..160k
        if (!reference) cfg.kickMax = 25 * (1 + static_cast<int>(rng.below(16)));          // 25..400
        if (!reference) cfg.chainDepth = 2 + static_cast<int>(rng.below(4));               // 2..5
        if (!reference) cfg.lnsMax = 10 + 10 * static_cast<int>(rng.below(8));             // 10..80
        if (!reference) cfg.banditDecay = 0.5 + 0.05 * static_cast<int>(rng.below(9));     // 0.50..0.90
        note("lahc_length", cfg.lahcLength);
        note("engine_dlas", cfg.engine == "dlas" ? 1 : 0);
        note("stagnation_moves", static_cast<double>(cfg.stagnationMoves));
        note("kick_max", cfg.kickMax);
        note("chain_depth", cfg.chainDepth);
        note("lns_max", cfg.lnsMax);
        note("bandit_decay", cfg.banditDecay);
      }
      if (ms == "sa") {
        if (!reference) cfg.saT0Factor = 0.0005 * (1 + static_cast<int>(rng.below(12)));   // 0.0005..0.006
        if (!reference) cfg.saCooling = 0.980 + 0.002 * static_cast<int>(rng.below(10));   // 0.980..0.998
        if (!reference) cfg.saReheatAfter = 50000LL << static_cast<int>(rng.below(6));     // 50k..1.6M
        if (!reference) cfg.stagnationMoves = 10000LL << static_cast<int>(rng.below(5));
        if (!reference) cfg.kickMax = 25 * (1 + static_cast<int>(rng.below(16)));
        if (!reference) cfg.lnsMax = 10 + 10 * static_cast<int>(rng.below(8));
        if (!reference) cfg.banditDecay = 0.5 + 0.05 * static_cast<int>(rng.below(9));
        note("sa_t0_factor", cfg.saT0Factor);
        note("sa_cooling", cfg.saCooling);
        note("sa_reheat_after", static_cast<double>(cfg.saReheatAfter));
        note("stagnation_moves", static_cast<double>(cfg.stagnationMoves));
        note("kick_max", cfg.kickMax);
        note("lns_max", cfg.lnsMax);
        note("bandit_decay", cfg.banditDecay);
      }
      if (ms == "rrhc") {
        // The two counters that gate the restart; both reach the restart test.
        if (!reference) cfg.rrhcRestart = 2500LL << static_cast<int>(rng.below(7));        // 2.5k..160k
        if (!reference) cfg.stagnationMoves = 2000LL << static_cast<int>(rng.below(6));    // 2k..64k
        note("restart_after", static_cast<double>(cfg.rrhcRestart));
        note("stagnation_moves", static_cast<double>(cfg.stagnationMoves));
      }
      cfg.bigNeighbourhoods = true;
      cfg.escape = true;

      for (int n : {400, 1600}) {
        for (int seed = 101; seed < 101 + seeds; ++seed) {
          GeneratedInstance gi = makeInstance(n, seed);
          cfg.workBudget = budget * n;
          cfg.seed = 50000 + seed;
          Outcome o = runMethod(ms, gi.instance, cfg);
          Score sc = validate(gi.instance, o.best);
          keys[ci].push_back(RunKey{sc.clean() ? 0 : 1, sc.unplaced, sc.hard(),
                                    sc.objective(kDefaultBeta)});
          detail << ms << "," << ci << "," << n << "," << seed << "," << sc.hard() << ","
                 << sc.soft() << "," << sc.objective(kDefaultBeta) << "," << o.work << ","
                 << sc.unplaced << "," << (sc.clean() ? 1 : 0) << "\n";
        }
      }
    }
    // Mean rank over the tuning instances.
    const size_t nInst = keys[0].size();
    std::vector<double> meanRank(nConfigs + 1, 0.0);
    for (size_t k = 0; k < nInst; ++k) {
      for (int ci = 0; ci <= nConfigs; ++ci) {
        int less = 0, equal = 0;
        for (int cj = 0; cj <= nConfigs; ++cj) {
          if (keys[cj][k] < keys[ci][k]) ++less;
          else if (keys[cj][k] == keys[ci][k]) ++equal;   // includes ci itself
        }
        meanRank[ci] += less + (equal + 1) / 2.0;
      }
    }
    for (int ci = 0; ci <= nConfigs; ++ci) {
      meanRank[ci] /= static_cast<double>(nInst);
      for (const auto& pv : sampledAll[ci])
        f << ms << "," << ci << "," << pv.first << "," << pv.second << "," << meanRank[ci] << ","
          << nInst << "\n";
    }
  }
  std::printf("tune done\n");
}

/// Read the selected configuration per method back out of the tuning run, so that every later
/// experiment uses exactly what the tuning chose: the smallest mean rank, ties to the lower index.
std::map<std::string, MethodConfig> loadTuned(long long /*unused*/) {
  std::map<std::string, MethodConfig> out;
  std::ifstream f(gResults + "/tuning.csv");
  std::string line;
  if (!f) return out;
  std::getline(f, line);
  std::map<std::string, std::map<int, double>> best;        // method -> config -> mean rank
  std::map<std::string, std::map<int, std::map<std::string, double>>> params;
  while (std::getline(f, line)) {
    std::vector<std::string> col;
    size_t p = 0;
    while (true) {
      const size_t q = line.find(',', p);
      col.push_back(line.substr(p, q == std::string::npos ? q : q - p));
      if (q == std::string::npos) break;
      p = q + 1;
    }
    if (col.size() < 6) continue;
    const std::string m = col[0];
    const int ci = std::stoi(col[1]);
    params[m][ci][col[2]] = std::stod(col[3]);
    best[m][ci] = std::stod(col[4]);
  }
  for (const auto& kv : best) {
    int argbest = -1;
    double bestVal = 1e300;
    for (const auto& c : kv.second)   // std::map iterates configurations in index order
      if (c.second < bestVal - 1e-12) {
        bestVal = c.second;
        argbest = c.first;
      }
    MethodConfig cfg;
    if (argbest >= 0) {
      const auto& pr = params[kv.first][argbest];
      auto get = [&](const char* k, double dflt) {
        auto it = pr.find(k);
        return it == pr.end() ? dflt : it->second;
      };
      cfg.population = static_cast<int>(get("population", 30));
      cfg.tournament = static_cast<int>(get("tournament", 3));
      cfg.elite = static_cast<int>(get("elite", 2));
      cfg.crossoverRate = get("crossover_rate", 0.9);
      cfg.mutationRate = get("mutation_rate", 0.03);
      cfg.lsProbability = get("ls_probability", 0.3);
      cfg.lsCandidates = static_cast<long long>(get("ls_candidates", 2000));
      cfg.lsHistory = static_cast<int>(get("ls_history", 50));
      cfg.lsEngine = get("ls_engine_dlas", 0) > 0.5 ? "dlas" : "lahc";
      cfg.lahcLength = static_cast<int>(get("lahc_length", 100));
      cfg.engine = get("engine_dlas", 0) > 0.5 ? "dlas" : "lahc";
      cfg.saT0Factor = get("sa_t0_factor", 0.0015);
      cfg.saCooling = get("sa_cooling", 0.995);
      cfg.saReheatAfter = static_cast<long long>(get("sa_reheat_after", 400000));
      cfg.stagnationMoves = static_cast<long long>(get("stagnation_moves", 60000));
      cfg.rrhcRestart = static_cast<long long>(get("restart_after", 20000));
      cfg.kickMax = static_cast<int>(get("kick_max", 300));
      cfg.chainDepth = static_cast<int>(get("chain_depth", 3));
      cfg.lnsMax = static_cast<int>(get("lns_max", 40));
      cfg.banditDecay = get("bandit_decay", 0.7);
    }
    out[kv.first] = cfg;
  }
  return out;
}

std::map<std::string, MethodConfig> requireTuned(long long budget) {
  auto tuned = loadTuned(budget);
  std::vector<std::string> need = methods();
  need.push_back("ma");   // the crossover experiment's tuned arm
  for (const std::string& m : need)
    if (!tuned.count(m))
      die("no tuned configuration for '" + m + "' in " + gResults +
          "/tuning.csv; run `experiments tune` first (or copy the published tuning.csv)");
  return tuned;
}

void expCompare(int seeds, long long budget) {
  auto tuned = requireTuned(budget);
  auto f = openCsv("compare.csv",
                   std::string("n,seed,method,hard,soft,objective,planted_soft,budget,work,"
                               "elapsed_ms,evaluations,work_per_evaluation,generations,"
                               "best_generation,evaluator_agrees,") +
                       kLegalityHeader + "," + kComponentHeader);
  for (int n : {200, 400, 800, 1600, 3200}) {
    for (const std::string& m : methods()) {
      for (int seed = 1; seed <= seeds; ++seed) {
        GeneratedInstance gi = makeInstance(n, seed);
        Score ref = validate(gi.instance, gi.planted);
        MethodConfig cfg = tuned.count(m) ? tuned[m] : MethodConfig{};
        cfg.workBudget = budget * n;
        cfg.seed = 606060 + seed;
        Outcome o = runMethod(m, gi.instance, cfg);
        Score sc = validate(gi.instance, o.best);
        f << n << "," << seed << "," << m << "," << sc.hard() << "," << sc.soft() << ","
          << sc.objective(kDefaultBeta) << "," << ref.soft() << "," << cfg.workBudget << ","
          << o.work << "," << o.elapsedMs << "," << o.evaluations << "," << o.workPerEvaluation
          << "," << o.generations << "," << o.bestGeneration << "," << agrees(o, sc) << ","
          << legality(sc) << "," << components(o) << "\n";
      }
    }
  }
  std::printf("compare done\n");
}

void expBudgetMode(int seeds, bool tuned, long long budget) {
  // The same six methods, the same instances, two budget currencies.  Under a budget counted in
  // *evaluations* -- one offspring for a population method, one candidate move for a trajectory
  // method -- the outer units do not cost the same work.  Under a budget counted in bucket
  // recomputations they are charged in the same unit.
  //
  // Run twice: at the reference configurations, and at the tuned ones the comparison uses.
  auto tunedCfg = tuned ? requireTuned(budget) : std::map<std::string, MethodConfig>{};
  auto f = openCsv(tuned ? "budgetmode_tuned.csv" : "budgetmode.csv",
                   std::string("n,seed,method,currency,budget,hard,soft,objective,work,evaluations,"
                               "generations,elapsed_ms,") +
                       kLegalityHeader + "," + kComponentHeader);
  for (int n : {400, 1600}) {
    if (gMaxN > 0 && n > gMaxN) continue;
    if (n < gMinN) continue;
    for (int seed = gSeedStart; seed <= seeds; ++seed) {
      // One run of the local search under the work budget fixes the evaluation budget for every
      // method on this instance.
      long long slsEvaluations = 0;
      {
        GeneratedInstance gi = makeInstance(n, seed);
        MethodConfig cfg = tunedCfg.count("sls") ? tunedCfg["sls"] : MethodConfig{};
        cfg.workBudget = budget * n;
        cfg.seed = 1212 + seed;
        cfg.timeLimitMs = gTimeLimitMs;
        slsEvaluations = std::max(1LL, runMethod("sls", gi.instance, cfg).evaluations);
      }
      for (const std::string& m : methods()) {
        GeneratedInstance gi = makeInstance(n, seed);
        {
          MethodConfig cfg = tunedCfg.count(m) ? tunedCfg[m] : MethodConfig{};
          cfg.workBudget = budget * n;
          cfg.seed = 1212 + seed;
          cfg.timeLimitMs = gTimeLimitMs;
          Outcome o = runMethod(m, gi.instance, cfg);
          Score sc = validate(gi.instance, o.best);
          f << n << "," << seed << "," << m << ",work," << cfg.workBudget << "," << sc.hard() << ","
            << sc.soft() << "," << sc.objective(kDefaultBeta) << "," << o.work << ","
            << o.evaluations << "," << o.generations << "," << o.elapsedMs << ","
            << legality(sc) << "," << components(o) << "\n";
        }
        {
          // The evaluation budget is the number of evaluations the LOCAL SEARCH performed under the
          // work budget above.  The nominal work budget is still passed: it sizes a memetic
          // method's initial refinement (a fifth of it), so generation zero is refined the same way
          // under both currencies; it is not a stopping rule here.
          MethodConfig cfg = tunedCfg.count(m) ? tunedCfg[m] : MethodConfig{};
          cfg.evalBudget = slsEvaluations;
          cfg.workBudget = budget * n;
          cfg.seed = 1212 + seed;
          cfg.timeLimitMs = gTimeLimitMs;
          Outcome o = runMethod(m, gi.instance, cfg);
          Score sc = validate(gi.instance, o.best);
          f << n << "," << seed << "," << m << ",evaluations," << cfg.evalBudget << ","
            << sc.hard() << "," << sc.soft() << "," << sc.objective(kDefaultBeta) << "," << o.work
            << "," << o.evaluations << "," << o.generations << "," << o.elapsedMs << ","
            << legality(sc) << "," << components(o) << "\n";
        }
      }
    }
  }
  std::printf("budgetmode done (%s)\n", tuned ? "tuned" : "reference");
}

/// The work one evaluation costs, per method, against instance size, with the population methods'
/// work split into its components: the direct cost of scoring a child, and the total work per
/// offspring (which includes construction, refinement and the scoring of generation zero).
void expEvalCost(int seeds, bool tuned, long long budget) {
  auto tunedCfg = tuned ? requireTuned(budget) : std::map<std::string, MethodConfig>{};
  auto f = openCsv(tuned ? "evalcost_tuned.csv" : "evalcost.csv",
                   std::string("n,method,seed,work_per_evaluation,evaluations,work,") +
                       kComponentHeader);
  for (int n : {200, 400, 800, 1600, 3200, 6400}) {
    for (const std::string& m : methods()) {
      for (int seed = 1; seed <= seeds; ++seed) {
        GeneratedInstance gi = makeInstance(n, seed);
        MethodConfig cfg = tunedCfg.count(m) ? tunedCfg[m] : MethodConfig{};
        cfg.workBudget = 3000LL * n;
        cfg.seed = 111 + seed;
        Outcome o = runMethod(m, gi.instance, cfg);
        f << n << "," << m << "," << seed << "," << o.workPerEvaluation << "," << o.evaluations
          << "," << o.work << "," << components(o) << "\n";
      }
    }
  }
  std::printf("evalcost done (%s)\n", tuned ? "tuned" : "reference");
}

void expConverge(int seeds, long long budget) {
  // Budget response: separate runs of the TUNED configuration of every method at a ladder of work
  // budgets, final quality recorded.  This is not a single-run best-so-far trace: each rung is its
  // own run, and a memetic method's initial-refinement share scales with its budget.
  auto tuned = requireTuned(budget);
  auto f = openCsv("converge.csv",
                   std::string("n,method,seed,work_budget,hard,soft,objective,work,") +
                       kLegalityHeader);
  for (int n : {800}) {
    const long long full = budget * n;
    const long long ladder[] = {full / 32, full / 16, full / 8, full / 4, full / 2, full,
                                2 * full, 4 * full};
    for (const std::string& m : methods()) {
      for (long long b : ladder) {
        for (int seed = 1; seed <= seeds; ++seed) {
          GeneratedInstance gi = makeInstance(n, seed);
          MethodConfig cfg = tuned.count(m) ? tuned[m] : MethodConfig{};
          cfg.workBudget = b;
          cfg.seed = 2024 + seed;
          Outcome o = runMethod(m, gi.instance, cfg);
          Score sc = validate(gi.instance, o.best);
          f << n << "," << m << "," << seed << "," << b << "," << sc.hard() << "," << sc.soft()
            << "," << sc.objective(kDefaultBeta) << "," << o.work << "," << legality(sc) << "\n";
        }
      }
    }
  }
  std::printf("converge done\n");
}

/// The memetic algorithm with its recombination operator disabled (a crossover-disabled MA: parent
/// selection, copying, mutation, refinement and generational replacement are all kept).
///
/// Under two configurations: the tuned one, at which the memetic algorithm completes few
/// generations, and a small population with a cheap refinement chosen to complete as many
/// generations as the budget allows.
void expCrossover(int seeds, long long budget) {
  auto f = openCsv("crossover.csv",
                   std::string("n,seed,config,arm,hard,soft,objective,budget,work,generations,"
                               "best_generation,offspring,") +
                       kLegalityHeader + "," + kComponentHeader);
  const char* arms[] = {"ma", "ma-nox"};
  const auto tuned = requireTuned(budget);
  for (int n : {400, 1600}) {
    for (int variant = 0; variant < 2; ++variant) {
      for (const char* a : arms) {
        for (int seed = 1; seed <= seeds; ++seed) {
          GeneratedInstance gi = makeInstance(n, seed);
          MethodConfig cfg;
          if (variant == 0) {
            auto it = tuned.find("ma");
            if (it != tuned.end()) cfg = it->second;
          } else {
            cfg.population = 12;
            cfg.tournament = 2;
            cfg.elite = 1;
            cfg.crossoverRate = 0.9;
            cfg.mutationRate = 0.02;
            cfg.lsProbability = 0.25;
            cfg.lsCandidates = 1000;
            cfg.lsHistory = 50;
            cfg.lsEngine = "lahc";
          }
          cfg.workBudget = budget * n;
          cfg.seed = 9090 + seed;
          Outcome o = runMethod(a, gi.instance, cfg);
          Score sc = validate(gi.instance, o.best);
          f << n << "," << seed << "," << (variant == 0 ? "tuned" : "generational") << "," << a
            << "," << sc.hard() << "," << sc.soft() << "," << sc.objective(kDefaultBeta) << ","
            << cfg.workBudget << "," << o.work << "," << o.generations << "," << o.bestGeneration
            << "," << o.evaluations << "," << legality(sc) << "," << components(o) << "\n";
        }
      }
    }
  }
  std::printf("crossover done\n");
}

/// Equal wall-clock time, measured directly.  On each instance the tuned local search is first run
/// under the work budget of the main comparison and its elapsed time T is recorded; every method,
/// at its tuned configuration, is then run on the same instance with no work budget and a wall-clock
/// limit of T.  Elapsed time is machine-dependent, so this experiment is reproducible in design but
/// not byte for byte; it is run alone, single-threaded, on an otherwise idle machine.
void expTimed(int seeds, long long budget) {
  auto tuned = requireTuned(budget);
  auto f = openCsv("timed.csv",
                   std::string("n,seed,method,time_limit_ms,hard,soft,objective,work,elapsed_ms,"
                               "evaluations,generations,") +
                       kLegalityHeader);
  for (int n : {800, 1600, 3200}) {
    if (gMaxN > 0 && n > gMaxN) continue;
    if (n < gMinN) continue;
    for (int seed = gSeedStart; seed <= seeds; ++seed) {
      GeneratedInstance gi = makeInstance(n, seed);
      int64_t limitMs = 0;
      {
        MethodConfig cfg = tuned.count("sls") ? tuned["sls"] : MethodConfig{};
        cfg.workBudget = budget * n;
        cfg.seed = 707070 + seed;
        limitMs = std::max<int64_t>(1, runMethod("sls", gi.instance, cfg).elapsedMs);
      }
      for (const std::string& m : methods()) {
        MethodConfig cfg = tuned.count(m) ? tuned[m] : MethodConfig{};
        // No work stopping rule: a budget large enough never to bind.  For a memetic method the
        // nominal work budget still sizes the initial refinement, so it is kept at the main
        // comparison's value and the stopping rule is switched to time by an unreachable
        // offspring budget.
        cfg.workBudget = budget * n;
        if (m == "ga" || m == "ma" || m == "ma-adaptive") cfg.evalBudget = (1LL << 62);
        else cfg.workBudget = (1LL << 62);
        cfg.timeLimitMs = limitMs;
        cfg.seed = 707070 + seed;
        Outcome o = runMethod(m, gi.instance, cfg);
        Score sc = validate(gi.instance, o.best);
        f << n << "," << seed << "," << m << "," << limitMs << "," << sc.hard() << ","
          << sc.soft() << "," << sc.objective(kDefaultBeta) << "," << o.work << "," << o.elapsedMs
          << "," << o.evaluations << "," << o.generations << "," << legality(sc) << "\n";
      }
    }
  }
  std::printf("timed done\n");
}

/// Every (size, seed) pair any experiment generates, so that data/ holds every instance the
/// article's runs used:
///   sizes 200..3200, seeds 1..20  -- main comparison (1..10), legality (1..20 at 400, 1600),
///                                    currency, cost probe, budget response, crossover, equal time
///   sizes 400, 1600, seeds 101..105 -- configuration search
///   size 6400, seeds 1..6          -- cost probe
std::vector<std::pair<int, int>> instancesUsed() {
  std::vector<std::pair<int, int>> out;
  for (int n : {200, 400, 800, 1600, 3200})
    for (int seed = 1; seed <= 20; ++seed) out.emplace_back(n, seed);
  for (int n : {400, 1600})
    for (int seed = 101; seed <= 105; ++seed) out.emplace_back(n, seed);
  for (int seed = 1; seed <= 6; ++seed) out.emplace_back(6400, seed);
  return out;
}

void writeInstances(const std::string& dir) {
  std::error_code ec;
  std::filesystem::create_directories(dir, ec);
  int written = 0;
  for (const auto& [n, seed] : instancesUsed()) {
    GeneratedInstance gi = makeInstance(n, seed);
    char path[512];
    std::snprintf(path, sizeof(path), "%s/n%05d-s%d.txt", dir.c_str(), n, seed);
    if (!writeInstance(path, gi.instance)) die(std::string("cannot write ") + path);
    std::snprintf(path, sizeof(path), "%s/n%05d-s%d.planted.txt", dir.c_str(), n, seed);
    if (!writeSchedule(path, gi.planted)) die(std::string("cannot write ") + path);
    ++written;
  }
  std::printf("%d instances written to %s\n", written, dir.c_str());
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::printf("usage: experiments <agreement|tune|compare|budgetmode|evalcost|converge|crossover|timed|instances|selftest> "
                "[--seeds N] [--seedstart N] [--budget W] [--tuned] [--maxn N] [--timelimit MS] "
                "[--only m1,m2] [--results DIR] [--data DIR]\n");
    return 2;
  }
  const std::string cmd = argv[1];
  int seeds = 20;
  long long budget = 6000;   // work units per class
  std::string data = "data";
  bool tuned = false;
  for (int i = 2; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--tuned")) tuned = true;
    // --timelimit raises the per-run safety deadline of the currency experiment: an equal-EVALUATION
    // run of a memetic method can take longer than the default fifteen minutes, and a truncated run
    // would measure only a lower bound on the work it buys.  --maxn / --minn restrict the sizes.
    else if (!std::strcmp(argv[i], "--timelimit") && i + 1 < argc)
      gTimeLimitMs = std::atoll(argv[++i]);
    else if (!std::strcmp(argv[i], "--maxn") && i + 1 < argc) gMaxN = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--minn") && i + 1 < argc) gMinN = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--seedstart") && i + 1 < argc)
      gSeedStart = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--seeds") && i + 1 < argc) seeds = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--budget") && i + 1 < argc) budget = std::atoll(argv[++i]);
    else if (!std::strcmp(argv[i], "--maxbudget") && i + 1 < argc)
      tt::setBudgetCap(std::atoll(argv[++i]));
    else if (!std::strcmp(argv[i], "--results") && i + 1 < argc) gResults = argv[++i];
    else if (!std::strcmp(argv[i], "--only") && i + 1 < argc) {
      std::stringstream ss(argv[++i]);
      std::string m;
      while (std::getline(ss, m, ',')) gOnly.push_back(m);
    }
    else if (!std::strcmp(argv[i], "--data") && i + 1 < argc) data = argv[++i];
    else die(std::string("unknown or incomplete option '") + argv[i] + "'");
  }
  for (const std::string& m : gOnly)
    if (std::find(kAllMethods.begin(), kAllMethods.end(), m) == kAllMethods.end())
      die("--only: unknown method '" + m + "'");
  if (cmd == "agreement") expAgreement(seeds, budget);
  else if (cmd == "tune") expTune(seeds, budget);
  else if (cmd == "compare") expCompare(seeds, budget);
  else if (cmd == "evalcost") expEvalCost(seeds, tuned, budget);
  else if (cmd == "budgetmode") expBudgetMode(seeds, tuned, budget);
  else if (cmd == "converge") expConverge(seeds, budget);
  else if (cmd == "crossover") expCrossover(seeds, budget);
  else if (cmd == "timed") expTimed(seeds, budget);
  else if (cmd == "instances") writeInstances(data);
  else if (cmd == "selftest") return runSelfTest(gResults);
  else {
    std::printf("unknown subcommand %s\n", cmd.c_str());
    return 2;
  }
  return 0;
}
