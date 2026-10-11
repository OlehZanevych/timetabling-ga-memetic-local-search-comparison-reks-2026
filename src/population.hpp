// Genetic and memetic algorithms over the same instances and the same objective as the local
// search they are compared against.
//
// **One reported objective.**  Every method here scores a schedule with the same `State`, hence the
// same nine penalty counts and the same aggregation, and every reported result is re-checked by the
// independent validator afterwards.  The methods' *internal* orderings differ and are documented
// where they are defined: the population methods order individuals by (illegal, unplaced, H, f),
// the trajectory methods accept on a weighted surrogate.
//
// **One budget, in one counted unit.**  Both families are charged in bucket recomputations of the
// shared incremental evaluator -- a deterministic proxy for evaluator work, not a measure of all
// computation.  Everything a population method does through an evaluator is charged: the greedy
// constructions of the initial population, the scoring of every individual, and every refinement.
// Nested refinements are capped at the work that remains, so the overshoot past the budget is at
// most one refinement step plus the scoring of one child, and the actual work is reported.
#pragma once

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "core/rng.hpp"
#include "core/search.hpp"
#include "core/state.hpp"

namespace ga {

using namespace tt;

struct Individual {
  std::vector<Placement> genes;
  double objective = 1e300;
  double hard = 1e300;
  int unplaced = 0;
  /// Breaches of the two placement-time rules -- room availability and the per-(entity, day) load
  /// cap -- which the local search enforces as filters and recombination and mutation do not.  They
  /// are the first key of the ordering, so an illegal individual never displaces a legal one and the
  /// returned schedule is legal whenever generation zero is (a greedy construction always is).
  int illegal = 0;
  bool better(const Individual& o) const {
    if (illegal != o.illegal) return illegal < o.illegal;
    if (unplaced != o.unplaced) return unplaced < o.unplaced;
    if (hard != o.hard) return hard < o.hard;
    return objective < o.objective;
  }
};

struct GaOptions {
  int population = 30;
  int tournament = 3;
  int elite = 2;
  double crossoverRate = 0.9;
  double mutationRate = 0.03;    ///< per-gene probability of a random relocation
  /// Unit of inheritance.  Per-gene crossover of two schedules destroys the structure that makes a
  /// schedule good -- classes sharing a lecturer or a group have to move together or the child is
  /// just both parents' conflicts at once -- so the default inherits whole communities of the
  /// class-entity graph.  Per-gene crossover is kept as an arm of the comparison.
  bool clusterCrossover = true;
  bool greedyInit = true;        ///< seed the whole population with randomised greedy constructions
  bool localSearch = false;      ///< memetic: refine offspring with the local search
  double lsProbability = 0.3;    ///< share of offspring refined
  bool adaptiveLs = false;       ///< raise the share quickly on success, lower it slowly otherwise
  long long lsCandidates = 2000; ///< candidate limit of one refinement (also capped by remaining work)
  std::string lsEngine = "lahc"; ///< acceptance rule of the refinement: lahc | dlas
  int lsHistory = 50;            ///< its acceptance history length
  /// The work budget.  Under an offspring budget it is not a stopping rule but still sizes the
  /// initial refinement (kInitialRefinementShare of it), so that generation zero is refined the
  /// same way under both currencies.
  long long workBudget = 20000000;
  /// An alternative budget in *offspring evaluations*, the currency the evolutionary-computation
  /// literature usually counts.  Zero leaves the work budget in charge.  Running the comparison
  /// under both currencies is the point of one of the experiments: they do not agree, and the
  /// factor between them is exactly the cost of evaluating a child.
  long long offspringBudget = 0;
  int64_t timeLimitMs = 3600000;
  uint64_t seed = 1;
  bool crossover = true;         ///< false isolates how much of a memetic result is the population
};

struct GaResult {
  Individual best;
  long long work = 0;            ///< total: the sum of the five components below
  long long constructionWork = 0;    ///< greedy constructions of the initial population
  long long initialScoringWork = 0;  ///< scoring generation zero (before and after refinement)
  long long initialRefineWork = 0;   ///< refining generation zero
  long long childScoringWork = 0;    ///< scoring offspring
  long long childRefineWork = 0;     ///< refining offspring
  long long refinedChildren = 0;     ///< offspring that were refined
  long long refineCandidates = 0;    ///< candidates evaluated inside all refinements
  long long generations = 0;
  long long offspring = 0;
  /// The generation in which the returned best was found: 0 means it came out of the refined
  /// initial population and no child ever improved on it, which is the thing a memetic algorithm
  /// has to beat before its population can be said to be doing anything.
  long long bestGeneration = 0;
  long long elapsedMs = 0;
  double workPerOffspring = 0;       ///< total work / offspring: the comparable outer-unit price
  double childScoringPerOffspring = 0;  ///< childScoringWork / offspring: the direct scoring cost
  double meanChangedGenes = 0;       ///< mean genes a child differs from the evaluator's state
  double meanParentDistance = 0;     ///< mean genes a (final) child differs from its first parent
  double finalLsProbability = 0;
};

/// Evaluates individuals against one shared state by applying only the genes that differ from the
/// schedule the state currently holds.  That is the fastest an honest population method can be, and
/// it still costs Theta(n) per child, which is the mechanism the paper measures.
class Evaluator {
 public:
  explicit Evaluator(const Instance& in) : in_(in), st_(in) {
    current_.assign(in.classes.size(), Placement{});
    for (int32_t c : in.movable) current_[c] = Placement{};
    overCap_.assign(static_cast<size_t>(in.nEntities()) * (in.nDays + 1), 0);
    unavailable_.assign(in.classes.size(), 0);
  }

  /// Score an individual; returns the number of genes that differed from the evaluator's state
  /// (the Hamming distance h of the proposition).  The scan that finds them is O(n) comparisons,
  /// which the work counter does not charge; only the bucket recomputations are counted.
  int score(Individual& ind) {
    touched_.clear();
    int changed = 0;
    for (int32_t c : in_.movable)
      if (!(ind.genes[c] == current_[c])) {
        ++changed;
        noteBuckets(c, current_[c]);
        st_.place(c, ind.genes[c]);
        current_[c] = ind.genes[c];
        noteBuckets(c, current_[c]);
        const Placement& p = current_[c];
        const char u = (p.placed() && p.room >= 0 && !in_.roomFree(p.room, p.day, p.time)) ? 1 : 0;
        unavailableCount_ += u - unavailable_[c];
        unavailable_[c] = u;
      }
    st_.flush();
    for (int b : touched_) {
      const int e = b / (in_.nDays + 1), d = b % (in_.nDays + 1);
      const int cap = in_.maxPerDay[e];
      const char o = (!in_.isAbstractEntity(e) && cap < (1 << 20) &&
                      static_cast<int>(st_.members(e, d).size()) > cap) ? 1 : 0;
      overCapCount_ += o - overCap_[b];
      overCap_[b] = o;
    }
    ind.objective = st_.objective();
    ind.hard = st_.hard();
    ind.unplaced = st_.unplaced();
    ind.illegal = overCapCount_ + unavailableCount_;
    return changed;
  }

  int64_t work() const { return st_.work(); }
  const State& state() const { return st_; }

 private:
  void noteBuckets(int c, const Placement& p) {
    if (!p.placed()) return;
    st_.entitiesOf(c, p, ents_);
    for (int32_t e : ents_) touched_.push_back(st_.bucketIndex(e, p.day));
  }

  const Instance& in_;
  State st_;
  std::vector<Placement> current_;
  std::vector<char> overCap_, unavailable_;
  int overCapCount_ = 0, unavailableCount_ = 0;
  std::vector<int> touched_;
  std::vector<int32_t> ents_;
};

/// A random admissible placement for one class, ignoring everything the rest of the schedule does.
inline Placement randomGene(const Instance& in, int c, Rng& rng) {
  const ClassSpec& cs = in.classes[c];
  if (cs.slots.empty()) return Placement{};
  const Slot& s = cs.slots[rng.below(static_cast<uint32_t>(cs.slots.size()))];
  const int r = cs.rooms.empty() ? -1
                                 : cs.rooms[rng.below(static_cast<uint32_t>(cs.rooms.size()))];
  return Placement{static_cast<int8_t>(s.day), static_cast<int16_t>(s.time), s.parity, r};
}

inline std::vector<Placement> randomSchedule(const Instance& in, Rng& rng) {
  std::vector<Placement> g(in.classes.size(), Placement{});
  for (size_t i = 0; i < in.classes.size(); ++i)
    g[i] = in.classes[i].movable ? randomGene(in, static_cast<int>(i), rng)
                                 : in.classes[i].fixed;
  return g;
}

GaResult runGa(const Instance& in, const GaOptions& opt);

}  // namespace ga
