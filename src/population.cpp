#include "population.hpp"

#include <chrono>

namespace ga {

/// The share of the work budget a memetic algorithm may spend refining generation zero.  Above this
/// the generational loop stops running at all; the value is a design decision of this comparison and
/// is stated in the paper.  It is enforced in work, per individual: each member of generation zero
/// may be refined for at most lsCandidates candidates and at most share * W / population work.
constexpr double kInitialRefinementShare = 0.2;

using Clock = std::chrono::steady_clock;

namespace {

/// The refinement: a bounded local search with the cheap neighbourhoods only and no escape.
SearchOptions refinementOptions(const GaOptions& opt, const std::vector<Placement>* start,
                                uint64_t seed, long long candidateLimit, long long workLimit,
                                int64_t timeLimitMs) {
  SearchOptions so;
  so.timeLimitMs = std::max<int64_t>(1, timeLimitMs);
  so.candidateLimit = std::max<long long>(1, candidateLimit);
  so.workLimit = std::max<long long>(0, workLimit);   // 0 = no work limit (offspring currency)
  so.threads = 1;
  so.seed = seed;
  so.warmStart = start;
  so.useDeep = so.useKick = so.useFresh = false;
  so.useLns = so.useRepack = so.useKopt = false;
  so.cooperate = false;
  so.engine = opt.lsEngine;
  so.lahcLength = opt.lsHistory;
  return so;
}

}  // namespace

GaResult runGa(const Instance& in, const GaOptions& opt) {
  Rng rng(opt.seed);
  Evaluator ev(in);
  const auto started = Clock::now();
  const auto deadline = started + std::chrono::milliseconds(opt.timeLimitMs);
  auto msLeft = [&]() {
    return static_cast<int64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(deadline - Clock::now()).count());
  };
  const bool workCurrency = opt.offspringBudget <= 0;

  GaResult out;
  long long refineWork = 0;   // initial + child refinement
  auto totalWork = [&]() { return out.constructionWork + ev.work() + refineWork; };
  // Under the work currency a nested refinement may spend only what is left of the budget.
  auto remaining = [&]() { return opt.workBudget - totalWork(); };

  // ---- the unit of inheritance -----------------------------------------------------------------
  std::vector<int32_t> cluster = buildClusters(in, opt.seed ^ 0x2545F4914F6CDD1DULL);
  int nClusters = 1;
  for (int32_t l : cluster) nClusters = std::max(nClusters, l + 1);

  // ---- the initial population ------------------------------------------------------------------
  // Randomised most-constrained-first constructions.  Each construction runs in its own evaluator
  // state inside solve(), and that work is charged: a trajectory method pays for its construction
  // in its own counter, so a population method has to pay for its mu constructions in the same way.
  std::vector<Individual> pop(opt.population);
  for (int i = 0; i < opt.population; ++i) {
    if (opt.greedyInit) {
      SearchOptions so;
      so.timeLimitMs = 60000;
      so.candidateLimit = 1;   // construction only (and one candidate, which is charged too)
      so.threads = 1;
      so.seed = opt.seed + 1000 * i + 7;
      SearchResult r = solve(in, so);
      pop[i].genes = r.best;
      out.constructionWork += r.workUnits;
    } else {
      pop[i].genes = randomSchedule(in, rng);
    }
    ev.score(pop[i]);
  }

  long long initialRefineWork = 0;
  if (opt.localSearch) {
    const long long share = static_cast<long long>(kInitialRefinementShare * opt.workBudget);
    const long long perIndividual = std::max<long long>(1, share / std::max(1, opt.population));
    for (int i = 0; i < opt.population; ++i) {
      long long limit = perIndividual;
      if (workCurrency) limit = std::min(limit, remaining());
      if (limit <= 0 || msLeft() <= 0) break;
      SearchOptions so = refinementOptions(opt, &pop[i].genes, opt.seed + 31 * i + 3,
                                           opt.lsCandidates, limit, msLeft());
      SearchResult r = solve(in, so);
      pop[i].genes = r.best;
      refineWork += r.workUnits;
      initialRefineWork += r.workUnits;
      out.refineCandidates += r.candidates;
      ev.score(pop[i]);
    }
  }
  out.initialRefineWork = initialRefineWork;
  out.initialScoringWork = ev.work();

  out.best = *std::min_element(pop.begin(), pop.end(),
                               [](const Individual& a, const Individual& b) { return a.better(b); });

  double lsProb = opt.lsProbability;
  long long sinceImprovement = 0;
  long long changedGenes = 0, parentDistance = 0;
  auto exhausted = [&]() {
    return workCurrency ? totalWork() >= opt.workBudget : out.offspring >= opt.offspringBudget;
  };

  auto tournament = [&]() -> const Individual& {
    const Individual* best = &pop[rng.below(static_cast<uint32_t>(pop.size()))];
    for (int i = 1; i < opt.tournament; ++i) {
      const Individual& c = pop[rng.below(static_cast<uint32_t>(pop.size()))];
      if (c.better(*best)) best = &c;
    }
    return *best;
  };

  std::vector<Individual> next;
  while (!exhausted() && msLeft() > 0) {
    next.clear();
    // Elitism.
    std::vector<const Individual*> sorted;
    for (const Individual& p : pop) sorted.push_back(&p);
    std::sort(sorted.begin(), sorted.end(),
              [](const Individual* a, const Individual* b) { return a->better(*b); });
    for (int i = 0; i < opt.elite && i < static_cast<int>(sorted.size()); ++i)
      next.push_back(*sorted[i]);

    // The generation may end part-way: the budget is checked before every child, so the run stops
    // within one child of the budget rather than at the end of a generation.  A partial generation
    // is still counted, and the incumbent is taken over it.
    bool partial = false;
    while (static_cast<int>(next.size()) < opt.population) {
      if (exhausted() || msLeft() <= 0) {
        partial = true;
        break;
      }
      Individual child;
      const Individual& a = tournament();
      if (opt.crossover && rng.coin(opt.crossoverRate)) {
        const Individual& b = tournament();
        child.genes = a.genes;
        if (opt.clusterCrossover) {
          // Inherit a random half of the communities from the second parent.
          std::vector<char> take(nClusters, 0);
          for (int k = 0; k < nClusters; ++k) take[k] = rng.coin(0.5) ? 1 : 0;
          for (int32_t g : in.movable)
            if (take[cluster[g]]) child.genes[g] = b.genes[g];
        } else {
          for (int32_t g : in.movable)
            if (rng.coin(0.5)) child.genes[g] = b.genes[g];
        }
      } else {
        child.genes = a.genes;
      }
      for (int32_t g : in.movable)
        if (rng.coin(opt.mutationRate)) child.genes[g] = randomGene(in, g, rng);

      if (opt.localSearch && rng.coin(lsProb)) {
        const long long limit = workCurrency ? remaining() : 0;
        if (!workCurrency || limit > 0) {
          SearchOptions so = refinementOptions(opt, &child.genes,
                                               opt.seed + 7919 * (out.offspring + 1),
                                               opt.lsCandidates, limit, msLeft());
          SearchResult r = solve(in, so);
          child.genes = r.best;
          refineWork += r.workUnits;
          out.childRefineWork += r.workUnits;
          out.refineCandidates += r.candidates;
          ++out.refinedChildren;
        }
      }

      for (int32_t g : in.movable)
        if (!(child.genes[g] == a.genes[g])) ++parentDistance;
      const long long before = ev.work();
      changedGenes += ev.score(child);
      out.childScoringWork += ev.work() - before;
      ++out.offspring;
      next.push_back(child);
    }
    if (partial) {
      // Fill the rest of the generation from the old one so that the incumbent update below sees
      // every child that was scored; no further work is spent.
      for (size_t i = 0; static_cast<int>(next.size()) < opt.population && i < sorted.size(); ++i)
        next.push_back(*sorted[i]);
    }
    pop.swap(next);
    ++out.generations;

    const Individual& gen = *std::min_element(
        pop.begin(), pop.end(),
        [](const Individual& a, const Individual& b) { return a.better(b); });
    if (gen.better(out.best)) {
      out.best = gen;
      out.bestGeneration = out.generations;
      sinceImprovement = 0;
      if (opt.adaptiveLs) lsProb = std::min(1.0, lsProb + 0.02);
    } else {
      ++sinceImprovement;
      if (opt.adaptiveLs && sinceImprovement % 5 == 0) lsProb = std::max(0.15, lsProb - 0.005);
    }
  }

  out.work = totalWork();
  out.elapsedMs =
      std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - started).count();
  out.workPerOffspring = out.offspring ? static_cast<double>(out.work) / out.offspring : 0.0;
  out.childScoringPerOffspring =
      out.offspring ? static_cast<double>(out.childScoringWork) / out.offspring : 0.0;
  out.meanChangedGenes =
      out.offspring ? static_cast<double>(changedGenes) / out.offspring : 0.0;
  out.meanParentDistance =
      out.offspring ? static_cast<double>(parentDistance) / out.offspring : 0.0;
  out.finalLsProbability = lsProb;
  return out;
}

}  // namespace ga
