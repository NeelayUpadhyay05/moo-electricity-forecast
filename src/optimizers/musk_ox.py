import numpy as np


class MuskOxOptimizer:
    """A bespoke Musk Ox multi-objective optimizer.

    This is a population-based MOEA that keeps an external archive of
    non-dominated solutions and uses gaussian perturbation + tournament
    selection for variation. API mirrors NSGA2 for interchangeability.
    """

    def __init__(
        self,
        fitness_fn,
        bounds,
        pop_size=10,
        generations=3,
        seed=42,
        sigma=0.1,
    ):
        self.fitness_fn = fitness_fn
        self.bounds = bounds
        self.pop_size = pop_size
        self.generations = generations
        self.rng = np.random.default_rng(seed)
        self.sigma = sigma

    def initialize(self):
        pop = []
        for _ in range(self.pop_size):
            ind = [self.rng.uniform(low, high) for (low, high) in self.bounds]
            pop.append(ind)
        return np.array(pop)

    @staticmethod
    def dominates(a, b):
        return ((a[0] <= b[0] and a[1] <= b[1]) and (a[0] < b[0] or a[1] < b[1]))

    def update_archive(self, archive, archive_objs, candidate, candidate_obj):
        # remove dominated archive members
        keep = []
        keep_objs = []
        dominated = False
        for a, ao in zip(archive, archive_objs):
            if self.dominates(ao, candidate_obj):
                dominated = True
                break
            if self.dominates(candidate_obj, ao):
                continue
            keep.append(a)
            keep_objs.append(ao)
        if not dominated:
            keep.append(candidate)
            keep_objs.append(candidate_obj)
        return np.array(keep), np.array(keep_objs)

    def perturb(self, parent):
        child = parent.copy()
        for i in range(len(child)):
            low, high = self.bounds[i]
            span = high - low
            child[i] += self.rng.normal(0, self.sigma * span)
            child[i] = np.clip(child[i], low, high)
        return child

    def tournament(self, population, objectives):
        i, j = self.rng.choice(len(population), size=2, replace=False)
        # prefer lower val_mse
        if objectives[i][0] < objectives[j][0]:
            return population[i]
        if objectives[j][0] < objectives[i][0]:
            return population[j]
        return population[i] if self.rng.random() < 0.5 else population[j]

    def optimize(self):
        print("\n================ Musk Ox Optimization Started ================")
        population = self.initialize()
        objectives = np.array([self.fitness_fn(ind) for ind in population])
        archive = population.copy()
        archive_objs = objectives.copy()
        history = []
        best = float("inf")
        for obj in objectives:
            best = min(best, obj[0])
            history.append(best)

        for gen in range(self.generations):
            print(f"\nGeneration {gen+1}/{self.generations}")
            offspring = []
            off_objs = []
            while len(offspring) < self.pop_size:
                parent = self.tournament(population, objectives)
                child = self.perturb(parent)
                obj = self.fitness_fn(child)
                offspring.append(child)
                off_objs.append(obj)
                # update archive
                archive, archive_objs = self.update_archive(archive, archive_objs, child, obj)
                best = min(best, obj[0])
                history.append(best)

            population = np.array(offspring)
            objectives = np.array(off_objs)

            print(f"Gen {gen+1} complete — archive size: {len(archive)} | best val_mse: {best:.6f}")

        # format pareto solutions from archive
        pareto = []
        for p, o in zip(archive, archive_objs):
            pareto.append({
                "params": p,
                "val_mse": float(o[0]),
                "complexity": int(o[1]),
            })

        print("\n================ Musk Ox Optimization Finished ================")
        return pareto, history
