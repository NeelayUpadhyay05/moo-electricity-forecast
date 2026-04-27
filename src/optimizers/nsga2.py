import numpy as np


class NSGA2:
    """A compact NSGA-II implementation compatible with the project's optimizer API.

    API:
      nsga = NSGA2(fitness_fn, bounds, pop_size, generations, seed)
      pareto_solutions, history = nsga.optimize()
    """

    def __init__(
        self,
        fitness_fn,
        bounds,
        pop_size=10,
        generations=3,
        seed=42,
        p_cross=0.9,
        eta_c=20,
        eta_m=20,
    ):
        self.fitness_fn = fitness_fn
        self.bounds = bounds
        self.pop_size = pop_size
        self.generations = generations
        self.rng = np.random.default_rng(seed)

        self.p_cross = p_cross
        self.eta_c = eta_c
        self.eta_m = eta_m
        self.mutation_rate = 1.0 / len(bounds)

    def initialize(self):
        pop = []
        for _ in range(self.pop_size):
            ind = [self.rng.uniform(low, high) for (low, high) in self.bounds]
            pop.append(ind)
        return np.array(pop)

    @staticmethod
    def dominates(a, b):
        # a and b are objective vectors (minimize)
        return ((a[0] <= b[0] and a[1] <= b[1]) and (a[0] < b[0] or a[1] < b[1]))

    def non_dominated_sort(self, objectives):
        n = len(objectives)
        domination_counts = np.zeros(n, dtype=int)
        dominated_sets = [[] for _ in range(n)]
        fronts = [[]]

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if self.dominates(objectives[i], objectives[j]):
                    dominated_sets[i].append(j)
                elif self.dominates(objectives[j], objectives[i]):
                    domination_counts[i] += 1
            if domination_counts[i] == 0:
                fronts[0].append(i)

        cur = 0
        while fronts[cur]:
            next_front = []
            for i in fronts[cur]:
                for j in dominated_sets[i]:
                    domination_counts[j] -= 1
                    if domination_counts[j] == 0:
                        next_front.append(j)
            cur += 1
            fronts.append(next_front)

        return fronts[:-1]

    def crowding_distance(self, front, objectives):
        if len(front) == 0:
            return np.array([])
        objs = objectives[front]
        m = objs.shape[1]
        distances = np.zeros(len(front))
        for k in range(m):
            idx = np.argsort(objs[:, k])
            distances[idx[0]] = distances[idx[-1]] = np.inf
            denom = objs[idx[-1], k] - objs[idx[0], k]
            if denom == 0:
                continue
            for i in range(1, len(front) - 1):
                distances[idx[i]] += (objs[idx[i + 1], k] - objs[idx[i - 1], k]) / denom
        return distances

    def crossover(self, p1, p2):
        if self.rng.random() > self.p_cross:
            return p1.copy(), p2.copy()
        child1 = p1.copy()
        child2 = p2.copy()
        for i in range(len(p1)):
            if self.rng.random() > 0.5:
                continue
            x1, x2 = p1[i], p2[i]
            if abs(x1 - x2) < 1e-14:
                continue
            low, high = self.bounds[i]
            if x1 > x2:
                x1, x2 = x2, x1
            beta = 1.0 + 2.0 * min(x1 - low, high - x2) / (x2 - x1)
            alpha = 2.0 - beta ** (-(self.eta_c + 1.0))
            u = self.rng.random()
            if u <= 1.0 / alpha:
                betaq = (u * alpha) ** (1.0 / (self.eta_c + 1.0))
            else:
                betaq = (1.0 / (2.0 - u * alpha)) ** (1.0 / (self.eta_c + 1.0))
            c1 = 0.5 * ((x1 + x2) - betaq * (x2 - x1))
            c2 = 0.5 * ((x1 + x2) + betaq * (x2 - x1))
            child1[i] = np.clip(c1, low, high)
            child2[i] = np.clip(c2, low, high)
        return child1, child2

    def mutate(self, x):
        y = x.copy()
        for i in range(len(y)):
            if self.rng.random() >= self.mutation_rate:
                continue
            low, high = self.bounds[i]
            delta = high - low
            u = self.rng.random()
            if u < 0.5:
                val = (2 * u + (1 - 2 * u) * (1 - (y[i]-low)/delta) ** (self.eta_m + 1)) ** (1/(self.eta_m + 1)) - 1
                y[i] = np.clip(y[i] + val * delta, low, high)
            else:
                val = 1 - (2 * (1 - u) + 2 * (u - 0.5) * (1 - (high - y[i]) / delta) ** (self.eta_m + 1)) ** (1/(self.eta_m + 1))
                y[i] = np.clip(y[i] + val * delta, low, high)
        return y

    def optimize(self):
        print("\n================ NSGA-II Optimization Started ================")
        population = self.initialize()
        objectives = np.array([self.fitness_fn(ind) for ind in population])
        history = []
        best_so_far = float("inf")
        for obj in objectives:
            best_so_far = min(best_so_far, obj[0])
            history.append(best_so_far)

        for gen in range(self.generations):
            print(f"\nGeneration {gen+1}/{self.generations}")
            # create offspring
            offspring = []
            while len(offspring) < self.pop_size:
                i1, i2 = self.rng.choice(len(population), size=2, replace=False)
                p1, p2 = population[i1], population[i2]
                c1, c2 = self.crossover(p1, p2)
                c1 = self.mutate(c1)
                c2 = self.mutate(c2)
                offspring.append(c1)
                if len(offspring) < self.pop_size:
                    offspring.append(c2)
            offspring = np.array(offspring)
            off_objs = np.array([self.fitness_fn(ind) for ind in offspring])
            for obj in off_objs:
                best_so_far = min(best_so_far, obj[0])
                history.append(best_so_far)

            # combine and select next generation
            combined = np.vstack((population, offspring))
            combined_objs = np.vstack((objectives, off_objs))
            fronts = self.non_dominated_sort(combined_objs)
            new_pop = []
            new_objs = []
            for front in fronts:
                if len(new_pop) + len(front) > self.pop_size:
                    distances = self.crowding_distance(front, combined_objs)
                    ranked = [front[i] for i in np.argsort(-distances)]
                    remaining = self.pop_size - len(new_pop)
                    selected = ranked[:remaining]
                else:
                    selected = front
                for idx in selected:
                    new_pop.append(combined[idx])
                    new_objs.append(combined_objs[idx])
                if len(new_pop) >= self.pop_size:
                    break
            population = np.array(new_pop)
            objectives = np.array(new_objs)

            print(f"Gen {gen+1} complete — current best val_mse: {min(objectives[:,0]):.6f}")

        # build pareto solutions
        fronts = self.non_dominated_sort(objectives)
        final_front = fronts[0]
        pareto = []
        for idx in final_front:
            pareto.append({
                "params": population[idx],
                "val_mse": float(objectives[idx][0]),
                "complexity": int(objectives[idx][1]),
            })

        print("\n================ NSGA-II Optimization Finished ================")
        return pareto, history
