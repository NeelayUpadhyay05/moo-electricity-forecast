import numpy as np


class MOOOptimizer:
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

        # SBX crossover parameters
        self.p_cross = p_cross
        self.eta_c = eta_c

        # Polynomial mutation parameters
        self.eta_m = eta_m
        self.mutation_rate = 1.0 / len(bounds)

        self.population = None
        self.objectives = None

    # -----------------------------
    # Initialization
    # -----------------------------
    def initialize_population(self):
        pop = []
        for _ in range(self.pop_size):
            individual = [
                self.rng.uniform(low, high)
                for (low, high) in self.bounds
            ]
            pop.append(individual)
        return np.array(pop)

    # -----------------------------
    # Evaluation
    # -----------------------------
    def evaluate_population(self, population):
        objectives = []
        for individual in population:
            obj = self.fitness_fn(individual)
            objectives.append(obj)
        return np.array(objectives)

    # -----------------------------
    # Non-dominated Sorting
    # -----------------------------
    def non_dominated_sort(self, objectives):
        num = len(objectives)
        domination_counts = np.zeros(num)
        dominated_solutions = [[] for _ in range(num)]
        fronts = [[]]

        for i in range(num):
            for j in range(num):
                if i == j:
                    continue

                if self.dominates(objectives[i], objectives[j]):
                    dominated_solutions[i].append(j)
                elif self.dominates(objectives[j], objectives[i]):
                    domination_counts[i] += 1

            if domination_counts[i] == 0:
                fronts[0].append(i)

        current_front = 0
        while fronts[current_front]:
            next_front = []
            for i in fronts[current_front]:
                for j in dominated_solutions[i]:
                    domination_counts[j] -= 1
                    if domination_counts[j] == 0:
                        next_front.append(j)
            current_front += 1
            fronts.append(next_front)

        return fronts[:-1]

    @staticmethod
    def dominates(obj1, obj2):
        return (
            (obj1[0] <= obj2[0] and obj1[1] <= obj2[1])
            and (obj1[0] < obj2[0] or obj1[1] < obj2[1])
        )

    # -----------------------------
    # Crowding Distance
    # -----------------------------
    def crowding_distance(self, front, objectives):
        distance = np.zeros(len(front))
        front_objs = objectives[front]

        for m in range(front_objs.shape[1]):
            sorted_idx = np.argsort(front_objs[:, m])
            distance[sorted_idx[0]] = distance[sorted_idx[-1]] = np.inf

            obj_values = front_objs[sorted_idx, m]
            norm = obj_values[-1] - obj_values[0]
            if norm == 0:
                continue

            for i in range(1, len(front) - 1):
                distance[sorted_idx[i]] += (
                    obj_values[i + 1] - obj_values[i - 1]
                ) / norm

        return distance

    # -----------------------------
    # Tournament Selection (with front rank + crowding distance)
    # -----------------------------
    def tournament_selection(self, population, ranks, crowding_dists):
        idx1, idx2 = self.rng.choice(len(population), size=2, replace=False)

        # Prefer lower front rank
        if ranks[idx1] < ranks[idx2]:
            return population[idx1]
        if ranks[idx2] < ranks[idx1]:
            return population[idx2]

        # Same front rank — prefer higher crowding distance
        if crowding_dists[idx1] > crowding_dists[idx2]:
            return population[idx1]
        if crowding_dists[idx2] > crowding_dists[idx1]:
            return population[idx2]

        return population[idx1] if self.rng.random() < 0.5 else population[idx2]

    # -----------------------------
    # SBX Crossover
    # -----------------------------
    def crossover(self, parent1, parent2):
        child1 = parent1.copy()
        child2 = parent2.copy()

        if self.rng.random() > self.p_cross:
            return child1, child2

        for i in range(len(parent1)):
            if self.rng.random() > 0.5:
                continue

            p1, p2 = parent1[i], parent2[i]
            if abs(p1 - p2) < 1e-14:
                continue

            low, high = self.bounds[i]
            if p1 > p2:
                p1, p2 = p2, p1

            # Compute beta
            beta = 1.0 + 2.0 * min(p1 - low, high - p2) / (p2 - p1)
            alpha = 2.0 - beta ** (-(self.eta_c + 1.0))

            u = self.rng.random()
            if u <= 1.0 / alpha:
                betaq = (u * alpha) ** (1.0 / (self.eta_c + 1.0))
            else:
                betaq = (1.0 / (2.0 - u * alpha)) ** (1.0 / (self.eta_c + 1.0))

            c1 = 0.5 * ((p1 + p2) - betaq * (p2 - p1))
            c2 = 0.5 * ((p1 + p2) + betaq * (p2 - p1))

            child1[i] = np.clip(c1, low, high)
            child2[i] = np.clip(c2, low, high)

        return child1, child2

    # -----------------------------
    # Polynomial Mutation
    # -----------------------------
    def polynomial_mutate(self, individual):
        mutated = individual.copy()
        for i in range(len(mutated)):
            if self.rng.random() >= self.mutation_rate:
                continue

            low, high = self.bounds[i]
            delta = high - low
            val = mutated[i]

            delta1 = (val - low) / delta
            delta2 = (high - val) / delta

            u = self.rng.random()
            if u < 0.5:
                xy = 1.0 - delta1
                val_mut = (2.0 * u + (1.0 - 2.0 * u) * xy ** (self.eta_m + 1.0))
                deltaq = val_mut ** (1.0 / (self.eta_m + 1.0)) - 1.0
            else:
                xy = 1.0 - delta2
                val_mut = 2.0 * (1.0 - u) + 2.0 * (u - 0.5) * xy ** (self.eta_m + 1.0)
                deltaq = 1.0 - val_mut ** (1.0 / (self.eta_m + 1.0))

            mutated[i] = np.clip(val + deltaq * delta, low, high)

        return mutated

    # -----------------------------
    # Compute front ranks and crowding distances for population
    # -----------------------------
    def compute_ranks_and_crowding(self, objectives):
        fronts = self.non_dominated_sort(objectives)
        n = len(objectives)
        ranks = np.zeros(n, dtype=int)
        crowding_dists = np.zeros(n)

        for rank, front in enumerate(fronts):
            for idx in front:
                ranks[idx] = rank
            distances = self.crowding_distance(front, objectives)
            for i, idx in enumerate(front):
                crowding_dists[idx] = distances[i]

        return fronts, ranks, crowding_dists

    # -----------------------------
    # Main Optimization Loop
    # -----------------------------
    def optimize(self):
        print("\n================ MOO Optimization Started ================")
        print(f"Population Size: {self.pop_size}")
        print(f"Generations: {self.generations}")
        print(f"Crossover: SBX (p={self.p_cross}, eta_c={self.eta_c})")
        print(f"Mutation: Polynomial (rate={self.mutation_rate:.2f}, eta_m={self.eta_m})")
        print("==========================================================")

        population = self.initialize_population()

        print("\nEvaluating Initial Population...")
        objectives_list = []
        history = []
        best_so_far = float("inf")
        for individual in population:
            obj = self.fitness_fn(individual)
            objectives_list.append(obj)
            best_so_far = min(best_so_far, obj[0])
            history.append(best_so_far)
        objectives = np.array(objectives_list)
        total_evals = self.pop_size

        fronts, ranks, crowding_dists = self.compute_ranks_and_crowding(objectives)
        print(f"Initial Pareto Front Size: {len(fronts[0])}")

        # ==============================
        # Main Generational Loop
        # ==============================
        for gen in range(self.generations):

            print(f"\n########## Generation {gen+1}/{self.generations} ##########")

            # Generate offspring via crossover + mutation
            offspring = []
            while len(offspring) < self.pop_size:
                p1 = self.tournament_selection(population, ranks, crowding_dists)
                p2 = self.tournament_selection(population, ranks, crowding_dists)
                c1, c2 = self.crossover(p1, p2)
                c1 = self.polynomial_mutate(c1)
                c2 = self.polynomial_mutate(c2)
                offspring.append(c1)
                if len(offspring) < self.pop_size:
                    offspring.append(c2)

            offspring = np.array(offspring)

            print("Evaluating Offspring...")
            offspring_objectives_list = []
            for individual in offspring:
                obj = self.fitness_fn(individual)
                offspring_objectives_list.append(obj)
                best_so_far = min(best_so_far, obj[0])
                history.append(best_so_far)
            offspring_objectives = np.array(offspring_objectives_list)
            total_evals += self.pop_size

            # Combine parents + offspring
            combined_pop = np.vstack((population, offspring))
            combined_obj = np.vstack((objectives, offspring_objectives))

            fronts = self.non_dominated_sort(combined_obj)

            new_population = []
            new_objectives = []

            for front in fronts:
                if len(new_population) + len(front) > self.pop_size:
                    distances = self.crowding_distance(front, combined_obj)
                    sorted_front = [
                        front[i]
                        for i in np.argsort(-distances)
                    ]
                    remaining = self.pop_size - len(new_population)
                    selected = sorted_front[:remaining]
                else:
                    selected = front

                for idx in selected:
                    new_population.append(combined_pop[idx])
                    new_objectives.append(combined_obj[idx])

                if len(new_population) >= self.pop_size:
                    break

            population = np.array(new_population)
            objectives = np.array(new_objectives)

            # Recompute ranks and crowding for next generation's tournament selection
            fronts, ranks, crowding_dists = self.compute_ranks_and_crowding(objectives)

            best_val = min(objectives[fronts[0]][:, 0])

            print(f"Generation {gen+1} Complete")
            print(f"Current Pareto Front Size: {len(fronts[0])}")
            print(f"Best Validation MSE in Front: {best_val:.6f}")
            print(f"Total Evaluations So Far: {total_evals}")

        # ==============================
        # Final Pareto Front
        # ==============================
        final_front = fronts[0]

        pareto_solutions = []
        for idx in final_front:
            pareto_solutions.append(
                {
                    "params": population[idx],
                    "val_mse": objectives[idx][0],
                    "complexity": objectives[idx][1],
                }
            )

        print("\n================ MOO Optimization Finished ================")
        print(f"Final Pareto Front Size: {len(final_front)}")
        print(f"Total Evaluations: {total_evals}")
        print("============================================================")

        return pareto_solutions, history
