import numpy as np


class MOOOptimizer:
    def __init__(
        self,
        fitness_fn,
        bounds,
        pop_size=10,
        generations=3,
        seed=42,
    ):
        self.fitness_fn = fitness_fn
        self.bounds = bounds
        self.pop_size = pop_size
        self.generations = generations
        self.rng = np.random.default_rng(seed)

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
    # Selection
    # -----------------------------
    def tournament_selection(self, population, objectives):
        idx1, idx2 = self.rng.choice(len(population), size=2, replace=False)

        if self.dominates(objectives[idx1], objectives[idx2]):
            return population[idx1]
        if self.dominates(objectives[idx2], objectives[idx1]):
            return population[idx2]

        return population[idx1] if self.rng.random() < 0.5 else population[idx2]

    # -----------------------------
    # Mutation
    # -----------------------------
    def mutate(self, individual, mutation_rate=0.2):
        mutated = individual.copy()
        for i in range(len(mutated)):
            if self.rng.random() < mutation_rate:
                low, high = self.bounds[i]
                sigma = 0.1 * (high - low)
                mutated[i] += self.rng.normal(0, sigma)
                mutated[i] = np.clip(mutated[i], low, high)
        return mutated

    # -----------------------------
    # Main Optimization Loop
    # -----------------------------
    def optimize(self):
        print("\n================ MOO Optimization Started ================")
        print(f"Population Size: {self.pop_size}")
        print(f"Generations: {self.generations}")
        print("==========================================================")

        population = self.initialize_population()

        print("\nEvaluating Initial Population...")
        objectives = self.evaluate_population(population)
        total_evals = self.pop_size

        fronts = self.non_dominated_sort(objectives)
        print(f"Initial Pareto Front Size: {len(fronts[0])}")

        history = [float(min(objectives[:, 0]))]

        # ==============================
        # Main Generational Loop
        # ==============================
        for gen in range(self.generations):

            print(f"\n########## Generation {gen+1}/{self.generations} ##########")

            offspring = []

            while len(offspring) < self.pop_size:
                parent = self.tournament_selection(population, objectives)
                child = self.mutate(parent)
                offspring.append(child)

            offspring = np.array(offspring)

            print("Evaluating Offspring...")
            offspring_objectives = self.evaluate_population(offspring)
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

            current_front = self.non_dominated_sort(objectives)[0]
            best_val = min(objectives[current_front][:, 0])

            print(f"Generation {gen+1} Complete")
            print(f"Current Pareto Front Size: {len(current_front)}")
            print(f"Best Validation MSE in Front: {best_val:.6f}")
            print(f"Total Evaluations So Far: {total_evals}")
            history.append(float(best_val))

        # ==============================
        # Final Pareto Front
        # ==============================
        final_front = self.non_dominated_sort(objectives)[0]

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