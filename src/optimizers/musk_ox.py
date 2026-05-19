import numpy as np

class MuskOxOptimizer:
    """Multi-objective Musk Ox Optimizer (MOO).

    Models the migratory, foraging, and defensive phases, with a Pareto
    archive representing herd guards (alpha leaders).
    """

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

    def initialize(self):
        pop = []
        for _ in range(self.pop_size):
            ind = [self.rng.uniform(low, high) for (low, high) in self.bounds]
            pop.append(ind)
        return np.array(pop)

    @staticmethod
    def dominates(a, b):
        return ((a[0] <= b[0] and a[1] <= b[1]) and (a[0] < b[0] or a[1] < b[1]))

    def get_guards(self, population, objectives):
        """Pick non-dominated solutions to act as herd guards."""
        is_guard = np.ones(len(population), dtype=bool)
        for i in range(len(population)):
            for j in range(len(population)):
                if i != j and self.dominates(objectives[j], objectives[i]):
                    is_guard[i] = False
                    break
        
        # Fallback if no non-dominated solutions are found.
        if not np.any(is_guard):
            is_guard[np.argmin(objectives[:, 0])] = True
            
        guards = population[is_guard]
        guard_objs = objectives[is_guard]
        return guards, guard_objs

    def update_archive(self, archive, archive_objs, candidate, candidate_obj):
        dominated = False
        keep = []
        keep_objs = []
        for a, ao in zip(archive, archive_objs):
            if self.dominates(ao, candidate_obj):
                dominated = True
                keep.append(a)
                keep_objs.append(ao)
            elif self.dominates(candidate_obj, ao):
                continue
            else:
                keep.append(a)
                keep_objs.append(ao)
        if not dominated:
            keep.append(candidate)
            keep_objs.append(candidate_obj)
        return np.array(keep), np.array(keep_objs)

    def optimize(self):
        print("\n================ Musk Ox Optimization Started ================")
        population = self.initialize()
        objectives = np.array([self.fitness_fn(ind) for ind in population])
        
        archive = population.copy()
        archive_objs = objectives.copy()
        history = []
        best = np.min(objectives[:, 0])
        history.append(best)

        for gen in range(self.generations):
            print(f"\nGeneration {gen+1}/{self.generations}")
            
            # Identify the guards (Pareto front).
            guards, guard_objs = self.get_guards(population, objectives)
            
            # Identify the worst solution as a predator proxy for the defensive phase.
            # Normalize to compare across objectives.
            norm_objs = (objectives - np.min(objectives, axis=0)) / (np.max(objectives, axis=0) - np.min(objectives, axis=0) + 1e-8)
            worst_idx = np.argmax(np.sum(norm_objs, axis=1))
            worst_solution = population[worst_idx]

            offspring = []
            off_objs = []
            
            for i in range(self.pop_size):
                current_ox = population[i].copy()
                guard = guards[self.rng.choice(len(guards))]
                
                # Phase selection mimics stochastic herd behavior.
                phase_prob = self.rng.random()
                new_ox = np.zeros_like(current_ox)
                
                if phase_prob < 0.33:
                    # Migratory phase: move toward the guard.
                    r1 = self.rng.uniform(0.5, 1.5, size=len(current_ox))
                    new_ox = current_ox + r1 * (guard - current_ox)
                
                elif phase_prob < 0.66:
                    # Foraging phase: explore locally near the current position.
                    r2 = self.rng.uniform(0, 0.5, size=len(current_ox))
                    local_noise = self.rng.normal(0, 0.05 * (np.array([b[1] for b in self.bounds]) - np.array([b[0] for b in self.bounds])))
                    new_ox = current_ox + r2 * (guard - current_ox) + local_noise
                
                else:
                    # Defensive phase: move away from the worst solution (predator).
                    r3 = self.rng.uniform(0.5, 1.0, size=len(current_ox))
                    new_ox = current_ox + r3 * (current_ox - worst_solution)
                
                # Apply boundary constraints.
                for j in range(len(new_ox)):
                    low, high = self.bounds[j]
                    new_ox[j] = np.clip(new_ox[j], low, high)
                
                obj = self.fitness_fn(new_ox)
                offspring.append(new_ox)
                off_objs.append(obj)
                
                # Update the external Pareto archive.
                archive, archive_objs = self.update_archive(archive, archive_objs, new_ox, obj)

            population = np.array(offspring)
            objectives = np.array(off_objs)
            
            best = np.min(objectives[:, 0])
            history.append(best)
            print(f"Gen {gen+1} complete — archive size: {len(archive)} | best val_mse: {best:.6f}")

        pareto = []
        for p, o in zip(archive, archive_objs):
            pareto.append({
                "params": p,
                "val_mse": float(o[0]),
                "complexity": int(o[1]),
            })

        print("\n================ Musk Ox Optimization Finished ================")
        return pareto, history