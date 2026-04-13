import numpy as np


class PSO:
    def __init__(
        self,
        fitness_fn,
        bounds,
        swarm_size=5,
        iterations=5,
        w=0.7,
        c1=1.5,
        c2=1.5,
        seed=42,
        archive_size=None,
    ):

        self.fitness_fn = fitness_fn
        self.orig_bounds = np.array(bounds)
        self.swarm_size = swarm_size
        self.iterations = iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2

        self.rng = np.random.default_rng(seed)

        self.dim = self.orig_bounds.shape[0]
        self._lows = self.orig_bounds[:, 0]
        self._highs = self.orig_bounds[:, 1]

        self.v_max = np.ones(self.dim)

        self.positions = self.rng.uniform(0.0, 1.0, (self.swarm_size, self.dim))
        self.velocities = self.rng.uniform(-1.0, 1.0, (self.swarm_size, self.dim)) * 0.1

        self.pbest_positions = np.copy(self.positions)
        self.pbest_scores = [None] * self.swarm_size

        self.archive_positions = []
        self.archive_scores = []
        self.archive_size = archive_size if archive_size is not None else self.swarm_size

    def _denormalize(self, position):
        return self._lows + position * (self._highs - self._lows)

    @staticmethod
    def _dominates(score_a, score_b):
        return (
            (score_a[0] <= score_b[0] and score_a[1] <= score_b[1])
            and (score_a[0] < score_b[0] or score_a[1] < score_b[1])
        )

    def _update_archive(self, position, score):
        non_dominated_positions = []
        non_dominated_scores = []
        candidate_dominated = False

        for pos, sc in zip(self.archive_positions, self.archive_scores):
            if self._dominates(sc, score):
                candidate_dominated = True
                break
            if self._dominates(score, sc):
                continue
            non_dominated_positions.append(pos)
            non_dominated_scores.append(sc)

        if candidate_dominated:
            return

        non_dominated_positions.append(position.copy())
        non_dominated_scores.append((float(score[0]), float(score[1])))

        self.archive_positions = non_dominated_positions
        self.archive_scores = non_dominated_scores

        if len(self.archive_positions) > self.archive_size:
            distances = self._crowding_distance(self.archive_scores)
            keep_indices = np.argsort(-distances)[:self.archive_size]
            self.archive_positions = [self.archive_positions[i] for i in keep_indices]
            self.archive_scores = [self.archive_scores[i] for i in keep_indices]

    def _crowding_distance(self, scores):
        n = len(scores)
        if n == 0:
            return np.array([])
        if n == 1:
            return np.array([np.inf])

        obj = np.array(scores, dtype=float)
        distance = np.zeros(n, dtype=float)

        for m in range(obj.shape[1]):
            order = np.argsort(obj[:, m])
            distance[order[0]] = np.inf
            distance[order[-1]] = np.inf

            span = obj[order[-1], m] - obj[order[0], m]
            if span == 0:
                continue

            for i in range(1, n - 1):
                distance[order[i]] += (obj[order[i + 1], m] - obj[order[i - 1], m]) / span

        return distance

    def _select_leader(self):
        if not self.archive_positions:
            idx = int(self.rng.integers(0, self.swarm_size))
            return self.positions[idx]

        distances = self._crowding_distance(self.archive_scores)
        if np.isinf(distances).all():
            idx = int(self.rng.integers(0, len(self.archive_positions)))
            return self.archive_positions[idx]

        finite = np.where(np.isfinite(distances), distances, np.max(distances[np.isfinite(distances)]))
        probs = finite + 1e-12
        probs = probs / probs.sum()
        idx = int(self.rng.choice(len(self.archive_positions), p=probs))
        return self.archive_positions[idx]

    def optimize(self):

        history = []
        best_val_so_far = float("inf")

        print("\n########## MOPSO Initial Evaluation ##########")

        for i in range(self.swarm_size):
            print(f"\n---- Initial Particle {i + 1}/{self.swarm_size} ----")

            score = self.fitness_fn(self._denormalize(self.positions[i]))
            score = (float(score[0]), float(score[1]))

            self.pbest_scores[i] = score
            self.pbest_positions[i] = self.positions[i].copy()
            self._update_archive(self.positions[i], score)

            best_val_so_far = min(best_val_so_far, score[0])
            history.append(best_val_so_far)

        for iteration in range(self.iterations):

            print(f"\n########## MOPSO Iteration {iteration + 1}/{self.iterations} ##########")

            for i in range(self.swarm_size):

                print(f"\n---- Particle {i + 1}/{self.swarm_size} ----")

                r1 = self.rng.random(self.dim)
                r2 = self.rng.random(self.dim)

                leader = self._select_leader()

                cognitive = self.c1 * r1 * (self.pbest_positions[i] - self.positions[i])
                social = self.c2 * r2 * (leader - self.positions[i])

                self.velocities[i] = self.w * self.velocities[i] + cognitive + social
                self.velocities[i] = np.clip(self.velocities[i], -self.v_max, self.v_max)

                self.positions[i] += self.velocities[i]
                clipped = np.clip(self.positions[i], 0.0, 1.0)
                hit_boundary = clipped != self.positions[i]
                self.velocities[i][hit_boundary] = 0.0
                self.positions[i] = clipped

                score = self.fitness_fn(self._denormalize(self.positions[i]))
                score = (float(score[0]), float(score[1]))

                pbest = self.pbest_scores[i]
                if self._dominates(score, pbest):
                    self.pbest_scores[i] = score
                    self.pbest_positions[i] = self.positions[i].copy()
                elif not self._dominates(pbest, score):
                    if self.rng.random() < 0.5:
                        self.pbest_scores[i] = score
                        self.pbest_positions[i] = self.positions[i].copy()

                self._update_archive(self.positions[i], score)

                best_val_so_far = min(best_val_so_far, score[0])
                history.append(best_val_so_far)

        pareto_solutions = []
        for pos, sc in zip(self.archive_positions, self.archive_scores):
            pareto_solutions.append(
                {
                    "params": self._denormalize(pos),
                    "val_mse": float(sc[0]),
                    "complexity": float(sc[1]),
                }
            )

        pareto_solutions.sort(key=lambda x: (x["val_mse"], x["complexity"]))

        return pareto_solutions, history
