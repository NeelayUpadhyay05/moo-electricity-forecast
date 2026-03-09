import numpy as np


class PSO:
    def __init__(self, fitness_fn, bounds, swarm_size=5, iterations=5,
                 w=0.7, c1=1.5, c2=1.5, seed=42):

        self.fitness_fn = fitness_fn
        self.orig_bounds = np.array(bounds)   # shape: (dim, 2) — kept for denormalization
        self.swarm_size = swarm_size
        self.iterations = iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2

        self.rng = np.random.default_rng(seed)

        self.dim = self.orig_bounds.shape[0]
        self._lows  = self.orig_bounds[:, 0]
        self._highs = self.orig_bounds[:, 1]

        # All internal positions and velocities live in [0, 1]^d.
        # This ensures cognitive/social terms are on equal footing
        # regardless of the original dimension scales (e.g., hidden_dim
        # range 224 vs dropout range 0.3).
        self.v_max = np.ones(self.dim)   # full normalized range per dimension

        # Initialize positions uniformly in [0, 1]
        self.positions = self.rng.uniform(0.0, 1.0, (self.swarm_size, self.dim))

        # Initialize velocities as small fractions of the normalized range
        self.velocities = self.rng.uniform(-1.0, 1.0, (self.swarm_size, self.dim)) * 0.1

        # Personal best (normalized)
        self.pbest_positions = np.copy(self.positions)
        self.pbest_scores = np.array([np.inf] * self.swarm_size)

        # Global best (normalized)
        self.gbest_position = None
        self.gbest_score = np.inf

    def _denormalize(self, position):
        """Map a normalized position in [0,1]^d back to original bounds."""
        return self._lows + position * (self._highs - self._lows)

    def optimize(self):

        history = []

        # --- Initial evaluation ---
        print("\n########## PSO Initial Evaluation ##########")

        for i in range(self.swarm_size):

            print(f"\n---- Initial Particle {i+1}/{self.swarm_size} ----")

            score = self.fitness_fn(self._denormalize(self.positions[i]))
            self.pbest_scores[i] = score
            self.pbest_positions[i] = self.positions[i]

            if score < self.gbest_score:
                self.gbest_score = score
                self.gbest_position = self.positions[i].copy()

            history.append(self.gbest_score)

        # --- Main loop ---
        for iteration in range(self.iterations):

            print(f"\n########## PSO Iteration {iteration+1}/{self.iterations} ##########")

            for i in range(self.swarm_size):

                print(f"\n---- Particle {i+1}/{self.swarm_size} ----")

                r1 = self.rng.random(self.dim)
                r2 = self.rng.random(self.dim)

                cognitive = self.c1 * r1 * (self.pbest_positions[i] - self.positions[i])
                social = self.c2 * r2 * (self.gbest_position - self.positions[i])

                self.velocities[i] = (
                    self.w * self.velocities[i]
                    + cognitive
                    + social
                )

                # Clamp velocities to normalized range
                self.velocities[i] = np.clip(
                    self.velocities[i], -self.v_max, self.v_max
                )

                # Update normalized position and reflect velocity at boundaries
                self.positions[i] += self.velocities[i]
                clipped = np.clip(self.positions[i], 0.0, 1.0)
                hit_boundary = clipped != self.positions[i]
                self.velocities[i][hit_boundary] = 0.0
                self.positions[i] = clipped

                # Evaluate fitness using denormalized position
                score = self.fitness_fn(self._denormalize(self.positions[i]))

                # Update personal best
                if score < self.pbest_scores[i]:
                    self.pbest_scores[i] = score
                    self.pbest_positions[i] = self.positions[i].copy()

                # Update global best
                if score < self.gbest_score:
                    self.gbest_score = score
                    self.gbest_position = self.positions[i].copy()

                history.append(self.gbest_score)

        return self._denormalize(self.gbest_position), self.gbest_score, history
