import numpy as np


class PSO:
    def __init__(self, fitness_fn, bounds, swarm_size=5, iterations=5,
                 w=0.7, c1=1.5, c2=1.5, seed=42):

        self.fitness_fn = fitness_fn
        self.bounds = np.array(bounds)  # shape: (dim, 2)
        self.swarm_size = swarm_size
        self.iterations = iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2

        np.random.seed(seed)

        self.dim = self.bounds.shape[0]

        # Initialize positions uniformly within bounds
        self.positions = np.random.uniform(
            self.bounds[:, 0],
            self.bounds[:, 1],
            (self.swarm_size, self.dim)
        )

        # Initialize velocities to small random values
        self.velocities = np.random.uniform(
            -abs(self.bounds[:, 1] - self.bounds[:, 0]),
            abs(self.bounds[:, 1] - self.bounds[:, 0]),
            (self.swarm_size, self.dim)
        ) * 0.1

        # Personal best
        self.pbest_positions = np.copy(self.positions)
        self.pbest_scores = np.array([np.inf] * self.swarm_size)

        # Global best
        self.gbest_position = None
        self.gbest_score = np.inf

    def optimize(self):

        # --- Initial evaluation ---
        print("\n########## PSO Initial Evaluation ##########")

        for i in range(self.swarm_size):

            print(f"\n---- Initial Particle {i+1}/{self.swarm_size} ----")
            
            score = self.fitness_fn(self.positions[i])
            self.pbest_scores[i] = score
            self.pbest_positions[i] = self.positions[i]

            if score < self.gbest_score:
                self.gbest_score = score
                self.gbest_position = self.positions[i].copy()

        # --- Main loop ---
        for iteration in range(self.iterations):

            print(f"\n########## PSO Iteration {iteration+1}/{self.iterations} ##########")

            for i in range(self.swarm_size):

                print(f"\n---- Particle {i+1}/{self.swarm_size} ----")

                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)

                cognitive = self.c1 * r1 * (self.pbest_positions[i] - self.positions[i])
                social = self.c2 * r2 * (self.gbest_position - self.positions[i])

                self.velocities[i] = (
                    self.w * self.velocities[i]
                    + cognitive
                    + social
                )

                # Update position
                self.positions[i] += self.velocities[i]

                # Clip to bounds
                self.positions[i] = np.clip(
                    self.positions[i],
                    self.bounds[:, 0],
                    self.bounds[:, 1]
                )

                # Evaluate fitness
                score = self.fitness_fn(self.positions[i])

                # Update personal best
                if score < self.pbest_scores[i]:
                    self.pbest_scores[i] = score
                    self.pbest_positions[i] = self.positions[i].copy()

                # Update global best
                if score < self.gbest_score:
                    self.gbest_score = score
                    self.gbest_position = self.positions[i].copy()

        return self.gbest_position, self.gbest_score