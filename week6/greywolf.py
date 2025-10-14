import numpy as np

# Define the network nodes (fixed)
network_nodes = np.array([
    [2, 3],
    [5, 4],
    [9, 6],
    [4, 7],
    [8, 1],
    [7, 9]
])

# Number of resources (wolves) to place in the network (dimension of the solution)
num_resources = 3  # each resource is a point (x, y)
dim = num_resources * 2  # x and y coordinates for each resource

# Define the search space boundaries for resource locations
x_min, x_max = 0, 10
y_min, y_max = 0, 10

def fitness_function(position):
    """
    Given a flattened position array [x1, y1, x2, y2, ..., xn, yn] for n resources,
    compute sum of minimum Euclidean distances from each network node to nearest resource.
    """
    resources = position.reshape(num_resources, 2)
    total_distance = 0
    for node in network_nodes:
        dist_to_resources = np.linalg.norm(resources - node, axis=1)
        total_distance += np.min(dist_to_resources)
    return total_distance

class GreyWolfOptimizer:
    def __init__(self, fitness_func, dim, num_wolves=5, max_iter=30):
        self.fitness_func = fitness_func
        self.dim = dim
        self.num_wolves = num_wolves
        self.max_iter = max_iter

        # Initialize wolves' positions randomly within bounds
        self.positions = np.random.uniform(low=[x_min, y_min] * num_resources, 
                                           high=[x_max, y_max] * num_resources, 
                                           size=(num_wolves, dim))
        self.alpha_pos = np.zeros(dim)
        self.alpha_score = float('inf')

        self.beta_pos = np.zeros(dim)
        self.beta_score = float('inf')

        self.delta_pos = np.zeros(dim)
        self.delta_score = float('inf')

    def optimize(self):
        for iter_no in range(self.max_iter):
            for i in range(self.num_wolves):
                # Boundary check
                self.positions[i] = np.clip(self.positions[i], [x_min, y_min] * num_resources, [x_max, y_max] * num_resources)

                fitness = self.fitness_func(self.positions[i])

                # Update Alpha, Beta, Delta
                if fitness < self.alpha_score:
                    self.delta_score = self.beta_score
                    self.delta_pos = self.beta_pos.copy()

                    self.beta_score = self.alpha_score
                    self.beta_pos = self.alpha_pos.copy()

                    self.alpha_score = fitness
                    self.alpha_pos = self.positions[i].copy()

                elif fitness < self.beta_score:
                    self.delta_score = self.beta_score
                    self.delta_pos = self.beta_pos.copy()

                    self.beta_score = fitness
                    self.beta_pos = self.positions[i].copy()

                elif fitness < self.delta_score:
                    self.delta_score = fitness
                    self.delta_pos = self.positions[i].copy()

            a = 2 - iter_no * (2 / self.max_iter)  # linearly decreased from 2 to 0

            # Update wolf positions
            for i in range(self.num_wolves):
                for d in range(self.dim):
                    r1 = np.random.rand()
                    r2 = np.random.rand()

                    A1 = 2 * a * r1 - a
                    C1 = 2 * r2

                    D_alpha = abs(C1 * self.alpha_pos[d] - self.positions[i, d])
                    X1 = self.alpha_pos[d] - A1 * D_alpha

                    r1 = np.random.rand()
                    r2 = np.random.rand()
                    A2 = 2 * a * r1 - a
                    C2 = 2 * r2

                    D_beta = abs(C2 * self.beta_pos[d] - self.positions[i, d])
                    X2 = self.beta_pos[d] - A2 * D_beta

                    r1 = np.random.rand()
                    r2 = np.random.rand()
                    A3 = 2 * a * r1 - a
                    C3 = 2 * r2

                    D_delta = abs(C3 * self.delta_pos[d] - self.positions[i, d])
                    X3 = self.delta_pos[d] - A3 * D_delta

                    self.positions[i, d] = (X1 + X2 + X3) / 3

            print(f"Iteration {iter_no+1}/{self.max_iter} - Best Fitness: {self.alpha_score:.4f}")

        return self.alpha_pos, self.alpha_score

# Running the optimizer
gwo = GreyWolfOptimizer(fitness_function, dim, num_wolves=10, max_iter=50)
best_position, best_fitness = gwo.optimize()

print("\nOptimal resource positions (x,y):")
resources_optimal = best_position.reshape(num_resources, 2)
for idx, pos in enumerate(resources_optimal):
    print(f"Resource {idx+1}: {pos}")

print(f"\nMinimum total latency (sum of distances): {best_fitness:.4f}")
