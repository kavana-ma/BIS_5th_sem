import numpy as np

class AntColony:
    def __init__(self, distances, n_ants, n_iterations, alpha=1.0, beta=2.0, rho=0.5, initial_pheromone=1.0):
        """
        distances: 2D numpy array, distances[i][j] is the distance from city i to j
        n_ants: number of ants
        n_iterations: number of iterations to run
        alpha: importance of pheromone
        beta: importance of heuristic (1/distance)
        rho: pheromone evaporation rate
        initial_pheromone: initial pheromone value on all edges
        """
        self.distances = distances
        self.pheromone = np.ones(self.distances.shape) * initial_pheromone
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.n_cities = distances.shape[0]
        self.best_route = None
        self.best_distance = float('inf')

    def run(self):
        for iteration in range(self.n_iterations):
            all_routes = self.construct_solutions()
            self.update_pheromones(all_routes)
            print(f"Iteration {iteration + 1}/{self.n_iterations}: Best distance so far = {self.best_distance:.4f}")
        return self.best_route, self.best_distance


    def construct_solutions(self):
        all_routes = []
        for ant in range(self.n_ants):
            route = self.construct_solution()
            distance = self.route_distance(route)
            if distance < self.best_distance:
                self.best_distance = distance
                self.best_route = route
            all_routes.append((route, distance))
        return all_routes

    def construct_solution(self):
        route = []
        unvisited = list(range(self.n_cities))
        current_city = np.random.choice(unvisited)
        route.append(current_city)
        unvisited.remove(current_city)

        while unvisited:
            probabilities = self.transition_probabilities(current_city, unvisited)
            next_city = np.random.choice(unvisited, p=probabilities)
            route.append(next_city)
            unvisited.remove(next_city)
            current_city = next_city

        return route

    def transition_probabilities(self, current_city, unvisited):
        pheromone = self.pheromone[current_city, unvisited] ** self.alpha
        heuristic = (1.0 / self.distances[current_city, unvisited]) ** self.beta
        product = pheromone * heuristic
        probabilities = product / np.sum(product)
        return probabilities

    def update_pheromones(self, all_routes):
        # Evaporate pheromones
        self.pheromone *= (1 - self.rho)

        # Deposit new pheromones based on quality of solutions
        for route, distance in all_routes:
            pheromone_deposit = 1.0 / distance
            for i in range(len(route)):
                from_city = route[i]
                to_city = route[(i + 1) % self.n_cities]  # wrap around to start
                self.pheromone[from_city, to_city] += pheromone_deposit
                self.pheromone[to_city, from_city] += pheromone_deposit  # symmetric

    def route_distance(self, route):
        distance = 0.0
        for i in range(len(route)):
            from_city = route[i]
            to_city = route[(i + 1) % self.n_cities]  # wrap-around for return trip
            distance += self.distances[from_city, to_city]
        return distance

# Example usage:

def create_distance_matrix(coords):
    n = len(coords)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist_matrix[i][j] = np.linalg.norm(np.array(coords[i]) - np.array(coords[j]))
    return dist_matrix

if __name__ == "__main__":
    # Define cities (coordinates)
    cities = [
        (0, 0),
        (1, 5),
        (5, 2),
        (6, 6),
        (8, 3),
        (2, 1)
    ]

    distances = create_distance_matrix(cities)

    # Initialize ACO parameters
    n_ants = 10
    n_iterations = 10
    alpha = 1.0
    beta = 5.0
    rho = 0.5
    initial_pheromone = 1.0

    aco = AntColony(distances, n_ants, n_iterations, alpha, beta, rho, initial_pheromone)
    best_route, best_distance = aco.run()

    print("Best route found:", best_route)
    print("Best distance:", best_distance)
