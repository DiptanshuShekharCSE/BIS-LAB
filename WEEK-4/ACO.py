import numpy as np

# Sample data: coordinates of depot + customers
locations = np.array([
    [50, 50],  # Depot
    [20, 30], [40, 70], [60, 20], [80, 40], [70, 80]  # Customers
])

num_vehicles = 2
vehicle_capacity = 100

# Demand of each customer (index 0 is depot with 0 demand)
demands = [0, 40, 50, 30, 60, 20]

num_customers = len(locations) - 1

# Distance matrix
def distance_matrix(loc):
    n = len(loc)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist[i][j] = np.linalg.norm(loc[i] - loc[j])
    return dist

dist = distance_matrix(locations)

# Parameters
num_ants = 10
alpha = 1.0    # pheromone importance
beta = 5.0     # heuristic importance
rho = 0.5      # evaporation rate
Q = 100        # pheromone deposit factor
num_iterations = 5

# Initialize pheromone trails
pheromone = np.ones_like(dist)

# Heuristic info (inverse of distance)
heuristic = 1 / (dist + np.eye(len(dist)) * 1e6)  # Avoid division by zero on diagonal

# Function for ant to build routes for all vehicles
def construct_solution():
    routes = [[] for _ in range(num_vehicles)]
    loads = [0] * num_vehicles
    unvisited = set(range(1, num_customers + 1))

    # Start each vehicle at depot
    current_positions = [0] * num_vehicles

    while unvisited:
        moved = Falseimport numpy as np

locations = np.array([
    [50, 50],
    [20, 30], [40, 70], [60, 20], [80, 40], [70, 80]
])

num_vehicles = 2
vehicle_capacity = 100
demands = [0, 40, 50, 30, 60, 20]
num_customers = len(locations) - 1

def distance_matrix(loc):
    n = len(loc)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist[i][j] = np.linalg.norm(loc[i] - loc[j])
    return dist

dist = distance_matrix(locations)
num_ants = 10
alpha = 1.0
beta = 5.0
rho = 0.5
Q = 100
num_iterations = 50
pheromone = np.ones_like(dist)
heuristic = 1 / (dist + np.eye(len(dist)) * 1e6)

def construct_solution():
    routes = [[] for _ in range(num_vehicles)]
    loads = [0] * num_vehicles
    unvisited = set(range(1, num_customers + 1))
    current_positions = [0] * num_vehicles
    while unvisited:
        moved = False
        for v in range(num_vehicles):
            allowed = [c for c in unvisited if loads[v] + demands[c] <= vehicle_capacity]
            if not allowed:
                if len(routes[v]) == 0 or routes[v][-1] != 0:
                    routes[v].append(0)
                continue
            current_city = current_positions[v]
            probs = []
            for city in allowed:
                tau = pheromone[current_city][city] ** alpha
                eta = heuristic[current_city][city] ** beta
                probs.append(tau * eta)
            probs = np.array(probs)
            probs /= probs.sum()
            next_city = np.random.choice(allowed, p=probs)
            routes[v].append(next_city)
            loads[v] += demands[next_city]
            current_positions[v] = next_city
            unvisited.remove(next_city)
            moved = True
        if not moved:
            break
    for v in range(num_vehicles):
        if len(routes[v]) == 0 or routes[v][-1] != 0:
            routes[v].append(0)
    return routes

def total_distance(routes):
    total_dist = 0
    for route in routes:
        for i in range(len(route) - 1):
            total_dist += dist[route[i]][route[i+1]]
    return total_dist

def update_pheromone(all_routes, all_distances):
    global pheromone
    pheromone *= (1 - rho)
    for routes, dist_length in zip(all_routes, all_distances):
        deposit = Q / dist_length
        for route in routes:
            for i in range(len(route) - 1):
                pheromone[route[i]][route[i+1]] += deposit

best_solution = None
best_distance = float('inf')

for iteration in range(num_iterations):
    all_routes = []
    all_distances = []
    for _ in range(num_ants):
        solution = construct_solution()
        dist_length = total_distance(solution)
        all_routes.append(solution)
        all_distances.append(dist_length)
        if dist_length < best_distance:
            best_distance = dist_length
            best_solution = solution
    update_pheromone(all_routes, all_distances)
    print(f"Iteration {iteration+1} Best distance: {best_distance:.2f}")

print("\nBest solution routes:")
for i, route in enumerate(best_solution):
    print(f" Vehicle {i+1}: {route}")
print(f"Total Distance: {best_distance:.2f}")

        for v in range(num_vehicles):
            # Allowed customers for this vehicle (demand fits in remaining capacity)
            allowed = [c for c in unvisited if loads[v] + demands[c] <= vehicle_capacity]
            if not allowed:
                # Return vehicle to depot (route ends)
                if len(routes[v]) == 0 or routes[v][-1] != 0:
                    routes[v].append(0)
                continue

            current_city = current_positions[v]

            # Calculate probabilities for allowed cities
            probs = []
            for city in allowed:
                tau = pheromone[current_city][city] ** alpha
                eta = heuristic[current_city][city] ** beta
                probs.append(tau * eta)
            probs = np.array(probs)
            probs /= probs.sum()

            # Choose next city probabilistically
            next_city = np.random.choice(allowed, p=probs)

            # Update route, load, position
            routes[v].append(next_city)
            loads[v] += demands[next_city]
            current_positions[v] = next_city
            unvisited.remove(next_city)
            moved = True

        # If no vehicle could move (capacity constraints), break loop
        if not moved:
            break

    # Return to depot if not already there
    for v in range(num_vehicles):
        if len(routes[v]) == 0 or routes[v][-1] != 0:
            routes[v].append(0)

    return routes

# Calculate total distance of all routes
def total_distance(routes):
    total_dist = 0
    for route in routes:
        for i in range(len(route) - 1):
            total_dist += dist[route[i]][route[i+1]]
    return total_dist

# Update pheromone trails
def update_pheromone(all_routes, all_distances):
    global pheromone
    pheromone *= (1 - rho)
    for routes, dist_length in zip(all_routes, all_distances):
        deposit = Q / dist_length
        for route in routes:
            for i in range(len(route) - 1):
                pheromone[route[i]][route[i+1]] += deposit

# Main ACO loop
best_solution = None
best_distance = float('inf')

for iteration in range(num_iterations):
    all_routes = []
    all_distances = []

    for _ in range(num_ants):
        solution = construct_solution()
        dist_length = total_distance(solution)
        all_routes.append(solution)
        all_distances.append(dist_length)

        if dist_length < best_distance:
            best_distance = dist_length
            best_solution = solution

    update_pheromone(all_routes, all_distances)
    print(f"Iteration {iteration+1} Best distance: {best_distance:.2f}")

print("\nBest solution routes:")
for i, route in enumerate(best_solution):
    print(f" Vehicle {i+1}: {route}")
print(f"Total Distance: {best_distance:.2f}")
