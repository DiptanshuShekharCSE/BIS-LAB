import random
import numpy as np

def total_distance(tour, dist_matrix):
    distance = 0
    for i in range(len(tour)):
        distance += dist_matrix[tour[i - 1]][tour[i]]
    return distance

def random_tour(n_cities):
    tour = list(range(n_cities))
    random.shuffle(tour)
    return tour

def levy_flight_move(tour):
    n = len(tour)
    new_tour = tour.copy()
    i, j = sorted(random.sample(range(n), 2))
    new_tour[i:j+1] = reversed(new_tour[i:j+1])
    a, b = random.sample(range(n), 2)
    new_tour[a], new_tour[b] = new_tour[b], new_tour[a]
    return new_tour

def abandon_nests(nests, fitness, Pa, best_tour):
    n = len(nests)
    n_abandon = int(Pa * n)
    worst_indices = np.argsort(fitness)[-n_abandon:]
    for idx in worst_indices:
        new_tour = levy_flight_move(best_tour)
        nests[idx] = new_tour
        fitness[idx] = None

def cuckoo_search_tsp(dist_matrix, n_nests=25, Pa=0.25, max_iter=1000):
    n_cities = len(dist_matrix)
    nests = [random_tour(n_cities) for _ in range(n_nests)]
    fitness = [total_distance(tour, dist_matrix) for tour in nests]
    best_idx = np.argmin(fitness)
    best_tour = nests[best_idx].copy()
    best_fitness = fitness[best_idx]

    for t in range(max_iter):
        for i in range(n_nests):
            new_tour = levy_flight_move(nests[i])
            new_fit = total_distance(new_tour, dist_matrix)
            j = i
            while j == i:
                j = random.randint(0, n_nests - 1)
            if new_fit < fitness[j]:
                nests[j] = new_tour
                fitness[j] = new_fit
                if new_fit < best_fitness:
                    best_fitness = new_fit
                    best_tour = new_tour.copy()
        abandon_nests(nests, fitness, Pa, best_tour)
        for i in range(n_nests):
            if fitness[i] is None:
                fitness[i] = total_distance(nests[i], dist_matrix)
                if fitness[i] < best_fitness:
                    best_fitness = fitness[i]
                    best_tour = nests[i].copy()
        if (t + 1) % 100 == 0 or t == 0:
            print(f"Iteration {t+1}, Best distance: {best_fitness:.4f}")
    return best_tour, best_fitness

if __name__ == "__main__":
    dist_matrix = [
        [0, 2, 9, 10, 7],
        [2, 0, 6, 4, 3],
        [9, 6, 0, 8, 5],
        [10, 4, 8, 0, 6],
        [7, 3, 5, 6, 0]
    ]
    best_path, best_dist = cuckoo_search_tsp(dist_matrix, n_nests=20, Pa=0.3, max_iter=500)
    print("Best tour found:", best_path)
    print("Best tour length:", best_dist)
