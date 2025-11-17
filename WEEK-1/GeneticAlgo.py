import random
import math

def fitness_function(design):
    thickness, supports, density = design
    strength = (thickness * 5) + (supports * 12) - (density * 0.5)
    materials_used = (thickness * 2) + (supports * 5) + (density * 1.5)
    if materials_used <= 0:
        return 0
    return strength / materials_used

def create_individual():
    thickness = random.uniform(1, 10)
    supports = random.randint(1, 20)
    density = random.uniform(1, 8)
    return [thickness, supports, density]

def create_population(size):
    return [create_individual() for _ in range(size)]

def selection(population, fitness_scores):
    total_fit = sum(fitness_scores)
    pick = random.uniform(0, total_fit)
    current = 0
    for i, f in enumerate(fitness_scores):
        current += f
        if current > pick:
            return population[i]

def crossover(parent1, parent2, crossover_rate=0.8):
    if random.random() > crossover_rate:
        return parent1[:], parent2[:]
    point = random.randint(1, 2)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2

def mutate(individual, mutation_rate=0.1):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            if i == 0:
                individual[i] = random.uniform(1, 10)
            elif i == 1:
                individual[i] = random.randint(1, 20)
            elif i == 2:
                individual[i] = random.uniform(1, 8)
    return individual

def genetic_algorithm(population_size=20, generations=50, crossover_rate=0.8, mutation_rate=0.1):
    population = create_population(population_size)
    best_solution = None
    best_fitness = -999

    for gen in range(generations):
        fitness_scores = [fitness_function(ind) for ind in population]
        gen_best = population[fitness_scores.index(max(fitness_scores))]

        if max(fitness_scores) > best_fitness:
            best_fitness = max(fitness_scores)
            best_solution = gen_best

        new_population = []
        while len(new_population) < population_size:
            parent1 = selection(population, fitness_scores)
            parent2 = selection(population, fitness_scores)
            child1, child2 = crossover(parent1, parent2, crossover_rate)
            child1 = mutate(child1, mutation_rate)
            child2 = mutate(child2, mutation_rate)
            new_population.append(child1)
            if len(new_population) < population_size:
                new_population.append(child2)

        population = new_population

    return best_solution, best_fitness

best_design, best_score = genetic_algorithm()
print("\nBest Bridge Design Found:")
print(f"Thickness: {best_design[0]:.2f} cm")
print(f"Supports : {best_design[1]}")
print(f"Density  : {best_design[2]:.2f} g/cm³")
print(f"Fitness Score: {best_score:.4f}")
