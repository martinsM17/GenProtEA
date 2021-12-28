from inspyred.ec.emo import Pareto
import random
import numpy as np


def dominance_test(solution1, solution2, maximize = True):
    """
    Testes Pareto dominance
    args
        solution1 : The first solution 
        solution2 : The second solution
        maximize (bool): maximization (True) or minimization (False)
    
    returns 
         1 : if the first solution dominates the second 
        -1 : if the second solution dominates the first
         0 : if non of the solutions dominates the other
    """
    best_is_one = 0
    best_is_two = 0

    if isinstance(solution1.fitness,Pareto):
        values1 = solution1.fitness.values 
        values2 = solution2.fitness.values
    else :
        values1 = [solution1.fitness] 
        values2 = [solution2.fitness]

    for i in range(len(values1)):
        value1 = values1[i]
        value2 = values2[i]
        if value1 != value2:
            if value1 < value2:
                best_is_two = 1
            if value1 > value2:
                best_is_one = 1

    if best_is_one > best_is_two:
        if maximize:
            result = 1
        else:    
            result = -1
    elif best_is_two > best_is_one:
        if maximize:
            result = -1
        else:
            result = 1
    else:
        result = 0

    return result




def non_dominated_population(population, maximize = True):
    """
    returns the non dominated solutions from the population.
    """
    population.sort(reverse = True)
    non_dominated = []
    for i in range(len(population)-1):
        individual = population[i]
        j = 0
        dominates = True
        while j < len(population) and dominates:
            if dominance_test(individual,population[j]) == -1:
                dominates = False
            else:
                j += 1
        if dominates:
            non_dominated.append(individual)

    return non_dominated



def ls_mixer(latent, n = 100, ratios = 5):
    new_ls = []
    ratios = np.linspace(0, 1, ratios)[1:-1]
    for i in range(n):
        j = random.randrange(0, len(latent), 1)
        k = random.randrange(0, len(latent), 1)
        latent1 = latent[j:j + 1][0]
        latent0 = latent[k:k + 1][0]
        for r in ratios:
            #rlatent = [x * (1.0 - r) for x in latent0] + [x * r for x in latent1]
            rlatent = [round(x + y, 4) for x, y in zip([x * (1.0 - r) for x in latent0], [x * r for x in latent1])]
            for _ in range(5): rlatent[random.randint(0,len(latent))] = random.random()
            new_ls.append(rlatent)

    return new_ls
