import os
import random
import warnings
warnings.filterwarnings('ignore')
import sys
sys.path.append('/home/mmartins/GenProtEA')
import numpy as np
from loadModels import loadVAE
from optimization import EA
from optimization import set_default_engine, get_default_engine
from utils.utilities import ls_mixer
from Bio import SeqIO
from optimization.problem import proteinReporter
from utils.data_loaders import one_hot_generator

'''
Set an run the experiment and define the output directory and file
'''

def initializePop():
    init_pop = []
    return init_pop

def perturbPop(init_pop, size=100, ratios=5):
    return ls_mixer(init_pop, size, ratios)

def savePop(final_pop, fNames, fUsed, totalGenerations):

    model = loadVAE()
    model.batch_size = 100
    model.E.batch_size = model.batch_size
    model.G.batch_size = model.batch_size
    print(len(final_pop))
    destFile = "/home/mmartins/GenProtEA/output/VAE_nsga.csv"
            
    with open(destFile,'a') as f: 
        f.write("Proteins;"+fNames+"\n")

        pop = []

        listLatents = [solut.values for solut in final_pop]
        fn_problem = proteinReporter(fUsed)
        listScores, listProts = fn_problem.evaluate_solution(listLatents)
        print(len(listScores))
        print(len(listProts))
        for i, solution in enumerate(final_pop):
            solution.fitness = [0.0 for _ in range(len(listScores[0]))]

            for j in range(len(listScores[i])):
                solution.fitness[j] = listScores[i][j]

            pop.append(listProts[i])
            a = [str(float(score)) for score in solution.fitness]

            f.write(str(listProts[i])+";"+";".join(a)+"\n")
    
    
    return pop        


def run(objective):
    
    #set_default_engine('inspyred')
    set_default_engine('jmetal')

    # Read configurations
    generations = 160
    algorithm = "NSGAII"

    # Initialize Population    
    #init_pop = initializePop()

    # Initialize Objectives
    problem, fNames, fUsed = objective()

    # Initialize EA
    ea = EA(problem, max_generations=generations, mp=False,  visualizer=False, algorithm=algorithm, batched=True)
    
    # Run EA
    final_pop = ea.run()
    
    # Generations
    if get_default_engine() == "jmetal":
        totalGenerations = int(ea.termination_criterion.evaluations / ea.population_size)
    else: totalGenerations = ea.max_generations
    
    # Save population

    destFile = savePop(final_pop, fNames, fUsed, totalGenerations)
    
    return destFile

           
if __name__ == '__main__' and True:
    
    import caseStudies 
    # Set seeds
    seed = 41
    random.seed(seed)
    os.environ['PYTHONHASHSEED']=str(seed)
    np.random.seed(seed)
    import tensorflow as tf
    try: tf.random.set_seed(seed)
    except: tf.set_random_seed(seed)


    case = getattr(caseStudies, 'caseMinRules')
    #case = getattr(caseStudies, 'caseMaxHydro')
    objective = case.objective

    run(objective)

