import os
from datetime import datetime
from optimization.evaluation import Max_Hidrophobicity, Min_Rules_Solubility, Min_Rules_Synthesis, Min_reps
import sys
sys.path.append('/home/mmartins/GenProtEA')
from run import *
from loadModels import loadVAE
from optimization.problem import proteinProblem

import tensorflow as tf
import pandas as pd
import numpy as np

'''
Perform an eavluation of the intended case studies
'''

class caseStudy():
    def __init__(self):
        self.name = None
        self.bestScores = None
        self.bestPROTS = None
        self.mean = None
        self.std = None
        self.resNames = None

    def objective(self, configInsi, multiObjective=False):
        raise NotImplementedError

    def prepDirectory(self):
        pass

    def repeatOptimization(self):
        # Repeat optimization the defined number of times
        self.results = []
        #self.time = None
        #startTime = self.time

        for _ in range(100):
            #self.time = datetime.now().strftime('%m-%d_%H-%M-%S')
            destFile = run(self.objective)
            self.results.append(destFile)

        #self.time = startTime

    def parseResults(self):
        raise NotImplementedError

    def computeMetrics(self):
        singleScore = np.array(self.bestScores)[:, 0]
        self.mean = np.round(np.mean(singleScore), 4)
        self.std = np.round(np.std(singleScore), 4)
        print("\nRESULTS:", str(self.mean) + " +/- " + str(self.std))

        self.resTuple = list(zip(self.bestScores, self.bestPROTS))
        self.resTuple.sort(key=lambda tup: tup[0][0], reverse=True)

    def saveResults(self):
        with open('/output/' + "results.txt", 'w') as f:
            f.write(str(self.mean) + " +/- " + str(self.std) + "\n\n")
            f.write(self.resNames)
            for i in range(10):
                for score in self.resTuple[i][0]: f.write("\t" + str(round(score, 5)))
                f.write("\t" + str(self.resTuple[i][1]) + "\n")

    def run(self):
        model = loadVAE()
        self.origResultsPath = '/output/'
        self.repeatOptimization()
        self.parseResults()
        self.computeMetrics()
        self.saveResults()


class caseMinRules(caseStudy):

    def __init__(self):
        super(caseMinRules, self).__init__()
        self.name = "caseMinRules"
        self.resNames = "\tscore\tProts\n"

    def objective(self, multiObjective=True):
        
        f1 = Min_Rules_Solubility()
        f2 = Min_Rules_Synthesis()
        #f3 = Min_reps()
        problem = proteinProblem([f1, f2])#, f3

        fNames = "Min"
        fUsed = [f1, f2]#, f3

        print("\nObjective: Minimize Synthesis and Solubility rules failed", "Multi-objective")

        return problem, fNames, fUsed

    def parseResults(self):
            repeats = 1
            self.bestScores = []
            self.bestPROTS = []
            for i in range(repeats):
                df = pd.read_csv(self.results[i], sep=";", header=0, index_col=False)
                df["score"] = df["Min"]

                self.bestScores.append([np.max(df["score"])])
                self.bestPROTS.append(df[df["score"] == np.max(df["score"])]["Prots"].values[0])

class caseMaxHydro(caseStudy):

    def __init__(self):
        super(caseMaxHydro, self).__init__()
        self.name = "caseMaxHydro"
        self.resNames = "\tscore\tProts\n"

    def objective(self, multiObjective=False):
        
        f1 = Max_Hidrophobicity()
        problem = proteinProblem([f1])

        fNames = "MAX Hydro"
        fUsed = [f1]

        print("\nObjective: Maximize Hydrophobicity", "Single-objective")

        return problem, fNames, fUsed

    def parseResults(self):
            repeats = 1
            self.bestScores = []
            self.bestPROTS = []
            for i in range(repeats):
                df = pd.read_csv(self.results[i], sep=";", header=0, index_col=False)
                df["score"] = df["MAX"]

                self.bestScores.append([np.max(df["score"])])
                self.bestPROTS.append(df[df["score"] == np.max(df["score"])]["PROTS"].values[0])





if __name__ == "__main__":

    ## Set GPU ##
    gpu = 1
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu)
    
    # Set seeds
    random.seed(42)
    np.random.seed(42)
    try: tf.random.set_seed(42)
    except: tf.set_random_seed(42)

    #caseStud = caseMaxHydro()
    caseStud = caseMinRules()
    case = caseStud
    case.run()
    
    #a = caseMaxHydro()
    a = caseMinRules()
    #a.parseResults()
    a.computeMetrics()
    #a.saveResults()    
