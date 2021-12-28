from loadModels import loadGAN
from optimization.evaluation import Min_Rules_Synthesis, Min_Rules_Solubility, Max_Hidrophobicity, Prob_Hmm
import pandas as pd
import csv

'''
Generate samples with the GAN architecture, varying the latent seed and evaluating the intended case studies
'''

def generate_samples(seed):
    model = loadGAN()
    samples = model.generate(latent_seed=seed)
    samples = samples['sequence']
    return samples

def evaluate_seed(samples, seed):
    destFile = "/home/mmartins/GenProtEA/output/GAN_results.csv"
    header = ['proteins','hydrophobicity','solubility_rules','synthesis_rules','hmm','seed']
    with open(destFile, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for seq in samples:
            avg_hidro = Max_Hidrophobicity()
            avg_hidro = avg_hidro.get_fitness(seq, batched=False)
            solub_score = Min_Rules_Solubility()
            synthesis_score = Min_Rules_Synthesis()
            solub_score = solub_score.get_fitness(seq, batched=False)
            synthesis_score = synthesis_score.get_fitness(seq, batched=False) 
            prob_hmm = Prob_Hmm()
            prob_hmm = prob_hmm.get_fitness(seq, batched=False)
            data = [seq, avg_hidro, solub_score, synthesis_score, prob_hmm, seed]
            writer.writerow(data)

            
        f.close()
    return destFile


if __name__ == "__main__":
    import os
    import pandas as pd
    ## Set GPU ##
    gpu = 1
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu)

    # Set seeds and run
   
    
    
    for i in range(100):
        seed = i
        samples = generate_samples(seed)
        evaluate_seed(samples, seed)
        i+=1
    
