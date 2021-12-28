from loadModels import loadGAN
from optimization.evaluation import Min_Rules_Synthesis, Min_Rules_Solubility, Max_Hidrophobicity, Prob_Hmm
import pandas as pd
import csv
'''
Evaluate the Raw dataset on the intended metric
'''
def evaluate(samples):
    destFile = "/home/mmartins/GenProtEA/output/VAE_generated_results.csv"
    header = ['proteins','hydrophobicity','solubility_rules','synthesis_rules','hmm']
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
            data = [seq,avg_hidro,solub_score,synthesis_score,prob_hmm]
            writer.writerow(data)
            
        f.close()
    return destFile


if __name__ == "__main__":
    import os
    from Bio import SeqIO
    ## Set GPU ##
    gpu = 2
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu)

    # Set seeds and run
    list_prots=[]
    for record in SeqIO.parse('vae_generated.fasta', 'fasta'):
        sample = str(record.seq)
        list_prots.append(sample)
    samples = list_prots
    evaluate(samples)
        
    
