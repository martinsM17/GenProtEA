import random
import os
import numpy as np

seq = "MNPATWAYLLLGLAIIAEVTGSTFLVKSEGFTRLWPSLAVVVLFCIAFYLLSQVIKVIPLGIAYAIWAGVGIILTAIVGYIVFKQALDLPAFIGIALIISGVVVINLFSQAAGH"


HOME = '/home/mmartins/GenProtEA/hmmer/'

def evaluateHMM(sequence):
    filename = 'seq'+str(random.randint(1000000000000,999999999999999999999))+'.faa'

    with open(HOME+filename,'w') as f:
        f.write('>seq\n')
        f.write(sequence.replace('-','_'))
    
    stream = os.popen('hmmscan -T 0 '+HOME+'Bac_luciferase.hmm '+HOME+filename)
    output = stream.read()
    # print('-------------')
    # print(output)
    # print('-------------')
    result = -np.inf
    try:
        lines=output.split('\n')
        line = lines[15]
        # print('line',line)
        tokens = line.split()
        result =  float(tokens[1])
    except Exception as e:
        print(e)
    os.system('rm '+HOME+filename)
    #print(sequence,' ',result)
    return result

#if __name__ == "__main__":
    #res = evaluateHMM(seq)
    #print(res)
