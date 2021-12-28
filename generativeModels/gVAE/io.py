def output_fasta(names, seqs, filepath):
    with open(filepath, 'w') as fout:
        for name, seq in zip(names, seqs):
            fout.write('>{}\n'.format(name))
            fout.write(seq+'\n')