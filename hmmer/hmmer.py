import pyhmmer
import pyfasta

seqs=[]

f = pyfasta.Fasta("aln-fasta.fasta")
for header in f.keys():
    name = str(header)
    seqt = str(f[header])
    seq = pyhmmer.easel.TextSequence(name=bytes(name, 'ascii'),sequence=seqt)
    seqs.append(seq)

msa  = pyhmmer.easel.TextMSA(bytes('msa', 'ascii'), sequences=seqs)
alphabet = pyhmmer.easel.Alphabet.amino()
msa_d = msa.digitize(alphabet)

builder = pyhmmer.plan7.Builder(alphabet)
background = pyhmmer.plan7.Background(alphabet)
hmm, _, _ = builder.build_msa(msa_d, background)


print('consensus:',hmm.consensus)
print('cutoffs',hmm.cutoffs)


with open("IPR000390.hmm", "wb") as output_file:
    hmm.write(output_file)


