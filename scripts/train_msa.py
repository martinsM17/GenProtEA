import tensorflow as tf

import argparse
import numpy as np
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
import sys
sys.path.append('/home/mmartins/GenProtEA')
from generativeModels.gVAE.vaes import MSAVAE
from utils.io import load_gzdata, read_fasta
from utils.data_loaders import one_hot_generator

# Define training parameters
batch_size = 32
seed = 0
n_epochs = 20
verbose = 1
save_all_epochs = False

seed = np.random.seed(seed)

# Load aligned sequences
#_, msa_seqs = read_fasta('data/training_data/new_aligned_train.fasta')
#_, val_msa_seqs = read_fasta('data/training_data/new_aligned_test.fasta')
_, msa_seqs = load_gzdata('data/training_data/seqs_msavae_train.fasta.gz', one_hot=False)
_, val_msa_seqs = load_gzdata('data/training_data/seqs_msavae_test.fasta.gz', one_hot=False)
#_, msa_seqs = load_gzdata('data/training_data/luxafilt_llmsa_train.fa.gz', one_hot=False)
#_, val_msa_seqs = load_gzdata('data/training_data/luxafilt_llmsa_val.fa.gz', one_hot=False)
# Define data generators
train_gen = one_hot_generator(msa_seqs, padding=None)
val_gen = one_hot_generator(val_msa_seqs, padding=None)

# Define model
print('Building model')
model = MSAVAE(original_dim=1530, latent_dim=50)

# (Optionally) define callbacks
callbacks=[CSVLogger('output/logs/msavae.csv')]
callbacks.append(EarlyStopping(monitor='loss', patience=3))
if save_all_epochs:
    callbacks.append(ModelCheckpoint('output/weights/msavae'+'.{epoch:02d}-{luxa_errors_mean:.2f}.hdf5',
                                     save_best_only=False, verbose=1))

print('Training model')
# Train model https://github.com/keras-team/keras/issues/8595
model.VAE.fit_generator(generator=train_gen,
                        steps_per_epoch=len(msa_seqs) // batch_size,
                        verbose=verbose,
                        epochs=n_epochs,
                        validation_data=val_gen,
                        validation_steps=len(val_msa_seqs) // batch_size,
                        callbacks=callbacks)

if not save_all_epochs:
  model.save_weights('output/weights/msavae.h5')
