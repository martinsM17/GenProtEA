# GenProtEA
Framework implementing deep learning generative models for novel enzyme design

This endeavour was developed in the context of a Master's Thesis at University of Minho 

# Deep learning generative models for novel enzyme design.

The objective of this work was to develop a platform to generate new protein sequences, using different deep learning architectures and, 
through the use of evolutionary computation, steer the generative process towards determined optimization objectives. 

The work leveraged two previous implementations:

GAN - ProteoGAN (https://github.com/proteogan/proteogan.git)

VAE - deep-protein-generation (https://github.com/alex-hh/deep-protein-generation.git)

To run each architecture throughout all the experiments, two conda environment must be created (one for each deep learning architecture)

conda env create --file environment_gan.yml

conda env create --file environment_vae.yml

# Authors

- Miguel Martins (Msc Bioinformatics student at University of Minho)
- Miguel Rocha (University of Minho)
- Vítor Pereira (University of Minho)
