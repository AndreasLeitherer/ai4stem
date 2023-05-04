# ai4stem
This repository contains AI-STEM (Artificial-Intelligence-based Scanning Transmission Electron Microscopy), an Artificial-Intelligence tool for the characterization of atomic-resolution electron microscopy images. 

This repository is under active development (especially the folders 'ai4stem/descriptors' and ai4stem/augmentation') but already contains all the relevant code for reproducing the results of the manuscript

    A. Leitherer, B.C. Yeo, C. H. Liebscher, and L. M. Ghiringhelli.     
    "Automatic Identification of Crystal Structures and Interfaces via Artificial-Intelligence-based Electron Microscopy" 
    arXiv:2303.12702 (2023) https://doi.org/10.48550/arXiv.2303.12702

For several examples on how to conduct analysis with ai4stem, the scripts and notebooks folders in this repository should be a good starting point.

Alternatively, you may run notebooks on google colab:

[AI-STEM quickstart (supervised learning)](https://colab.research.google.com/github/AndreasLeitherer/ai4stem/blob/main/notebooks/Application_of_pretrained_model.ipynb)

[AI-STEM unsupervised analysis](https://colab.research.google.com/github/AndreasLeitherer/ai4stem/blob/main/notebooks/Unsupervised_analysis_example.ipynb)

------------------
### Installation
------------------

We recommend to create a virtual python >=3.7 environment 
(for instance, with conda: https://docs.anaconda.com/anaconda/install/linux/), and then execute

    git clone https://github.com/AndreasLeitherer/ai4stem.git
    git cd ai4stem
    pip install -e .

Moreover, several requirements are listed in the requirements text document and can be easily installed via

    pip install -r requirements.txt
