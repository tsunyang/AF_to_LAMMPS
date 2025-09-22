# AF_to_LAMMPS
This Python-based repository contains code and examples for converting Alphafold predictions and stuctural data into amino-acid level coarse-grained polymer models in LAMMPS. The methodology preserves the order and disorder regions of simulated protein chains.

# Required Downloads
To generate elastic networks from B-factor calculations, we make use of FitNMA described by Koehl, Orland, and Delarue (doi: 10.1002/jcc.26701) which can be downloaded [here](https://www.cs.ucdavis.edu/~koehl/Projects/index.html)

To fit chains into a simulation box, we make use of Packmol described by Martinez et al. (doi: 10.1002/jcc.21224) which can be downloaded [here](https://m3g.github.io/packmol/ciA%20package%20for%20building%20initial%20configurations%20for%20molecular%20dynamics%20simulations.%20Jtation.shtml)

Simulations are conducted in LAMMPS which can be downloaded [here](https://www.lammps.org/download.html). For these coarse-grain simulations, the forcefield developed by Joseph et al. (doi: 10.1038/s43588-021-00155-3) is used, which requires a version of LAMMPS that includes Wang/Frenkel pair style. Make sure LAMMPS is installed with the packages: EXTRA-DUMP, EXTRA-FIX, EXTRA-MOLECULE, EXTRA-PAIR, MOLECULE, and RIGID.

Additionally, several modules are downloaded in Python for the example. These include, jupyter, numpy, pandas, sys, scikit-learn, copy, json, re, scipy, biopython, itertools, Prody, matplotlib.

# Steps in Example
This example is found in the walkthrough folder, and makes use of jupyter notebook

1. Setup Conda environment.
- Create conda environment via `conda create -n testing` and activate via `conda activate testing`
- Install relevant Python modules `conda install jupyter` `conda install python` `conda install pandas` `conda install scikit-learn` `conda install conda-forge::biopython` `conda install ProDy` and `conda install matplotlib`, and `conda install -c conda-forge freud`

2. Open and load modules in Jupyter.
- Open jupyter notebook via `jupyter notebook`. Click on the jupyter notebook  gen_Vts1.ipynb in the walkthrough folder and follow the directions therein.
