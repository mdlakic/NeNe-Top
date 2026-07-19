
# Overview

NeNe-Top - <ins>Ne</ins>ural <ins>Ne</ins>twork for <ins>T</ins>emperature <ins>O</ins>ptimum <ins>P</ins>rediction

## ✨ Project goals
Microorganisms live and grow across a wide range of temperatures, and this project provides a way to predict the temperature at which they grow best. Measuring optimal growth temperatures in the lab is slow and, in many cases, impossible when microbes cannot be grown in pure cultures. This work builds on previous studies, which showed that using protein sequence information is a viable way to predict optimal growth temperatures. We gathered a large, carefully chosen set of complete genomes and metagenomes reconstructed from environmental samples. Next, we tested several modern machine learning methods, including neural networks. Our machine learning tools make predictions that match real lab results closely and work for many kinds of microbes. In addition, the predictions are fast and scale to large datasets, thus reducing the need for time-consuming experiments.

## 📚 References

NeNe-Top is described in:

Dlakic, M and Inskeep, WP (2026) [Improved prediction of microbial optimal growth temperatures with neural networks and protein language models.](https://www.frontiersin.org/journals/genetics/articles/10.3389/fgene.2026.1874451) *Front. Genet. 17:* 1874451.

If you find this work useful, please also cite the following papers:

* Sato Y, Okano K, Kimura H, Honda K. 2020. [TEMPURA: Database of Growth TEMPeratures of Usual and RAre Prokaryotes.](https://pubmed.ncbi.nlm.nih.gov/32727974/) <i>Microbes Environ.</i>, 35(3).
* Elnaggar A, et al. 2022. [ProtTrans: Toward Understanding the Language of Life Through Self-Supervised Learning.](https://pubmed.ncbi.nlm.nih.gov/34232869/) <i>IEEE Trans Pattern Anal Mach Intell.</i>, 44: 7112-7127.

## 🚀 Installation and testing

Please install [conda/mamba](https://conda-forge.org/miniforge/).

There are three shell scripts in this repository that will install a dedicated environment called `NeNe`, depending on your preferences.

1. install_no_ProtT5.sh - to install an environment without a ProtT5 protein language model  
2. install_with_ProtT5.sh - to install an environment with a ProtT5 protein language model  
3. install_with_ProtT5_CPU.sh - to install an environment with ProtT5 and CPU (no GPU)

Using ProtT5 without GPU will be very slow, and we suggest that you install ProtT5 only on a computer with GPU (option #2).

First, clone this repository and change into the main directory. Assuming that conda is installed, simply run:  
```
bash install_with_ProtT5.sh
OR
source install_with_ProtT5.sh
```

This procedure will create a NeNe environment, activate it, and install all the required packages. This procedure is done only once. After that, typing `conda deactivate` will exit this environment, and typing `conda activate NeNe` will enter it again.

After the installation process is complete, we suggest that you test neural network predictions. The only requirement is to have a FASTA file with all proteins from a given (meta)genome. There are two example files in this repository (`SulfMK5.faa` and `pyroWP30.faa`).

To make a prediction using dipeptide frequencies:  
```
python NeNe-Top-DIpep.py SulfMK5.faa
```

The screen output should be as follows:

```
Fold 1: 0 hours 0 minutes and 0.47 seconds.
 Fold 2: 0 hours 0 minutes and 0.34 seconds.
 Fold 3: 0 hours 0 minutes and 0.26 seconds.
 Fold 4: 0 hours 0 minutes and 0.25 seconds.
 Fold 5: 0 hours 0 minutes and 0.26 seconds.
 Fold 6: 0 hours 0 minutes and 0.26 seconds.
 Fold 7: 0 hours 0 minutes and 0.26 seconds.
 Fold 8: 0 hours 0 minutes and 0.26 seconds.
 Fold 9: 0 hours 0 minutes and 0.39 seconds.
 Fold 10: 0 hours 0 minutes and 0.26 seconds.
 Complete prediction: 0 hours 0 minutes and 3.0 seconds.
 10-fold average NN prediction for SulfMK5_DIpep: 69.08

  !!! As the NN prediction is above 45 degrees, we are making a high temperature-based prediction with NNs !!!
 Fold 1: 0 hours 0 minutes and 0.28 seconds.
 Fold 2: 0 hours 0 minutes and 0.27 seconds.
 Fold 3: 0 hours 0 minutes and 0.28 seconds.
 Fold 4: 0 hours 0 minutes and 0.43 seconds.
 Fold 5: 0 hours 0 minutes and 0.27 seconds.
 Fold 6: 0 hours 0 minutes and 0.27 seconds.
 Fold 7: 0 hours 0 minutes and 0.28 seconds.
 Fold 8: 0 hours 0 minutes and 0.27 seconds.
 Fold 9: 0 hours 0 minutes and 0.28 seconds.
 Fold 10: 0 hours 0 minutes and 0.43 seconds.
 Complete prediction: 0 hours 0 minutes and 3.06 seconds.

 10-fold average NN HT prediction for SulfMK5_DIpep_HT: 70.06
```

For predictions larger than 45 <sup>o</sup>C, a second prediction will be made automatically from a model that was trained only on high-temperature data. See our paper for details.

To use our best model based on ProtT5, first we must create the embeddings. This is demonstrated on the second proteome file in the same directory:  

```
python NeNe-ProtT5-XL-U50-embedding.py pyroWP30.faa
```

This will take anywhere from 1 to 10 minutes, depending on GPU speed and memory. A file named `pyroWP30_pLM.csv` will be made, which is used for optimal temperature prediction:  

```
python NeNe-Top-ProtT5.py pyroWP30_pLM.csv

 Fold 1: 0 hours 0 minutes and 0.44 seconds.
 Fold 2: 0 hours 0 minutes and 0.3 seconds.
 Fold 3: 0 hours 0 minutes and 0.3 seconds.
 Fold 4: 0 hours 0 minutes and 0.4 seconds.
 Fold 5: 0 hours 0 minutes and 0.3 seconds.
 Fold 6: 0 hours 0 minutes and 0.3 seconds.
 Fold 7: 0 hours 0 minutes and 0.3 seconds.
 Fold 8: 0 hours 0 minutes and 0.3 seconds.
 Fold 9: 0 hours 0 minutes and 0.3 seconds.
 Fold 10: 0 hours 0 minutes and 0.43 seconds.
 Complete prediction: 0 hours 0 minutes and 3.37 seconds.
 10-fold average prediction for pyroWP30_pLM_ProtT5: 90.41

 !!! As the general prediction is above 45 degrees, we are making a high temperature-based prediction with pLM !!!
 Fold 1: 0 hours 0 minutes and 0.34 seconds.
 Fold 2: 0 hours 0 minutes and 0.31 seconds.
 Fold 3: 0 hours 0 minutes and 0.3 seconds.
 Fold 4: 0 hours 0 minutes and 0.3 seconds.
 Fold 5: 0 hours 0 minutes and 0.44 seconds.
 Fold 6: 0 hours 0 minutes and 0.31 seconds.
 Fold 7: 0 hours 0 minutes and 0.31 seconds.
 Fold 8: 0 hours 0 minutes and 0.3 seconds.
 Fold 9: 0 hours 0 minutes and 0.3 seconds.
 Fold 10: 0 hours 0 minutes and 0.31 seconds.
 Complete prediction: 0 hours 0 minutes and 3.23 seconds.
 10-fold average prediction for pyroWP30_pLM_ProtT5_HT: 87.42
```

Again, we have a thermophilic organism with optimal growth temperature larger than 45 <sup>o</sup>C, so a second prediction will be made automatically from a model that was trained only on high-temperature data. Compare the results obtained on your computer with files in the `expected_output` folder.

## 📈 Performance

The measurements shown below are averages from a dataset with 1430 (meta)genomes. All results were obtained by running a single CPU on an AMD Ryzen 9 9950X 16-Core Processor with 192 GB RAM. Protein language model embeddings were made using an NVIDIA GeForce RTX 4090 card with 24 GB video memory.

| Model | Average time per genome |
|---------------|---------------|
| **Dipeptide NN** | 5.5 sec |
| **ProtT5 (embedding only)** | 2 min 47 sec |
| **ProtT5 NN (prediction only)** | 4.8 sec |

## © Copyright

Copyright 2026 Mensur Dlakic. See LICENSE for further details.
