NeNe-Top - Neural Network for Temperature Optimum Prediction  

Please install conda/mamba first. They are available:  
https://conda-forge.org/miniforge/  
There are three shell scripts in this repository that will install a dedicated environment called “NeNe” depending on your preferences.  

install_no_ProtT5.sh		- to install an environment without ProtT5  
install_with_ProtT5.sh	- to install an environment with ProtT5  
install_with_ProtT5_CPU.sh	- to install an environment with ProtT5 and CPU (no GPU)  

The last option will be very slow, and we suggest that you install ProtT5 on a computer with GPU (option #2).  

Assuming you have conda installed, simply run:  
```
bash install_with_ProtT5.sh
OR
source install_with_ProtT5.sh
```

This procedure will create a NeNe environment, activate it, and install all the required packages.  

When that process is complete, we suggest that you try your first neural network prediction:  
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

 !!! As the NN prediction is above 45 degrees, we are making a high temperature-based prediction with SVR !!!
 Fold 1: 0 hours 0 minutes and 0.0 seconds.
 Fold 2: 0 hours 0 minutes and 0.0 seconds.
 Fold 3: 0 hours 0 minutes and 0.0 seconds.
 Fold 4: 0 hours 0 minutes and 0.0 seconds.
 Fold 5: 0 hours 0 minutes and 0.0 seconds.
 Fold 6: 0 hours 0 minutes and 0.0 seconds.
 Fold 7: 0 hours 0 minutes and 0.0 seconds.
 Fold 8: 0 hours 0 minutes and 0.0 seconds.
 Fold 9: 0 hours 0 minutes and 0.0 seconds.
 Fold 10: 0 hours 0 minutes and 0.0 seconds.
 Complete prediction: 0 hours 0 minutes and 0.01 seconds.
 10-fold average SVR prediction for SulfMK5_SVR: 69.16
```

For predictions larger than 45 oC, a second prediction will be made automatically from a model that was train only on high-temperature data.  

If you wish to use our best model based on ProtT5 language models, first we must create the embeddings. This is demonstrated on the second proteome file in the same directory:  

```
python NeNe-ProtT5-XL-U50-embedding.py pyroWP30.faa
```

This will take anywhere from 1 to 10 minutes, depending on your GPU speed and memory. A file named “pyroWP30_pLM.csv” will be made, which is used for optimal temperature prediction:  

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

Again we have a thermophilic organism with optimal growth temperature larger than 45 oC, so a second prediction will be made automatically from a model that was trained only on high-temperature data.  

More details will be posted here when the manuscript is accepted for publication.  
