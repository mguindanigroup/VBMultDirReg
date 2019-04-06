# Variational Bayes  Dirichlet Multinomial model using non-local prior spike-and-slab regression.
This repository contains a tensorflow implementation of variational Bayes approach to variable selection using  
 Dirichlet Multinomial model. We demonstrate both simulation and real data experiments in this repository. 

## Introduction 

### Dirchlet Multinomial Model
Suppose we have multivariate counts  as response  data: ![equation](https://latex.codecogs.com/gif.latex?%24%5Cmathbf%7By%7D_%7Bi%7D%20%3D%20%28y_%7Bi1%7D%2C%20%5Cdots%2C%20y_%7BiJ%7D%29%24), e.g. microbiome count data.   
We model it using multinomial distribution with parameter  ![equation](https://latex.codecogs.com/svg.latex?%24%5Cmathbf%7B%5Cphi%7D_%7Bi%7D%20%3D%20%28%5Cphi_%7Bi1%7D%2C%20%5Cdots%2C%20%5Cphi_%7BiJ%7D%29%24). Further, to account for the overdispersion of the response,
it is natural to assume:
 
 ![equation](https://latex.codecogs.com/svg.latex?%24%5Cmathbf%7B%5Cphi%7D_%7Bi%7D%20%3D%20%28%5Cphi_%7Bi1%7D%2C%20%5Cdots%2C%20%5Cphi_%7BiJ%7D%29%20%5Csim%20Dirchlet%28%5Cmathbf%7B%5Cxi%7D_%7Bi%7D%20%29%24)





## Dependencies:


numpy  1.16.2

scipy 1.2.1

pandas 0.24.2

sklearn 0.20.3

matplotlib 3.0.3

[TensorFlow: 1.13.1](https://www.tensorflow.org/install)

[networkx 2.2](https://networkx.github.io/documentation/stable/install.html)

You can either download and unzip the files or git clone the project on your local machine. Then it is recommended to use [Pycharm](https://www.jetbrains.com/pycharm/download/#section=mac), which gives a convenient one-stop solution to install all the 
dependencies within this project. After open the project in Pycharm, find the project interpreter and add the above packages into the project's 
environment. 

## Running examples

**Simulations**

For p=q=50, run 50 repeated experiments. Data generating mechanism follows previous [work](https://github.com/duncanwadsworth/dmbvs) on Dirichlet Multinomial model. 
The R code for generating data is in ./data/R-code-simulate-data.R. Running the R code will generate covariate as well as response data and write them into [h5 file](https://en.wikipedia.org/wiki/Hierarchical_Data_Format). 
We also provide simulated data file in ./data/rep_50_001.h5 and rep_50_01.h5 for overdispersion 0.01 and 0.1 respectively. 

To start running and to print [precision, recall, MCC, AUC, F1, ACC] after finishing each repeated experiments on simulated data, type:


```
python train-DirMulVI.py 
```

In addition, you can find visualized results of the running example in the ipython notebook provided: non-local-prior-DMVS-SimulationExp.ipynb.
It includes ROC-AUC plot comparing to other methods. 

**Real Data Experiment**

We apply our variational method with non-local prior to a human gut microbiome
dataset, which has been previously used in Wu et al., 2011 to investigate the association of dietary
and environmental variables with the gut microbiota. 

To run the experiment, type:
```bash
python train-VI-RealData.py

```
It will return a bipartite graph indicating the selected association based on false discovery rate of 0.1. 



