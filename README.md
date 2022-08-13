# Acute Myeloid Leukemia Risk Group Prediction from Gene Expression Data with Feed-Forward Neural Networks

Abstract will be added on a later date. This repository includes the following files

## config_set.txt
* 100 randomly chosen network configurations
* *layers*: number of hidden layers in the network
* *units*: number of hidden units in each hidden layer
* *drop*: rate for randomly removing nodes in each layer
* *lr*: the learning rate
* *batch_size*: number of training samples for each epoch


## data_preparation.R
* New variable creation
* Division to training and test sets

## feature_selection.R
* RFE to reduce dimensions

## important_genes.txt
* List of 109 important genes found with RFE

## laml_data.zip
* Zip file including the gene expression data (laml.csv)

## laml_tuning.R
* Oversampling with SMOTE and ADASYN
* Tuning the network

## laml_tuning_rfe.R
* Feature selection with RFE
* Oversampling with SMOTE and ADASYN
* Tuning the network
