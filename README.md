# Adversarial Learning for Modeling Human Motion

This repository contains the open source code which reproduces the results for the paper: [Adversarial learning for modeling human motion](https://link.springer.com/article/10.1007%2Fs00371-018-1594-7). The authors of this paper are: Qi Wang, Thierry Artières, Mickael Chen, and Ludovic Denoyer.

# How to Reproduce the Results

1) For download the EMILYA dataset, you should contact the owner [Catherine Pelachaud](http://pages.isir.upmc.fr/~pelachaud/).

2) Clone this repository code to your computer and rename the root folder as "Seq_AAE_V1" . 

3) Install the relevant packages

   * Keras
   * matlotlib
   
4) Data Preprocessing: You should save the motions in EmilyaDataset into a npz file according to the readme.md file in the folder 'datasets/EmilyaDaset/'.

5) For training the models in the paper, you should navigate to the 
"\Training" directory under the root directory of the project and there you can find the following five folders:

Conditional\_SAAE				

Seq\_AAE 

Double\_GAN\_Continuous\_Emotion\_Representation

Seq\_VAE

Double\_Gan\_Condition\_SAAE 

These folders are named by the model's name. Enter each of the folders, you can find the training file and evaluation scripts in a subfolder named '\evaluation'. By running the training file in terminal, you can train the model from scratch. If you want to tune the hyperparameters, you can directly modified them in the training file.


  
