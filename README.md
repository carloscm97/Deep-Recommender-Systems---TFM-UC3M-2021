# Recommender System in PyTorch

Implementations of various top-N recommender systems in [PyTorch](pytorch.org) for practice.

Dataset:
- [Movielens](https://grouplens.org/datasets/movielens/) 1M.
- [Last.fm](http://ocelma.net/MusicRecommendationDataset/lastfm-360K.html) 360K.
- [Amazon Books](http://jmcauley.ucsd.edu/data/amazon/index_2014.html) 22M.

Last two datasets have been pre-processed so that a conventional computer can work with them.
 
## Available models
| Model    | Paper                                                                                                                                          |
|------------------|------------------------------------------------------------------------------------------------------------------------------------------------|
| BPRMF            | Steffen Rendle et al., BPR: Bayesian Personalized Ranking from Implicit Feedback. UAI 2009. [Link](https://arxiv.org/pdf/1205.2618) |
| ItemKNN          | Jun Wang et al., Unifying user-based and item-based collaborative filtering approaches by similarity fusion. SIGIR 2006. [Link](http://web4.cs.ucl.ac.uk/staff/jun.wang/papers/2006-sigir06-unifycf.pdf) |
| DAE, CDAE        | Yao Wu et al., Collaborative denoising auto-encoders for top-n recommender systems. WSDM 2016.[Link](https://alicezheng.org/papers/wsdm16-cdae.pdf) |
| MultVAE          | Dawen Liang et al., Variational Autoencoders for Collaborative Filtering. WWW 2018. [Link](https://arxiv.org/pdf/1802.05814) |

## How to run
1. Open Notebook ```main.ipynb```. This file is an example of execution on Google Colab.
2. The code is intended to launch on Google Colab, but if you want to launch locally ignore Google Drive mount lines.
3. Select between BPRMF, ItemKNN, DAE, CDAE and MultVAE and pass it as argument to model_exec.
4. Run model_exec initialize() method with 'ml-1m', 'lastfm' or 'books' argument depending on the dataset you want your model being trained.
5. Run model_exec.run() for training, the statistical data will be saved in 'saves' folder, with the name of the algorithm + day and hour of the execution.
6. Finally, we have an example of code to represent and compare the results of each model saved lately.

In 'config' folder you can change parameters of each model.

# Reference
Code base:
- RecSys_PyTorch. (yoongi0428) [Repository](https://github.com/yoongi0428/RecSys_PyTorch).
Some model implementations and util functions refers to these nice repositories.
- NeuRec: An open source neural recommender library. [Repository](https://github.com/wubinzzu/NeuRec)
- RecSys 2019 - DeepLearning RS Evaluation. [Paper](https://arxiv.org/pdf/1907.06902) [Repository](https://github.com/MaurizioFD/RecSys2019_DeepLearning_Evaluation)