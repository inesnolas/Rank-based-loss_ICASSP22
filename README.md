# Rank-based-loss
### Intro

In [Ines Nolasco and Dan Stowell. “Rank-based loss for learning hierarchical representations.” (2021).](https://arxiv.org/abs/2110.05941), we proposed the Rank based loss function (RbL) to learn embeddings that express hierarchical relationships.

![Rbl formulation](https://user-images.githubusercontent.com/33712250/137119430-3b18ca80-4e1d-4ef1-9454-8ab57c272842.png)

RbL follows a metric learning approach, where the objective is to learn embeddings in which the distances between them are meaningful to the problem being addressed. At each iteration of training we want to evaluate the distances between embeddings and push the embeddings closer or further away depending on the hierarchical relationship between labels. The hierarchical information is used here to define, for each element, the desired rank-ordering of all other elements in terms of their distance.

This repo contains:    
- **Rank_based_loss.py**: all the necessary functions to compute the RbL and a function to train a network with this loss function.
- **Quadruplet_loss.py**: in the paper we compare RbL to a quadruplet loss from [A. Jati, N. Kumar, R. Chen and P. Georgiou, "Hierarchy-aware Loss Function on a Tree Structured Label Space for Audio Event Detection," ICASSP 2019](https://ieeexplore.ieee.org/document/8682341), this is our implementation of that function.
- run_example.py: script that trains a netwwork with RbL and generates embeddings for a test set.
- example_data: necessary data to run example. data are precomputed Vggish embeddings and are normalized based on the training set. (see paper for more details)
- SingleLayer_net.py: Network architecture class.
- data_functions.py: Dataset class.
- evaluation.py: the evaluation is based on the Silhouette score metric from sklearn: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html

### Computing the rank based loss:

### Some results:

