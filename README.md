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
- evaluation.py: the evaluation is based on the [Silhouette score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html) metric from sklearn: 

### Computing the rank based loss:

1. Compute a rank map from the tree of ground truth labels: Each pair of examples has a rank given by the tree distance of their labels. The tree distance is given by the number of nodes that separate the two labels
2. Compute all the pairwise cosine distances in the batch in the embedding space, and sort them. 
3. For each rank, assign a target distance by selecting whatever distance in the sorted distances vector falls at each rank.
4. Compute **Ip** as: 0 if distance of the pair is within the correct positions in the sorted distances vector, else 1 if distance of the pair is wrong given the ground truth rank.
5. Compute the loss following equation above.

An example from the task of individual identification of animals: we assume the following hirarchical label structure:
![hierarchical_labels_tree](https://user-images.githubusercontent.com/33712250/137140261-5ad84e7f-1d31-4f95-8501-dd105c7b6439.png)
given a batch of examples, we can compute the RbL by following the 5 steps above:
![compute_RBL_example](https://user-images.githubusercontent.com/33712250/137123161-1b7c4eef-9b5e-4d79-bec5-d3892bec2382.png)


### Some results:

1) does it converge? and does that mean identifiable clusters? (RbL experiments with scatter plots of the embeddings)
2) how does it relate with Quadruplet loss?
3) rbl uncontrained, does it show more flexibility regarding the examples in the batch?
4) 

