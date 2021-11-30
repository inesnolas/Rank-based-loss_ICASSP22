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
- distance_functions.py: helping functions to compute distances.
- evaluation.py: the evaluation is based on the [Silhouette score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html) metric from sklearn.


```
└── Rank-based-loss_ICASSP22
    ├── example_data
    |   ├── Normalized_VGGish_embeddings_based_on_Training_Set/
    │   ├── val.csv
    │   ├── test.csv
    │   └── train.csv
    ├── loss_fuctions
    │   ├── quadruplet_loss.py
    │   └── rank_based_loss.py
    ├── models
    │   ├── SingleLayer_net.py
    │   ├── pretrained_models/
    ├── utils
    │   ├── data_functions.py
    │   ├── evaluation.py
    │   └── distance_functions.py
    ├── LICENCE.md
    ├── README.md
    ├── requirements.txt
    └── run_example.py
 ```

### Computing the rank based loss:

1. Compute a rank map from the tree of ground truth labels: Each pair of examples has a rank given by the tree distance of their labels. The tree distance is given by the number of nodes that separate the two labels
2. Compute all the pairwise cosine distances in the batch in the embedding space, and sort them. 
3. For each rank, assign a target distance by selecting whatever distance in the sorted distances vector falls at each rank.
4. Compute **Ip** as: 0 if distance of the pair is within the correct positions in the sorted distances vector, else 1 if distance of the pair is wrong given the ground truth rank.
5. Compute the loss following equation above.



An example from the task of individual identification of animals: we assume the following hirarchical label structure:
![hierarchical_labels_tree](https://user-images.githubusercontent.com/33712250/137140261-5ad84e7f-1d31-4f95-8501-dd105c7b6439.png)
given a minibatch of examples, we can compute the RbL by following the 5 steps above:
![compute_RBL_example](https://user-images.githubusercontent.com/33712250/137123161-1b7c4eef-9b5e-4d79-bec5-d3892bec2382.png)


### Some results:

<!-- 
![g4179](https://user-images.githubusercontent.com/33712250/137938178-fd05a7ea-636a-46f1-8891-fddf936d7160.png)
 -->


**InitEmb** Evaluation of initial pretrained embeddings: This serves the purpose of defining a baseline for comparison with all other experiments. The main goal is to understand if an improvement over the initial embeddings is achieved or not. % the purpose is to understand if the "quality" of the initial embeddings impacts the performance of the Rank based loss.

**QuadL** Comparison against Quadruplet Loss: The quadruplet loss of  [] is especially relevant to compare with the rank based loss. As  mentioned in sec.\ref{sec:intro}, this loss integrates hierarchical information through selection of the examples that generate the quadruplets.

**RbL** Rank based loss: Training the network with the RbL on the 3 datasets. batch size is 12 and the examples are selected to create a balanced batch across ranks.

**RbL_unc** Unconstrained batches: On all the previous experiments the batches are balanced regarding the hierarchical relationships between ground truth labels. Here that constraint is lifted, allowing, for example, batches to be missing pairs of one rank or have a disproportional number of pairs in another.

**RbL_RdmHierarchy** By randomly re-arranging the hierarchy of the problem we can evaluate if the RbL is indeed performing better than non-hierarchical approaches **(InitEmb)** due to using the hierarchical relationship information or simply because the problem at each level becomes smaller and thus easier to solve (like a divide and conquer approach).  We expect if the first case, that the Sil values obtained on this randomized label structure to be very bad, on the other hand, if it is a purely divide and conquer approach that is giving the improved scores then we would expect a Sil values to stay at the same values as the experiments in **RbL**





3 Bird Species Dataset:

|   		        | Sil Fine      | Sil Coarse    | avg Sil       | acc Fine KNN k=17 | acc Coarse KNN k=17|
| ---               | ---           | ---	        |---			|---		        |---                 |
| **InitEmb**       | -0.23 (0.0)   |	0.31 (0.0)	|0.04 (0.0)	    |   |     |
| **QuadL**         | -0.17 (+0.06) | 0.31 (0.0)    |0.07 (+0.03)   |   |   |
| **RbL**			| -0.08 (+0.15) | 0.42 (+0.11)  |0.17 (+0.13)   | 0.46              | 0.85               |
| **RbL_unc**       | -0.22 (+0.01) |	0.22 (-0.09)|0.0 (+0.04)    |   |   |
| **RbL_RdmHierarchy**| -0.15 (+0.08) | -0.09 (-0.41)| -0.12 (-0.16)|   |   |



Nsynth Dataset:


|   						 | Sil Fine | Sil Coarse | avg Sil | acc Fine KNN k=17 | acc Coarse KNN k=17|
| ---               | ---           | ---	        |---			|---		        |---         |
| **InitEmb**| -0.04 (0.0)|	0.65 (0.0) | 0.31 (0.0)	|
| **QuadL**  | 0.01 (+0.05)| 0.6 (-0.05)|	0.31 (0.0)|
| **RbL**				| -0.08 (-0.04)| 0.46 (-0.19)| 0.19 (-0.12) | 0.62 | 0.86 |
| **RbL_unc**|-0.16 (-0.12) |	0.38 (-0.27)	|0.11 (-0.2) |
| **RbL_RdmHierarchy**| -0.13 (-0.09) | -0.08 (-0.57)	 |	-0.11 (-0.42)|


TUTasc2016 Dataset:


|   		 | Sil Fine   | Sil Coarse    | avg Sil       | acc Fine KNN k=17 | acc Coarse KNN k=17|
| ---        | ---        | ---	          |---			  |---		          |---                    |
| **InitEmb**| 0.3 (0.0)  | 0.57 (0.0)    |0.43 (0.0)     |                   |                    |                    
| **QuadL**  | -0.19 (-0.5)	| 0.14 (-0.43) | -0.02	(-0.45) |                 |                   |                    
| **RbL**	 | 0.03 (-0.27) | 0.59 (+ 0.02) | 0.31 (-0.12) | 0.53             | 0.87                 |
| **RbL_unc**|  0.14 (-0.15)	|	0.67 (+0.1)	 | 0.41 (-0.02)|    |                   |                    
| **RbL_RdmHierarchy**| -0.32 (-0.62) | -0.1 (-0.67)	 |	 -0.21 (-0.64)| |                   |                    



<!-- #### embeddings visualization:  -->


<!-- Visualization of the embeddings obtained by RbL and how these evolve during training:
Colored by animal ID
https://user-images.githubusercontent.com/33712250/137915325-0f795074-a716-47dc-a1ce-ac9fc56aa3df.mp4

Colored by species:
https://user-images.githubusercontent.com/33712250/137915587-f5ba2418-731f-4bd4-b7e3-474cc239468b.mp4 -->


<!-- 3) how does it relate with Quadruplet loss?
4) rbl uncontrained, does it show more flexibility regarding the examples in the batch?
5) 

 -->
