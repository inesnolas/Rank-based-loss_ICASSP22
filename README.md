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



The values reported are obtained by applying the trained models on a test set consisting of un-used data from the same fine level classes of the training set. 

3 Bird Species Dataset:

|   		        | Sil Fine      | Sil Coarse    | avg Sil       | KNN acc Fine | KNN acc Coarse|
| ---               | ---           | ---	        |---			|---		    |---           |
| **InitEmb**       | 0.01 (0.0)   |	0.21 (0.0)	|0.11 (0.0)	    |  0.74 k=3      | 0.95 k=3  |
| **QuadL**         | -0.17 (-0.18) | 0.31 (+0.1)   |0.07 (-0.04)   |  0.29 k=16   |  0.74 k=4   |
| **RbL**			| -0.08 (-0.09) | 0.42 (+0.21)  |0.17 (+0.06)   |  0.41  k=29   | 0.82 k=23  |
| **RbL_unc**       | -0.22 (-0.23) |	0.22 (+0.01)|0.0 (-0.11)    |  0.29 k=17    |  0.63 k=39 |
| **RbL_RdmHierarchy**| -0.15 (-0.16) | -0.09 (-0.3)|-0.12 (-0.23)  | 0.31 k=32     | 0.33 k=4|



Nsynth Dataset:


|   						 | Sil Fine | Sil Coarse | avg Sil | acc Fine KNN | acc Coarse KNN |
| ---               | ---           | ---	        |---			|---      |---         |
| **InitEmb**       | 0.15 (0.0)    |0.30 (0.0)     | 0.22 (0.0)	    | 0.94 k=1 | 1.0 k=1    |
| **QuadL**         | 0.01 (-0.14)  |0.60 (+0.3)    | 0.31 (+0.09)    | 0.74 k=23 | 0.93 k=26 |
| **RbL**			| -0.08 (-0.23) |0.46 (+0.16)   | 0.19 (-0.03)     | 0.61 k=40 | 0.85 k=18 |
| **RbL_unc**       |-0.16 (-0.31)  |0.38 (+0.08)	| 0.11 (-0.11)   | 0.64 k=33 | 0.84 k=31 |
| **RbL_RdmHierarchy**|0.04 (-0.11) |0.41 (+0.11)   | 0.23 (+0.01)| 0.72 (k=39) | 0.81 (k=39) |


TUTasc2016 Dataset:

|   		            | Sil Fine   | Sil Coarse    | avg Sil          | KNN acc Fine  | KNN acc Coarse|
| ---                   | ---        | ---	          |---			    |---		    |---            |
| **InitEmb**           | 0.21 (0.0)  | 0.26 (0.0)     |0.24 (0.0)      | 0.96 k=1      | 1.0 k=1       | 
| **QuadL**             | -0.19 (-0.40)	| 0.14 (-0.12) | -0.02	(-0.26) | 0.4 k=23      | 0.63 k=23     |     
| **RbL**	            | 0.03 (-0.18) | 0.59 (+ 0.32) | 0.31 (+0.07)   | 0.60 k=38     | 0.87 k=19     |
| **RbL_unc**           |  0.14 (-0.06)	|0.67 (+0.41)  | 0.41 (+0.17)    | 0.4 k=8       | 0.90 k=1      |   
| **RbL_RdmHierarchy**  | -0.23 (-0.44) | -0.04 (-0.30) | -0.14 (-0.38) | 0.33 k=25     |  0.56 k=1   |  


Results on testset consisting on unseen leaf level classes during training:

3 Bird Species Dataset:

|   		        | Sil Fine      | Sil Coarse    | avg Sil       | acc Fine KNN | acc Coarse KNN|
| ---               | ---           | ---	        |---			|---		   |---          |
| **InitEmb**       | 0.17 (0.0)   |	0.39 (0.0)	|0.28 (0.0)	    |   0.0 k=3    | 0.90  k=3    |
| **QuadL**         | -0.07 (-0.24) | 0.57 (+0.18)  |0.25 (-0.03)   |   0.0 k=16   | 0.86 k=4    |  
| **RbL**			| -0.05 (-0.22) | 0.48 (+0.09)  |0.22 (-0.06)   |   0.0 k=29   | 0.80 k=23  |
| **RbL_unc**       | -0.19 (-0.36) |0.32 (-0.07)   |0.07 (-0.21)   |   0.0 k=17   | 0.69 k=39 |
| **RbL_RdmHierarchy**|   -0.16  (-0.33)  |   -0.12 (-0.51)       |      -0.14 (-0.42)      | 0.0 k=32      | 0.31 k=4 |



Nsynth Dataset:


|   						 | Sil Fine     | Sil Coarse      | avg Sil      | acc Fine KNN  | acc Coarse KNN |
| ---                        | ---          | ---	          |---	         |---		     |---         |
| **InitEmb**                |0.18 (0.0)    |	0.14 (0.0)    | 0.16 (0.0)	 |  0.0 k=1     | 0.65 k=1    |
| **QuadL**                  | -0.09 (-0.27)| -0.02 (-0.16)   |	-0.06 (-0.22)|  0.0 k=23    | 0.43 k=26     |
| **RbL**	                 | -0.04 (-0.22)| 0.13 (+0.01)    | 0.05 (-0.11) |  0.0 k=40    | 0.5 k=18 |
| **RbL_unc**                | -0.33 (-0.51)|	0.06 (-0.08)  |-0.14 (-0.30) |  0.0 k=33    | 0.55 k=31 |
| **RbL_RdmHierarchy**       |  -0.1 (-0.28)  | -0.12 (-0.26)	    |     -0.11 (-0.27)     | 0.0 k=39 | 0.21 k=39|       


TUTasc2016 Dataset:


|   		 | Sil Fine   | Sil Coarse    | avg Sil       | acc Fine KNN  | acc Coarse KNN |
| ---        | ---        | ---	          |---			  |---		      |---             |
| **InitEmb**| 0.47 (0.0)  | 0.39 (0.0)    |0.43 (0.0)    |  0.0 k=1      | 0.91 k=1      |      
| **QuadL**  | 0.14 (-0.33)	| 0.27 (-0.12) | 0.21 (-0.22) |  0.0 k=23     | 0.40 k=23     |          
| **RbL**	 | 0.15 (-0.31) | 0.8 (+0.41) | 0.48 (+0.05)  |  0.0 k=38     |  0.97 k=19     |
| **RbL_unc**| 	0.2 (-0.27)	 | 0.74 (+0.35)|  0.47 (+0.03)|  0.0 k=8      |  0.97 k=1      |                    
| **RbL_RdmHierarchy**| -0.02 (-0.49)  | 0.04  (-0.35) | 0.01 (-0.42)  |  0.0 k=25| 0.37 k=1   |                    


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
