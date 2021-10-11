import numpy as np
from sklearn.metrics import silhouette_score
from rank_based_loss import compute_pairwise_cosine_distances, compute_pairwise_euclidean_distances
import torch


def evaluate_cluster_quality_based_gt_annotations(embeddings, labels_gt, distance = 'cosine'):

    if distance == 'cosine':
        Pairwise_distances_between_samples = compute_pairwise_cosine_distances(torch.tensor(embeddings), full_matrix=True)
        Pairwise_distances_between_samples.fill_diagonal_( 0)
        Pairwise_distances_between_samples[np.where(Pairwise_distances_between_samples<=np.inf)] = 0   
    elif distance == 'euclidean':
        Pairwise_distances_between_samples = compute_pairwise_euclidean_distances(torch.transpose(torch.tensor(embeddings), 0, 1), embeddings.shape[1], embeddings.shape[0], full_matrix=True )    
    
    
    clusters_silhouette_score = silhouette_score(Pairwise_distances_between_samples.detach().numpy(), labels_gt,  metric='precomputed', sample_size=None)
    return clusters_silhouette_score