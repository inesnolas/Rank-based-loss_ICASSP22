import torch


def compute_pairwise_cosine_distances(minibatch_embeddings, full_matrix=False):
    # cosine_distance = 1 - cosine_similarity
    # cosine similarity (A,B)= cos(theta) =  (A . B ) / (||A||*||B||) , 
    # constrainining embeddings into a hypersphere (unit-sphere) so all norms are 1 reduces this to a matrix multiplication (A.B)

    D = 1 - torch.mm(minibatch_embeddings, torch.transpose(minibatch_embeddings, 0, 1))
    if not full_matrix:
        tri_idx = torch.triu_indices(minibatch_embeddings.shape[0],minibatch_embeddings.shape[0],1)
        pairwise_dist_vector = D[tri_idx[0],tri_idx[1]]
        return pairwise_dist_vector
    else:
        return D


def compute_pairwise_euclidean_distances(minibatch_embeddings, d, n, full_matrix=False ):
    # as per https://www.robots.ox.ac.uk/~albanie/notes/Euclidean_distance_trick.pdf alg.1
    
    X_view1 = minibatch_embeddings.reshape(d, n, 1)   
    X_view2 = minibatch_embeddings.reshape(d,1,n)

    diff_mat = X_view1-X_view2
    D = torch.sum(diff_mat**2,dim=0)
    if not full_matrix:
        tri_idx = torch.triu_indices(n,n,1)
        pairwise_dist_vector = D[tri_idx[0],tri_idx[1]]
        return torch.sqrt(pairwise_dist_vector)
    else :
        return torch.sqrt(D)
