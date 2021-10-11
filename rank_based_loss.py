
import torch
import itertools
import matplotlib as plt
from torch.nn.functional import embedding
import numpy as np
import wandb
import evaluation as ev
import itertools
import tqdm

def compute_pair_gt_tree_distance(labels_pair_one, labels_pair_two, number_of_ranks ):  
    # TODO generalize function for n levels in hierarchy, hardcoded 2 and 3 levels only
    if  len(labels_pair_one.split('_')) == 3:
        labels_one_expanded = [labels_pair_one, labels_pair_one.split('_')[1]+'_'+labels_pair_one.split('_')[2], labels_pair_one.split('_')[2] ]
        labels_two_expanded = [labels_pair_two, labels_pair_two.split('_')[1]+'_'+labels_pair_two.split('_')[2], labels_pair_two.split('_')[2] ]
        rank = 0
        i = 0
        while i <= len(labels_one_expanded):
            if labels_one_expanded[i] == labels_two_expanded[i] :
                return 2*i
                
            else:
                i +=1
        rank = 6
    elif len(labels_pair_one.split('_'))== 2:
        labels_one_expanded = [labels_pair_one, labels_pair_one.split('_')[-1] ]
        labels_two_expanded = [labels_pair_two, labels_pair_two.split('_')[-1]]
        rank = 0
        i = 0
        while i <= len(labels_one_expanded)-1:
            if labels_one_expanded[i] == labels_two_expanded[i] :
                return 2*i
                
            else:
                i +=1
        rank = 4

    return rank 

def pre_compute_rank_map(number_of_ranks, hierarchical_labels):
    # computes rank map for the whole dataset?.
    pairs_indexes = torch.tensor(list(itertools.combinations(range(len(hierarchical_labels)),2)))
    number_pairs = pairs_indexes.shape[0]
    pair_rank = torch.tensor(list(range(number_pairs)))*-1
    rank_map = torch.zeros((number_of_ranks+1, number_pairs) )
    for pair_index, (idx1, idx2) in enumerate(pairs_indexes):
        
        rank_index = compute_pair_gt_tree_distance(hierarchical_labels[idx1], hierarchical_labels[idx2], number_of_ranks )
        rank_map[rank_index, pair_index] = 1
        pair_rank[pair_index] = rank_index
    return rank_map, pair_rank


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



def compute_rank_based_loss(batch_size, minibatch_embedding_coordinates, minibatch_labels, minibatch_rank_map, number_of_ranks, distance):
    if torch.cuda.is_available():
        device = 'cuda'

    pairs_indexes = torch.tensor(list(itertools.combinations(range(batch_size),2))).to(device)
    number_pairs = pairs_indexes.shape[0]
    
    #1 Pairwise embedding distances:
    if distance = 'euclidean':
        embedding_pairwise_distances = compute_pairwise_euclidean_distances(torch.transpose(minibatch_embedding_coordinates,0,1).to(device), minibatch_embedding_coordinates.shape[1], batch_size )
    elif distance = 'cosine':
        embedding_pairwise_distances = compute_pairwise_cosine_distances(minibatch_embedding_coordinates.to(device))

    count_samples_in_each_rank = torch.sum(minibatch_rank_map, 1).reshape(number_of_ranks+1,1)

    #2 Sort pairs accordingly to embedding distance
    sorted_embedding_distances_indexes = torch.argsort(embedding_pairwise_distances)
    sorted_pairs_indexes =torch.argsort(torch.argsort(embedding_pairwise_distances)).to(device)
    
    #3 Compute target distances for each rank:
    target_distances_per_rank = torch.ones(number_of_ranks).to(device)*-1
    target_distances_per_rank = embedding_pairwise_distances[sorted_embedding_distances_indexes]
    target_distances_per_rank = target_distances_per_rank[0:number_of_ranks+1]     

    #4 check if pairs are in the correct rank -> compute Icorrect vector
    min_position_sorted_distances_array = torch.zeros((number_of_ranks+1, 1)).to(device)
    max_position_sorted_distances_array = torch.cumsum(count_samples_in_each_rank, 0).to(device)
    min_position_sorted_distances_array[1:] = max_position_sorted_distances_array[0:-1]

    
    rank_map_perpair = torch.where(torch.transpose(minibatch_rank_map, 0,1) == 1)[1]  
    target_distance_per_pair = target_distances_per_rank[rank_map_perpair]
    target_distance_per_pair = target_distance_per_pair.to(device)
    # print(target_distance_per_pair)

    min_position_sorted_distances_array_pair = min_position_sorted_distances_array[rank_map_perpair].to(device)
    max_position_sorted_distances_array_pair = max_position_sorted_distances_array[rank_map_perpair].to(device)-1

    I_correct_position_pairsA = sorted_pairs_indexes.reshape(number_pairs,1) <= max_position_sorted_distances_array_pair
    I_correct_position_pairsB = (sorted_pairs_indexes.reshape(number_pairs,1) >= min_position_sorted_distances_array_pair)
    I_correct_position_pairs = I_correct_position_pairsA == I_correct_position_pairsB
    I_correct_position_pairs = I_correct_position_pairs.long().to(device)
   
    #5 compute loss per pair:
    Loss_per_pair = (1 - torch.transpose(I_correct_position_pairs,0,1))*((embedding_pairwise_distances - target_distance_per_pair)**2)
    return torch.mean(Loss_per_pair), target_distance_per_pair, embedding_pairwise_distances, I_correct_position_pairs




def train_RbL(model, training_generator, validation_generator, output_folder, early_stopping_patience, save_embeddings_to_plot, n_epochs, configs, distance='euclidean', number_of_ranks=6): 
    len_train = len(training_generator.dataset)
    len_val = len(validation_generator.dataset)
    
    train_losses = []
    valid_losses = []

    best_avg_sils = -1 * np.inf
    best_avg_sils_epoch =0
    temporary_best_checkpoint = ''
    plot_frame = 0
    
    for epoch in range(n_epochs):
        model.train()
        embeddings_plot_array = np.empty((configs['BATCH_SIZE'], configs['output_EMBEDDINGS_SIZE']))
        labels=np.asarray([])
        embeddings_every_epoch = np.empty((configs['BATCH_SIZE'], configs['output_EMBEDDINGS_SIZE']))
        labels_every_epoch=np.asarray([])
            
        train_losses = []
        valid_losses = []
        losses_to_plot = []
        
        process = tqdm.tqdm(training_generator, dynamic_ncols=True)
        for X_batch, y_batch in process:   
            
            model.zero_grad()
            embeddings = model(X_batch)

            minibatch_rank_map, _ = pre_compute_rank_map(number_of_ranks, hierarchical_labels=y_batch )
            
            if distance == 'cosine':
                loss, target_distance_per_pair, embedding_pairwise_distances, I_correct_position_pairs =  compute_rank_based_loss(embeddings.shape[0], embeddings, y_batch, minibatch_rank_map, number_of_ranks, distance)
            elif distance == 'euclidean':
                loss, target_distance_per_pair, embedding_pairwise_distances, I_correct_position_pairs =  compute_rank_based_loss(embeddings.shape[0], embeddings, y_batch, minibatch_rank_map, number_of_ranks, distance) #log_probs, y_batch)
                     
            if save_embeddings_to_plot and epoch%100 == 0 :
                losses_to_plot.append(loss)
                embeddings_plot_array = np.concatenate((embeddings_plot_array,  embeddings.detach().numpy()), axis = 0)
                labels = np.concatenate((labels, np.asarray(y_batch)))

            embeddings_every_epoch = np.concatenate((embeddings_every_epoch,  embeddings.detach().numpy()), axis = 0) # same as embeddings_plot_array but for every epoch, while the other is only every 100 epochs, Eventually save only this and sample from here to produce the embeddings to plot array!
            labels_every_epoch = np.concatenate((labels_every_epoch, np.asarray(y_batch)))

            loss.backward()
            model.optimizer.step()
            train_losses.append(loss.item())
            
        if save_embeddings_to_plot and epoch%100 == 0 :
            embeddings_plot_array = np.delete(embeddings_plot_array, range(configs['BATCH_SIZE']), 0)  # ?? remove first entrance because empty initialization from np
            with open(os.path.join(output_folder, "Embeddings_plot", "batch_embeddings_at_"+str(plot_frame)+".csv"), 'wb') as f:
                np.save(f, embeddings_plot_array)
            with open(os.path.join(output_folder, "Embeddings_plot", "LABELS_embeddings_to_plot.csv"), 'wb') as fl:
                np.save(fl, labels)
            plot_frame = plot_frame + 1      
        
        embeddings_every_epoch =  np.delete(embeddings_every_epoch, range(configs['BATCH_SIZE']), 0)

        #validation loop
        with torch.no_grad():
            val_embeddings_every_epoch = np.empty((configs['BATCH_SIZE'], configs['output_EMBEDDINGS_SIZE']))
            val_labels_every_epoch = np.asarray([])
             
            model.eval()
            for x_val, y_val in validation_generator:
                       
                embeddings_val = model(x_val)      

                val_embeddings_every_epoch = np.concatenate((val_embeddings_every_epoch,  embeddings_val.detach().numpy()), axis = 0) # same as embeddings_plot_array but for every epoch, while the other is only every 100 epochs, Eventually save only this and sample from here to produce the embeddings to plot array!
                val_labels_every_epoch = np.concatenate((val_labels_every_epoch, np.asarray(y_val)))
                       
                minibatch_rank_map_val, _ =pre_compute_rank_map(number_of_ranks, hierarchical_labels=y_val )
                val_loss, _, _, _ = compute_rank_based_loss(embeddings_val.shape[0], embeddings_val, y_val, minibatch_rank_map_val, number_of_ranks) #log_probs, y_batch)
                valid_losses.append(val_loss.item())

            val_embeddings_every_epoch =  np.delete(val_embeddings_every_epoch, range(configs['BATCH_SIZE']), 0)
                  
        train_clusters_score_IDlabels = ev.evaluate_cluster_quality_based_gt_annotations(embeddings_every_epoch, labels_every_epoch)
        labels_species_every_epoch = [l.split('_')[1] for l in labels_every_epoch]
        train_clusters_score_SPECIESlabels = ev.evaluate_cluster_quality_based_gt_annotations(embeddings_every_epoch, labels_species_every_epoch)

        val_clusters_score_IDlabels = ev.evaluate_cluster_quality_based_gt_annotations(val_embeddings_every_epoch, val_labels_every_epoch)
        val_labels_species_every_epoch = [l.split('_')[1] for l in val_labels_every_epoch]
        val_clusters_score_SPECIESlabels = ev.evaluate_cluster_quality_based_gt_annotations(val_embeddings_every_epoch, val_labels_species_every_epoch)

        avg_sils = (val_clusters_score_IDlabels + val_clusters_score_SPECIESlabels)/2
        train_loss = np.average(train_losses)
        val_loss = np.average(valid_losses)
        
        wandb.log({"training_loss": train_loss, "val_loss": val_loss, "Train_silhouette_score_IDlabels": train_clusters_score_IDlabels, "Train_silhouette_score_SPECIESlabels": train_clusters_score_SPECIESlabels, "VAL_silhouette_score_IDlabels": val_clusters_score_IDlabels, "VAL_silhouette_score_SPECIESlabels": val_clusters_score_SPECIESlabels, "VAL_averaged_silhouette_scores": avg_sils})         
        print("Epoch", epoch, ", Loss:",round(train_loss,3), "Val_loss:", round(val_loss,3), "silhouete_score_ID_labels:", train_clusters_score_IDlabels, "silhouette_score_species_labels", train_clusters_score_SPECIESlabels, "VAL_silhouette_score_IDlabels", val_clusters_score_IDlabels, "VAL_silhouette_score_SPECIESlabels", val_clusters_score_SPECIESlabels, "VAL_averaged_silhouette_scores:",  avg_sils)
    
        if avg_sils > best_avg_sils:
            checkpoint = {
                'net_dict':model.state_dict(),
                'val_loss':val_loss,
                'epoch':epoch,
            }
     
            best_avg_sils = avg_sils
            best_avg_sils_epoch = epoch

            # remove last temporary checkpoint file
            if os.path.isfile(temporary_best_checkpoint): 
                os.remove(temporary_best_checkpoint)

            temporary_best_checkpoint = os.path.join(output_folder, "Checkpoint" + '_epoch' + str(epoch) + '_valloss' + str(round(val_loss,2)) + '_val_avg_sils' + str(round(avg_sils,2)) +'.pt')
            print("Saving parameters to", temporary_best_checkpoint)
            torch.save(checkpoint, temporary_best_checkpoint)
            torch.save(model.state_dict(), os.path.join(wandb.run.dir, 'model.pt'))
        
        # EARLY STOPPING :
        if epoch > best_avg_sils_epoch + early_stopping_patience :
            return temporary_best_checkpoint
           
    return temporary_best_checkpoint


def predict(model, test_generator, configs, output_folder):
    len_test=len(test_generator.dataset)

    with torch.no_grad():
        embeddings_test = np.ones((1, configs["output_EMBEDDINGS_SIZE"]))* -1
        labels_test = np.asarray([])
       
        for x, y in test_generator:
            embeddings = model(x)      

            embeddings_test = np.concatenate((embeddings_test,  embeddings.detach().numpy()), axis = 0)
            labels_test = np.concatenate((labels_test, np.asarray(y)))

        embeddings_test =  np.delete(embeddings_test, range(1), 0)
    
    clusters_score_IDlabels = ev.evaluate_cluster_quality_based_gt_annotations(embeddings_test, labels_test)
    labels_test_species= [l.split('_')[1] for l in labels_test]
    clusters_score_SPECIESlabels = ev.evaluate_cluster_quality_based_gt_annotations(embeddings_test, labels_test_species)
    avg_sils = (clusters_score_IDlabels + clusters_score_SPECIESlabels)/2

    return clusters_score_IDlabels, clusters_score_SPECIESlabels