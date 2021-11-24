import torch
import torch.nn as nn
# import wandb
import evaluation as ev
import itertools
import tqdm

def calc_euclidean(x1, x2):
    return (x1 - x2).pow(2).sum(1)

def compute_cosine_distances(a, b):
    D = 1 - torch.mm(a, torch.transpose(b, 0, 1))
    tri_idx = torch.triu_indices(a.shape[0],a.shape[0],1)
    pairwise_dist_vector = D[tri_idx[0],tri_idx[1]]
    return pairwise_dist_vector
    

def compute_triplet_loss(margin, anchor, pos, neg, distance):

    if distance == 'euclidean':
        distance_positive = calc_euclidean(anchor, pos)
        distance_negative = calc_euclidean(anchor, neg)
    elif distance == 'cosine':
        distance_positive = compute_cosine_distances(anchor, pos)
        distance_negative = compute_cosine_distances(anchor, neg)
    
    losses = torch.relu(distance_positive - distance_negative + margin) # torch.max(distance_positive - distance_negative + margin, 0)

    return losses

def compute_quadruplet_loss(batch_embeddings_anchor, batch_embeddings_pp, batch_embeddings_pn, batch_embeddings_n, distance, margin_alpha, margin_beta):

    losses_triplet_1st_lvl = compute_triplet_loss(margin_alpha - margin_beta, batch_embeddings_anchor, batch_embeddings_pp, batch_embeddings_pn, distance)
    losses_triplet_2nd_lvl = compute_triplet_loss(margin_beta, batch_embeddings_anchor, batch_embeddings_pn, batch_embeddings_n, distance )

    losses = losses_triplet_1st_lvl + losses_triplet_2nd_lvl
    return losses.mean()


def train_quadL(model, margin_alpha, margin_beta, training_generator, validation_generator, output_folder, early_stopping_patience, save_embeddings_to_plot, n_epochs, configs, distance='euclidean'): 
    len_train = len(training_generator.dataset)
    len_val = len(validation_generator.dataset)
    
    train_losses = []
    valid_losses = []

    best_avg_sils = -999
    best_avg_sils_epoch = 0
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
        for Xa, Xpp, Xpn, Xn, Ya, Ypp, Ypn, Yn in process:
         
            model.zero_grad()
      
            embedding_anchor = model(Xa)
            embedding_pp = model(Xpp)
            embedding_pn = model(Xpn)
            embedding_n = model(Xn)
            
            ## COMPUTE LOSS         
            loss = compute_quadruplet_loss(embedding_anchor, embedding_pp, embedding_pn, embedding_n, distance, margin_alpha, margin_beta)
                        
            
            if save_embeddings_to_plot and epoch%100 == 0 :
                losses_to_plot.append(loss)
                embeddings_plot_array = np.concatenate((embeddings_plot_array,  embedding_anchor.detach().numpy()), axis = 0)
                embeddings_plot_array = np.concatenate((embeddings_plot_array,  embedding_pp.detach().numpy()), axis = 0)
                embeddings_plot_array = np.concatenate((embeddings_plot_array,  embedding_pn.detach().numpy()), axis = 0)
                embeddings_plot_array = np.concatenate((embeddings_plot_array,  embedding_n.detach().numpy()), axis = 0)
                labels = np.concatenate((labels, np.asarray(Ya)))
                labels = np.concatenate((labels, np.asarray(Ypp)))
                labels = np.concatenate((labels, np.asarray(Ypn)))
                labels = np.concatenate((labels, np.asarray(Yn)))
               

               
            embeddings_every_epoch = np.concatenate((embeddings_every_epoch,  embedding_anchor.detach().numpy()), axis = 0)
            labels_every_epoch = np.concatenate((labels_every_epoch, np.asarray(Ya)))
            embeddings_every_epoch = np.concatenate((embeddings_every_epoch,  embedding_pp.detach().numpy()), axis = 0)
            labels_every_epoch = np.concatenate((labels_every_epoch, np.asarray(Ypp)))
            embeddings_every_epoch = np.concatenate((embeddings_every_epoch,  embedding_pn.detach().numpy()), axis = 0) 
            labels_every_epoch = np.concatenate((labels_every_epoch, np.asarray(Ypn)))
            embeddings_every_epoch = np.concatenate((embeddings_every_epoch,  embedding_n.detach().numpy()), axis = 0) 
            labels_every_epoch = np.concatenate((labels_every_epoch, np.asarray(Yn)))

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
            val_labels_every_epoch=np.asarray([])
            model.eval()
            for Xa_val, Xpp_val, Xpn_val, Xn_val, Ya_val, Ypp_val, Ypn_val, Yn_val in validation_generator:
                
                embedding_anchor_val = model(Xa_val)
                embedding_pp_val = model(Xpp_val)
                embedding_pn_val = model(Xpn_val)
                embedding_n_val = model(Xn_val)             
                
                val_embeddings_every_epoch = np.concatenate((val_embeddings_every_epoch,  embedding_anchor_val.detach().numpy()), axis = 0) 
                val_labels_every_epoch = np.concatenate((val_labels_every_epoch, np.asarray(Ya_val)))
                val_embeddings_every_epoch = np.concatenate((val_embeddings_every_epoch,  embedding_pp_val.detach().numpy()), axis = 0) 
                val_labels_every_epoch = np.concatenate((val_labels_every_epoch, np.asarray(Ypp_val)))
                val_embeddings_every_epoch = np.concatenate((val_embeddings_every_epoch,  embedding_pn_val.detach().numpy()), axis = 0)
                val_labels_every_epoch = np.concatenate((val_labels_every_epoch, np.asarray(Ypn_val)))
                val_embeddings_every_epoch = np.concatenate((val_embeddings_every_epoch,  embedding_n_val.detach().numpy()), axis = 0) 
                val_labels_every_epoch = np.concatenate((val_labels_every_epoch, np.asarray(Yn_val)))

                val_loss = compute_quadruplet_loss(embedding_anchor_val, embedding_pp_val, embedding_pn_val, embedding_n_val, distance, margin_alpha, margin_beta)
                valid_losses.append(val_loss.item())

            val_embeddings_every_epoch =  np.delete(val_embeddings_every_epoch, range(configs['BATCH_SIZE']), 0)

   

        #compute pairwise distances
        train_clusters_score_IDlabels = ev.evaluate_cluster_quality_based_gt_annotations(embeddings_every_epoch, labels_every_epoch)
        # print(train_clusters_score_IDlabels)
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
        
        # EARLY STOPPING HERE:
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
            embeddings_test = np.concatenate((embeddings_test,  embeddings.detach().numpy()), axis = 0) # same as embeddings_plot_array but for every epoch, while the other is only every 100 epochs, Eventually save only this and sample from here to produce the embeddings to plot array!
            labels_test = np.concatenate((labels_test, np.asarray(y)))
  
        embeddings_test =  np.delete(embeddings_test, range(1), 0)
    
    clusters_score_IDlabels = ev.evaluate_cluster_quality_based_gt_annotations(embeddings_test, labels_test)
    labels_test_species= [l.split('_')[1] for l in labels_test]
    clusters_score_SPECIESlabels = ev.evaluate_cluster_quality_based_gt_annotations(embeddings_test, labels_test_species)
    avg_sils = (clusters_score_IDlabels + clusters_score_SPECIESlabels)/2
            
    return clusters_score_IDlabels, clusters_score_SPECIESlabels