
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
# wandb.init(project='example')

exp_name = 'example'
# wandb.run.name = exp_name
standardized_data = True
save_training_embeddings_to_plot = True
shuffle = False  
drop_last = False 
Flag_compute_cluster_scores = True


experiments_folder ="/homes/in304/rank-based-embeddings/RankBasedLoss_for_DCASEasc2016_dataset/9scenes_3families"
# data_folder = "/import/c4dm-datasets/animal_identification/AAII_paper_augmented_dataset/"
# initial_embeddings_path = os.path.join(data_folder, "VGGish_embeddings_augmented_data")
# train_initial_embeddings_path = val_initial_embeddings_path = initial_embeddings_path


# master_csv = '' 


data_sets_csv_folder = experiments_folder

if standardized_data:
        initial_embeddings_path = os.path.join(experiments_folder, 'Normalized_VGGish_embeddings_based_on_Training_Set')
        train_initial_embeddings_path = os.path.join(initial_embeddings_path, 'train')
        val_initial_embeddings_path = os.path.join(initial_embeddings_path, 'val')


results_folder = os.path.join(experiments_folder, "results_"+exp_name)
checkpoints_folder = os.path.join(results_folder, "checkpoints")
if not os.path.exists(checkpoints_folder):
        os.makedirs(checkpoints_folder)

if save_training_embeddings_to_plot:
        if not os.path.exists(os.path.join(checkpoints_folder, "Embeddings_plot")):
                os.mkdir(os.path.join(checkpoints_folder, "Embeddings_plot"))
 
train_df = pd.read_csv(os.path.join(experiments_folder, 'shuf_train.csv'), dtype = str)
val_df = pd.read_csv(os.path.join(experiments_folder, 'shuf_val.csv'), dtype = str)
        # test_df = pd.read_csv(os.path.join(experiments_folder, 'test_dataset.csv'), dtype = str)


configs = {"EMBEDDINGS_SIZE" : 128,
"output_EMBEDDINGS_SIZE" :3, 
"EARLY_STOPPING_PTC" : 20,
"LR" : 1e-5,
"BATCH_SIZE" : 12,
"n_epochs" : 40000, 
}
params = {'batch_size': configs["BATCH_SIZE"],'shuffle': shuffle, 'drop_last': drop_last}





training_set = df.HierarchicalLabelsEmbeddings(train_df, train_initial_embeddings_path, target_labels='hierarchical_labels')#,'species','taxon'])
training_generator = torch.utils.data.DataLoader(training_set, **params)
len_train = len(training_set)


validation_set = df.HierarchicalLabelsEmbeddings(val_df , val_initial_embeddings_path, target_labels='hierarchical_labels')#,'species','taxon'])
params_val = {'batch_size': configs["BATCH_SIZE"],'shuffle': True, 'drop_last': False}
validation_generator = torch.utils.data.DataLoader(validation_set, **params_val)
len_val = len(validation_set)




model =single_layer.SingleLayerHypersphereConstraintRankBasedLossEmbedding(configs)

wandb.watch(model)
wandb.config = configs
wandb.config["architecture"] = "LinLayer_cosinedist_rankloss"
wandb.config["dataset"] = "NSYNTH"
with open(os.path.join(results_folder, 'configs_dict'), "w") as c:
        json.dump(configs, c)

checkpoint_name = single_layer.train_it(model, training_generator, validation_generator,
                                        checkpoints_folder, configs['EARLY_STOPPING_PTC'], save_training_embeddings_to_plot, 
                                        configs['n_epochs'], configs, distance='cosine',
                                        number_of_ranks = 4)

