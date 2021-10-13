import SingleLayer_net as single_layer
import rank_based_loss as rbl
# import wandb
import torch
import data_functions as df
import os
import json
import pandas as pd
import csv


os.environ['CUDA_VISIBLE_DEVICES'] = '3'
# wandb.init(project='example')

exp_name = 'example'
# wandb.run.name = exp_name
standardized_data = True
save_training_embeddings_to_plot = True
shuffle = False  
drop_last = False 

experiments_folder ="./example_data"

initial_embeddings_path = os.path.join(experiments_folder, 'Normalized_VGGish_embeddings_based_on_Training_Set')
train_initial_embeddings_path = os.path.join(initial_embeddings_path, 'train')
val_initial_embeddings_path = os.path.join(initial_embeddings_path, 'val')
test_initial_embeddings_path = os.path.join(initial_embeddings_path, 'test')

results_folder = os.path.join(experiments_folder, "results_"+exp_name)
checkpoints_folder = os.path.join(results_folder, "checkpoints")
if not os.path.exists(checkpoints_folder):
        os.makedirs(checkpoints_folder)

if save_training_embeddings_to_plot:
        if not os.path.exists(os.path.join(checkpoints_folder, "Embeddings_plot")):
                os.mkdir(os.path.join(checkpoints_folder, "Embeddings_plot"))
 
train_df = pd.read_csv(os.path.join(experiments_folder, 'train.csv'), dtype = str)
val_df = pd.read_csv(os.path.join(experiments_folder, 'val.csv'), dtype = str)
test_df = pd.read_csv(os.path.join(experiments_folder,  'test.csv'), dtype = str)

configs = {"EMBEDDINGS_SIZE" : 128,
"output_EMBEDDINGS_SIZE" :3, 
"EARLY_STOPPING_PTC" : 20,
"LR" : 1e-5,
"BATCH_SIZE" : 12,
"n_epochs" : 100, 
}
params = {'batch_size': configs["BATCH_SIZE"],'shuffle': shuffle, 'drop_last': drop_last}

training_set = df.RankBasedLossHierarchicalLabelsEmbeddings(train_df, train_initial_embeddings_path, target_labels='hierarchical_labels')#,'species','taxon'])
training_generator = torch.utils.data.DataLoader(training_set, **params)
len_train = len(training_set)


validation_set = df.RankBasedLossHierarchicalLabelsEmbeddings(val_df , val_initial_embeddings_path, target_labels='hierarchical_labels')#,'species','taxon'])
params_val = {'batch_size': configs["BATCH_SIZE"],'shuffle': False, 'drop_last': False}
validation_generator = torch.utils.data.DataLoader(validation_set, **params_val)
len_val = len(validation_set)

model =single_layer.SingleLayerHypersphereConstraint(configs)

# wandb.watch(model)
# wandb.config = configs
# wandb.config["architecture"] = "LinLayer_cosinedist"
# wandb.config["dataset"] = "TuT"
with open(os.path.join(results_folder, 'configs_dict'), "w") as c:
        json.dump(configs, c)

checkpoint_name = rbl.train_RbL(model, training_generator, validation_generator,
                                        checkpoints_folder, configs['EARLY_STOPPING_PTC'], save_training_embeddings_to_plot, 
                                        configs['n_epochs'], configs, distance='cosine',
                                        number_of_ranks = 4)



print( "\nFinished training, will now use the checkpoint to generate embeddings for the test set:")
# Predict with checkpoint:

# if save_embeddings_to_plot:
if not os.path.exists(os.path.join(results_folder, "test_Embeddings_plot")):
    os.mkdir(os.path.join(results_folder, "test_Embeddings_plot"))

test_set = df.RankBasedLossHierarchicalLabelsEmbeddings(test_df, test_initial_embeddings_path, target_labels = 'hierarchical_labels')
test_generator = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)
len_test = len(test_set)

# load the checkpoint, configs and model
with open(os.path.join(results_folder, "configs_dict") )as c:
    configs = json.load(c)

model=single_layer.SingleLayerHypersphereConstraint(configs)
model.load_state_dict(torch.load(checkpoint_name)["net_dict"])

sil_id, sil_species =rbl.predict(model, test_generator, configs, results_folder)
print("sil_fine level", sil_id)
print('sil_coarse level', sil_species)
with open(os.path.join(results_folder, 'silhouettes_on_test_set.csv'), 'w') as fout:
    writer = csv.writer(fout)
    writer.writerow(['sil_fine_level', str(sil_id)])
    writer.writerow(['sil_coarse_level', str(sil_species)])