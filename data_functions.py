
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class RankBasedLossHierarchicalLabelsEmbeddings(Dataset):
    def __init__(self, partition_dataframe, features_folder, target_labels='hierarchical_labels'):
        self.dataset = partition_dataframe 
        self.wavs_list = list(self.dataset.wavfilename)
        self.features_folder = features_folder
        self.target_labels = target_labels
        self.labels = list(self.dataset[target_labels])
      
    def __len__(self):
        return len(self.wavs_list)

    def __getitem__(self,idx):
        ID = self.wavs_list[idx][0:-4]+'.npy'
        X = np.load(os.path.join(self.features_folder, ID))
        y = self.labels[idx]
        return X, y


class QuadrupletsHierarchicalLabelsEmbeddings(Dataset):
    def __init__(self, partition_dataframe, features_folder): 
        self.data = partition_dataframe 
        self.features_folder = features_folder
        self.anchors = list(self.data.A_wavefilename)
        self.anchors_labels = list(self.data.A_labels )
        self.positive_plus = list(self.data.PP_wavefilename)
        self.positive_plus_labels = list(self.data.PP_labels)
        self.positive_neg = list(self.data.PP_wavefilename)
        self.positive_neg_labels = list(self.data.PN_labels)
        self.negatives = list(self.data.N_wavefilename)
        self.negative_labels = list(self.data.N_labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        
        anchor_filename = self.anchors[idx][0:-4]+'.npy'
        Xa = np.load(os.path.join(self.features_folder, anchor_filename))
        Ya = self.anchors_labels[idx]

        pp_filename = self.positive_plus[idx][0:-4]+'.npy'
        Xpp = np.load(os.path.join(self.features_folder, pp_filename))
        Ypp = self.positive_plus_labels[idx]

        pn_filename = self.positive_neg[idx][0:-4]+'.npy'
        Xpn = np.load(os.path.join(self.features_folder, pn_filename))
        Ypn = self.positive_neg_labels[idx]

        neg_filename = self.negatives[idx][0:-4]+'.npy'
        Xn = np.load(os.path.join(self.features_folder, neg_filename))
        Yn = self.negative_labels[idx]

        return Xa, Xpp, Xpn, Xn, Ya, Ypp, Ypn, Yn