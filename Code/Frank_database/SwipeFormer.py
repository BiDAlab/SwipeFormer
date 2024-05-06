# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 08:27:48 2023

@author: paula
"""

import torch
import torch.nn as nn
# import torch.tensor
import numpy as np
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from SwipeDatasetTriplet import *
from sklearn.metrics import roc_auc_score, roc_curve, det_curve
# device
from sklearn.preprocessing import label_binarize
import time
from transformerLayers import  Gaussian_Position, SwipeTransformer_arch
from tqdm import tqdm
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.covariance import ShrunkCovariance
from sklearn.neighbors import KernelDensity
from sklearn import svm
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import seaborn as sns
from sklearn.mixture import GaussianMixture

import scipy as sp
import matplotlib.ticker as mtick


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

############  TRAINING SETTINGS  #################
import argparse


def eer_compute(scores_g, scores_i):
    far = []
    frr = []
    ini = min(np.concatenate((scores_g, scores_i)))
    fin = max(np.concatenate((scores_g, scores_i)))
    if ini == fin:
        return 50.0, [], []
    paso = (fin - ini) / 10000
    threshold = ini - paso
    while threshold < fin + paso:
        far.append(len(np.where(scores_i <= threshold)[0]) / len(scores_i))
        frr.append(len(np.where(scores_g > threshold)[0]) / len(scores_g))
        threshold = threshold + paso
    diferencia = abs(np.asarray(far) - np.asarray(frr))
    j = np.where(diferencia == min(diferencia))[0]
    #    j = np.where(far==min(far-0.01))[0]
    index = j[0]
    return ((far[index] + frr[index]) / 2) * 100, frr, far


class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def calc_euclidean(self, x1, x2):
        return (x1 - x2).pow(2).sum(1)

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        distance_positive = self.calc_euclidean(anchor, positive)
        distance_negative = self.calc_euclidean(anchor, negative)
        losses = torch.relu(distance_positive - distance_negative + self.margin)

        return losses.mean()

def calc_euclidean(x1, x2):
    return (x1 - x2).pow(2).sum(1)

def gen_conf_matrix(pred, truth, conf_matrix):
    p = pred.cpu().tolist()
    l = truth.cpu().tolist()
    for i in range(len(p)):
        conf_matrix[l[i]][p[i]] += 1
    return conf_matrix

def write_to_file(conf_matrix):
    f = open("conf_matrix.txt", mode='w+', encoding='utf-8')
    for x in range(len(conf_matrix)):
        base = sum(conf_matrix[x])
        for y in range(len(conf_matrix[0])):
            value = str(format(conf_matrix[x][y]/base, '.2f'))
            f.write(value+'&')
        f.write('\n')
    f.flush()
    f.close()

  

result = np.array([])
experiment_times = 1        # number of experiments
validation_threshold_Transformer = 3  # after #validation_threshold times that the validation loss does not decrease, the training process stops
learning_rate_Transformer = 0.15                                                                              #^#

###########  NETWORK PARAMETERS  #################
data_length = 50       # number of signals per each gait cycle
channels = 11           # number of channnels
hlayers = 10
hheads = 50
K = 20
output_dim = 64

##################  DATASET  ######################
testing_dataset = torch.load('testing_data_frank.pt')
testing_dataloader = DataLoader(testing_dataset, batch_size=64, shuffle=False)

validation_dataset = torch.load('validation_data_frank.pt')
validation_dataloader = DataLoader(validation_dataset, batch_size=64, shuffle=False)


class SwipeFormer(torch.nn.Module):
    def __init__(self):
        super(SwipeFormer, self).__init__()
        self.sigmoid = torch.nn.LogSigmoid()
        self.transformer_h = SwipeTransformer_arch(data_length, hlayers, hheads)
        self.transformer_v = SwipeTransformer_arch(data_length, hlayers, hheads)
        self.kernel_num = 128
        self.kernel_num_v = 128
        self.filter_sizes = [channels, channels] #Reduced because of the input (6,80)
        self.filter_sizes_v = [channels, channels] #[2,2]
        self.pos_encoding_h = Gaussian_Position(data_length, channels, K)
        self.pos_encoding_v = Gaussian_Position(data_length, channels, K)
        self.dense = torch.nn.Linear(self.kernel_num * len(self.filter_sizes) + self.kernel_num_v * len(self.filter_sizes_v), output_dim)
        self.dense2 = torch.nn.Linear(self.kernel_num * len(self.filter_sizes), output_dim)
        self.dropout_rate = 0.5
        self.dropout = torch.nn.Dropout(self.dropout_rate)
        self.encoders = []
        self.encoder_v = []
        for i, filter_size in enumerate(self.filter_sizes):
            enc_attr_name = "encoder_%d" % i
            self.__setattr__(enc_attr_name,
                             torch.nn.Conv1d(in_channels=data_length,
                                       out_channels=self.kernel_num,
                                       kernel_size=filter_size).to('cuda')
                             )
            self.encoders.append(self.__getattr__(enc_attr_name))
        for i, filter_size in enumerate(self.filter_sizes_v):
            enc_attr_name_v = "encoder_v_%d" % i
            self.__setattr__(enc_attr_name_v,
                             torch.nn.Conv1d(in_channels=data_length,
                                       out_channels=self.kernel_num_v,
                                       kernel_size=filter_size).to('cuda')
                             )
            self.encoder_v.append(self.__getattr__(enc_attr_name_v))

    def _aggregate(self, o, v=None):
        enc_outs = []
        enc_outs_v = []
        for encoder in self.encoders:
            f_map = encoder(o.transpose(-1, -2))
            enc_ = F.relu(f_map)
            k_h = enc_.size()[-1]
            enc_ = F.max_pool1d(enc_, kernel_size=k_h)
            enc_ = enc_.squeeze(dim=-1)
            enc_outs.append(enc_)
        encoding = self.dropout(torch.cat(enc_outs, 1))
        q_re = F.relu(encoding)
        if self.transformer_v is not None:
            for encoder in self.encoder_v:
                f_map = encoder(v.transpose(-1, -2))
                enc_ = F.relu(f_map)
                k_h = enc_.size()[-1]
                enc_ = F.max_pool1d(enc_, kernel_size=k_h)
                enc_ = enc_.squeeze(dim=-1)
                enc_outs_v.append(enc_)
            encoding_v = self.dropout(torch.cat(enc_outs_v, 1))
            v_re = F.relu(encoding_v)
            q_re = torch.cat((q_re, v_re), dim=1)
        return q_re

    def forward(self, data, channels):
        x = data.view(data.shape[0], channels, -1)
        x = torch.fft.rfftn(x, s=data_length-1, norm="forward")
        x = torch.view_as_real(x)
        x = x.view(data.shape[0], channels, -1)
        x = self.pos_encoding_h(x)
        x = self.transformer_h(x)

        y = data.view(data.shape[0], channels, -1)
        y = self.pos_encoding_v(y)
        y = self.transformer_v(y)
        re = self._aggregate(x, y)
        predict = self.sigmoid(self.dense(re))

        return predict     



batch_size=64
test_dataloader = testing_dataloader

TransformerModel = SwipeFormer().double()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(TransformerModel.parameters(), lr=0.001)
validation_threshold_Transformer = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
TransformerModel = TransformerModel.to(device)


TransformerModel.load_state_dict(torch.load('../Models/Exp_Frank_Swipeformer_2_294'))

TransformerModel.eval()

impost_test_samples = 10
enrol_samples = 5
test_samples = 10

# embedded templates and the corresponding labels
test_embeddings = []
embedding_template = torch.tensor([])  # gait features extracted from Transformer 
test_label = torch.LongTensor([])

for batch_idx, (anchor_sgm, anchor_label, session_label) in enumerate(testing_dataloader, 0):
    embedded_temp = torch.tensor([])
    anchor_sgm = Variable(anchor_sgm).double()
    anchor_label = Variable(anchor_label).long()
    # load test data
    anchor_sgm = anchor_sgm.to(device)
    if anchor_label.dim() != 0:
        with torch.no_grad():
            embedding = TransformerModel(anchor_sgm, channels)
            if embedding.dim() != 0:
                embedded_temp = embedding.cpu()
                embedding_template = torch.cat([embedding_template, embedded_temp], dim=0)


            test_label = torch.cat([test_label, anchor_label], dim=0)

embedding_template = embedding_template.data.numpy()
test_label = test_label.numpy()


# embedded templates and the corresponding labels
val_embeddings = []
embedding_template_val = torch.tensor([])  # gait features extracted from Transformer 
val_label = torch.LongTensor([])

for batch_idx, (anchor_sgm, positive, negative, anchor_label) in enumerate(validation_dataloader, 0):
    embedded_temp = torch.tensor([])
    anchor_sgm = Variable(anchor_sgm).double()
    anchor_label = Variable(anchor_label).long()
    # load val data
    anchor_sgm = anchor_sgm.to(device)
    if anchor_label.dim() != 0:
        with torch.no_grad():
            embedding = TransformerModel(anchor_sgm, channels)
            if embedding.dim() != 0:
                embedded_temp = embedding.cpu()
                embedding_template_val = torch.cat([embedding_template_val, embedded_temp], dim=0)


            val_label = torch.cat([val_label, anchor_label], dim=0)

embedding_template_val = embedding_template_val.data.numpy()
val_label = val_label.numpy()




genuine_distances  = []
impostor_distances = []
for i, profile in enumerate(np.unique(test_label)):  # train_label_ID:

    index = np.where(test_label == profile)[0]
    enrol_index = index[:enrol_samples]
    gen_index = index[-test_samples:]
    oth_index = np.where(test_label != profile)[0]
    # oth_index = np.random.suffle(oth_index)
    imp_index = oth_index[:impost_test_samples]
    enrol_emb = embedding_template[enrol_index]
    gen_emb = embedding_template[gen_index]
    imp_emb = embedding_template[imp_index]
    genuine_distances.append([np.mean(euclidean_distances(enrol_emb, gen_emb), axis = 0)])
    impostor_distances.append([np.mean(euclidean_distances(enrol_emb, imp_emb), axis = 0)])


genuine_distances = np.array(genuine_distances, dtype=float)
impostor_distances = np.array(impostor_distances, dtype=float)
eer, frr, far = eer_compute(np.ravel(genuine_distances), np.ravel(impostor_distances))
genuine_distances = genuine_distances.reshape(-1,)
impostor_distances = impostor_distances.reshape(-1,)
y_true_EuclDist = np.concatenate([np.ones((len(genuine_distances),)), np.zeros((len(impostor_distances),))], axis=0)
y_pred_EuclDist = np.concatenate([genuine_distances, impostor_distances])

# EER in Euclidean Distance
fpr_EuclDist, tpr_EuclDist, thresh_EuclDist = roc_curve(y_true_EuclDist, y_pred_EuclDist)
eer_score_EuclDist = brentq(lambda x : 1. - x - interp1d(fpr_EuclDist, tpr_EuclDist)(x), 0., 1.)
# DET curve
fpr_EuclDist_det, fnr_EuclDist_det, thresh_EuclDist = det_curve(y_true_EuclDist, y_pred_EuclDist)

#Shrunk Covariance
number_of_template = 15
kernel_r = 'rbf'
gamma_r = 1
nu_r = 0.001
total_imposter_trying = 0
total_genuine_trying = 0
y_true = []
y_predict = []

for i, profile in enumerate(np.unique(test_label)):
    
    # print(curr_user)

    # get data of current user only
    idx = np.where(test_label == profile)
    idx = idx[0]
    curr_user_data = embedding_template[idx][:]

    # divide to training and testing
    number_of_template = np.shape(idx)[0]
    train_number = int(number_of_template/2)

    curr_user_data_train = curr_user_data[0:train_number][:]
    curr_user_data_test = curr_user_data[train_number:][:]

    # other users data
    idx_other = np.where(test_label != profile)
    idx_other = idx_other[0]
    idx_other = idx_other[:len(idx)]
    other_users =  embedding_template[idx_other][:]
    other_users_train = other_users[:len(curr_user_data_train),:]
    other_users_test = other_users[len(curr_user_data_train):,:]


    clf_transf = ShrunkCovariance().fit(curr_user_data_train)
    
    score_curr_user = clf_transf.mahalanobis(curr_user_data_test)
    score_other_users = clf_transf.mahalanobis(other_users_test)
    
    total_genuine_trying += np.shape(curr_user_data_test)[0]
    total_imposter_trying += np.shape(other_users_test)[0]

    users_y_test_ShrunkCov = np.concatenate([np.ones((len(curr_user_data_test),)), np.zeros((len(other_users_test),))], axis = 0)
    users_y_predict_ShrunkCov = np.concatenate([score_curr_user, score_other_users], axis = 0)
    
    y_true.append(users_y_test_ShrunkCov)
    y_predict.append(users_y_predict_ShrunkCov)

y_true_ShrunkCov = np.hstack(y_true)
y_predict_ShrunkCov = np.hstack(y_predict)

# EER in Shrunk Covariance
fpr_ShrunkCov, tpr_ShrunkCov, thresh_ShrunkCov = roc_curve(y_true_ShrunkCov, y_predict_ShrunkCov)
eer_score_ShrunkCov = brentq(lambda x : 1. - x - interp1d(fpr_ShrunkCov, tpr_ShrunkCov)(x), 0., 1.)
# DET curve
fpr_ShrunkCov_det, fnr_ShrunkCov_det, thresh_ShrunkCov = det_curve(y_true_ShrunkCov, y_predict_ShrunkCov)


# Density Kernel Estimator 
total_imposter_trying = 0
total_genuine_trying = 0
y_true = []
y_predict = []

for i, profile in enumerate(np.unique(test_label)):
    
    # print(curr_user)

    # get data of current user only
    idx = np.where(test_label == profile)
    idx = idx[0]
    curr_user_data = embedding_template[idx][:]

    # divide to training and testing
    number_of_template = np.shape(idx)[0]
    train_number = int(number_of_template/2)

    curr_user_data_train = curr_user_data[0:train_number][:]
    curr_user_data_test = curr_user_data[train_number:][:]

    # other users data
    idx_other = np.where(test_label != profile)
    idx_other = idx_other[0]
    idx_other = idx_other[:len(idx)]
    other_users =  embedding_template[idx_other][:]
    other_users_train = other_users[:len(curr_user_data_train),:]
    other_users_test = other_users[len(curr_user_data_train):,:]


    clf_transf = KernelDensity(bandwidth=0.9, kernel='gaussian', metric='manhattan').fit(curr_user_data_train)
    
    score_curr_user = clf_transf.score_samples(curr_user_data_test)
    score_other_users = clf_transf.score_samples(other_users_test)
    
    total_genuine_trying += np.shape(curr_user_data_test)[0]
    total_imposter_trying += np.shape(other_users_test)[0]

    users_y_test_DKE = np.concatenate([np.ones((len(curr_user_data_test),)), np.zeros((len(other_users_test),))], axis = 0)
    users_y_predict_DKE = np.concatenate([score_curr_user, score_other_users], axis = 0)
    
    y_true.append(users_y_test_DKE)
    y_predict.append(users_y_predict_DKE)

y_true_DKE = np.hstack(y_true)
y_predict_DKE = np.hstack(y_predict)

# EER in DKE
fpr_DKE, tpr_DKE, thresh_DKE = roc_curve(y_true_DKE, y_predict_DKE)
eer_score_DKE = brentq(lambda x : 1. - x - interp1d(fpr_DKE, tpr_DKE)(x), 0., 1.)
# DET curve
fpr_DKE_det, fnr_DKE_det, thresh_DKE = det_curve(y_true_DKE, y_predict_DKE)



# Gaussiam Mixture Model 
total_imposter_trying = 0
total_genuine_trying = 0
y_true = []
y_predict = []

for i, profile in enumerate(np.unique(test_label)):
    
    # print(curr_user)

    # get data of current user only
    idx = np.where(test_label == profile)
    idx = idx[0]
    curr_user_data = embedding_template[idx][:]

    # divide to training and testing
    number_of_template = np.shape(idx)[0]
    train_number = int(number_of_template/2)

    curr_user_data_train = curr_user_data[0:train_number][:]
    curr_user_data_test = curr_user_data[train_number:][:]

    # other users data
    idx_other = np.where(test_label != profile)
    idx_other = idx_other[0]
    idx_other = idx_other[:len(idx)]
    other_users =  embedding_template[idx_other][:]
    other_users_train = other_users[:len(curr_user_data_train),:]
    other_users_test = other_users[len(curr_user_data_train):,:]


    clf_transf = GaussianMixture(n_components = 2, covariance_type = 'diag', random_state = 0)
    clf_transf.fit(curr_user_data_train)
    
    score_curr_user = clf_transf.score_samples(curr_user_data_test)
    score_other_users = clf_transf.score_samples(other_users_test)
    
    total_genuine_trying += np.shape(curr_user_data_test)[0]
    total_imposter_trying += np.shape(other_users_test)[0]

    users_y_test_GMM = np.concatenate([np.ones((len(curr_user_data_test),)), np.zeros((len(other_users_test),))], axis = 0)
    users_y_predict_GMM = np.concatenate([score_curr_user, score_other_users], axis = 0)
    
    y_true.append(users_y_test_GMM)
    y_predict.append(users_y_predict_GMM)

y_true_GMM = np.hstack(y_true)
y_predict_GMM = np.hstack(y_predict)

# EER in GMM
fpr_GMM, tpr_GMM, thresh_GMM = roc_curve(y_true_GMM, y_predict_GMM)
eer_score_GMM = brentq(lambda x : 1. - x - interp1d(fpr_GMM, tpr_GMM)(x), 0., 1.)
# DET curve
fpr_GMM_det, fnr_GMM_det, thresh_GMM = det_curve(y_true_GMM, y_predict_GMM)


# One Class SVM
total_imposter_trying = 0
total_genuine_trying = 0
y_true = []
y_predict = []

for i, profile in enumerate(np.unique(test_label)):
    
    # print(curr_user)

    # get data of current user only
    idx = np.where(test_label == profile)
    idx = idx[0]
    curr_user_data = embedding_template[idx][:]

    # divide to training and testing
    number_of_template = np.shape(idx)[0]
    train_number = int(number_of_template/2)

    curr_user_data_train = curr_user_data[0:train_number][:]
    curr_user_data_test = curr_user_data[train_number:][:]

    # other users data
    idx_other = np.where(test_label != profile)
    idx_other = idx_other[0]
    idx_other = idx_other[:len(idx)]
    other_users =  embedding_template[idx_other][:]
    other_users_train = other_users[:len(curr_user_data_train),:]
    other_users_test = other_users[len(curr_user_data_train):,:]


    clf_transf = svm.OneClassSVM(kernel=kernel_r, gamma = 0.5, coef0 = 0.8, tol = 0.00001, shrinking = False, cache_size = 1000).fit(curr_user_data_train)
    
    score_curr_user = clf_transf.score_samples(curr_user_data_test)
    score_other_users = clf_transf.score_samples(other_users_test)
    
    total_genuine_trying += np.shape(curr_user_data_test)[0]
    total_imposter_trying += np.shape(other_users_test)[0]

    users_y_test_OCSVM = np.concatenate([np.ones((len(curr_user_data_test),)), np.zeros((len(other_users_test),))], axis = 0)
    users_y_predict_OCSVM = np.concatenate([score_curr_user, score_other_users], axis = 0)
    
    y_true.append(users_y_test_OCSVM)
    y_predict.append(users_y_predict_OCSVM)

y_true_OCSVM = np.hstack(y_true)
y_predict_OCSVM = np.hstack(y_predict)

# EER in OCSVM
fpr_OCSVM, tpr_OCSVM, thresh_OCSVM = roc_curve(y_true_OCSVM, y_predict_OCSVM)
eer_score_OCSVM = brentq(lambda x : 1. - x - interp1d(fpr_OCSVM, tpr_OCSVM)(x), 0., 1.)
# DET curve
fpr_OCSVM_det, fnr_OCSVM_det, thresh_OCSVM = det_curve(y_true_OCSVM, y_predict_OCSVM)


# Binary SVM
svm_score_threshold_0 = np.arange(0, 0.00001, 0.000001)
svm_score_threshold_1 = np.arange(0.00002, 0.0002, 0.00001)
svm_score_threshold_2 = np.arange(0.0004, 0.06, 0.0002)
svm_score_threshold_3 = np.array(
[0.07, 0.08, 0.09, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.22, 0.24, 0.26, 0.28, 0.3, 0.32, 0.34, 0.36, 0.38,
 0.4, 0.45, 0.5, 0.55, 0.56, 0.57, 0.58, 0.6, 0.7, 0.8, 0.9])
svm_score_threshold_4 = np.arange(0.1, 0.4, 0.001)
svm_score_threshold_5 = np.arange(0.4, 0.9, 0.001)

# svm_score_threshold = np.concatenate(
# [svm_score_threshold_0, svm_score_threshold_1, svm_score_threshold_2, svm_score_threshold_3], 0)
svm_score_threshold = np.concatenate([svm_score_threshold_4, svm_score_threshold_5], 0)

total_imposter_trying = 0
total_genuine_trying = 0
y_true = []
y_predict = []

for i, profile in enumerate(np.unique(test_label)):
    
    # print(curr_user)

    # get data of current user only
    idx = np.where(test_label == profile)
    idx = idx[0]
    curr_user_data = embedding_template[idx][:]

    # divide to training and testing
    number_of_template = np.shape(idx)[0]
    train_number = int(number_of_template/2)

    curr_user_data_train = curr_user_data[0:train_number][:]
    curr_user_data_test = curr_user_data[-train_number:][:]

    # other users data
    idx_other = np.where(test_label != profile)
    idx_other = idx_other[0]
    idx_other = idx_other[:len(idx)]
    other_users =  embedding_template[idx_other][:]
    
    other_users_train_val = embedding_template_val[idx_other][:]
    # other_users_train = other_users[:len(curr_user_data_train),:]
    other_users_train = other_users_train_val[:len(curr_user_data_train),:]
    other_users_test = other_users[-len(curr_user_data_train):,:]


    user_genuine = np.ones((len(curr_user_data_train),))
    user_impostor = np.zeros((len(curr_user_data_train),))
    users_y_train_SVM = np.concatenate([user_genuine, user_impostor], axis=0)
    # clf_Transf = svm.SVC(kernel=kernel_r, gamma='scale')
    clf_Transf = svm.SVC(kernel=kernel_r, gamma = 0.001)
    # data_users_train_SVM = np.concatenate([curr_user_data_train, val_embedded_RNN[0 :  len(curr_user_data_train), :]], axis=0)
    data_users_train_SVM = np.concatenate([curr_user_data_train, other_users_train], axis=0)
    clf_Transf.fit(data_users_train_SVM, users_y_train_SVM) 

    
    score_curr_user = clf_Transf.decision_function(curr_user_data_test)
    score_other_users = clf_Transf.decision_function(other_users_test)
    
    total_genuine_trying += np.shape(curr_user_data_test)[0]
    total_imposter_trying += np.shape(other_users_test)[0]


    # NORMALIZE all scored to range 0-1
    score_min = min(score_curr_user.min(), score_other_users.min())
    score_max = max(score_curr_user.max(), score_other_users.max())
    score_curr_user = (score_curr_user - score_min) / (score_max - score_min)
    score_other_users = (score_other_users - score_min) / (score_max - score_min)

    users_y_test_BSVM = np.concatenate([np.ones((len(curr_user_data_test),)), np.zeros((len(other_users_test),))], axis = 0)
    users_y_predict_BSVM = np.concatenate([score_curr_user, score_other_users], axis = 0)
    
    y_true.append(users_y_test_BSVM)
    y_predict.append(users_y_predict_BSVM)

y_true_BSVM = np.hstack(y_true)
y_predict_BSVM = np.hstack(y_predict)

# EER in BSVM
fpr_BSVM, tpr_BSVM, thresh_BSVM = roc_curve(y_true_BSVM, y_predict_BSVM)
eer_score_BSVM = brentq(lambda x : 1. - x - interp1d(fpr_BSVM, tpr_BSVM)(x), 0., 1.)
# DET curve
fpr_BSVM_det, fnr_BSVM_det, thresh_BSVM = det_curve(y_true_BSVM, y_predict_BSVM)



# # Plot DET curves and EER
# plt.figure(figsize = (10,10))
# plt.plot(sp.stats.norm.ppf(1-fpr_EuclDist_det), sp.stats.norm.ppf(1-fnr_EuclDist_det), label=f"Euclidean Distance: EER = {1-eer_score_EuclDist:1%}")
# plt.plot(sp.stats.norm.ppf(1-fpr_ShrunkCov_det) , sp.stats.norm.ppf(1-fnr_ShrunkCov_det) , label=f"Shrunk Covariance: EER = {1-eer_score_ShrunkCov:1%}")
# plt.plot(sp.stats.norm.ppf(fpr_DKE_det) , sp.stats.norm.ppf(fnr_DKE_det) , label=f"DKE: EER = {eer_score_DKE:1%}")
# plt.plot(sp.stats.norm.ppf(fpr_GMM_det) , sp.stats.norm.ppf(fnr_GMM_det) , label=f"GMM: EER = {eer_score_GMM:1%}")
# plt.plot(sp.stats.norm.ppf(fpr_OCSVM_det) , sp.stats.norm.ppf(fnr_OCSVM_det) , label=f"OC-SVM: EER = {eer_score_OCSVM:1%}")
# plt.plot(sp.stats.norm.ppf(fpr_BSVM_det) , sp.stats.norm.ppf(fnr_BSVM_det) , label=f"B-SVM: EER = {eer_score_BSVM:1%}")

# plt.plot([0, 1], [1, 0], c=sns.color_palette()[-1], lw = 1, ls=':')
# plt.plot([0, 1], [0, 1], c=sns.color_palette()[-2], lw = 1)

# plt.title('DET Curves')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Negative Rate')
# plt.legend()
# plt.show()



fontsizebigtitle = 16
fontsizetitle = 14
fontsizelegend = 10
fontsizeaxis = 12
fontsizeticks = 9

ticks = [0.01, 0.02, 0.05, 0.10, 0.20, 0.40, 0.80, 0.95, 0.99]  # [0.001, 0.01, 0.05, 0.20, 0.5, 0.80, 0.95, 0.99, 0.999]
tick_locations = sp.stats.norm.ppf(ticks)
tick_labels = ['{:.0%}'.format(s) if (100 * s).is_integer() else '{:.1%}'.format(s) for s in ticks]
plt.xticks(tick_locations, labels=tick_labels, fontsize=fontsizeticks)
plt.xlim(-2.25, 0)
plt.yticks(tick_locations, labels=tick_labels, fontsize=fontsizeticks)
plt.ylim(-2.25, 0)

plt.plot(sp.stats.norm.ppf(1-fpr_EuclDist_det), sp.stats.norm.ppf(1-fnr_EuclDist_det), label=f"Euclidean Distance: EER = {1-eer_score_EuclDist:1%}")
plt.plot(sp.stats.norm.ppf(1-fpr_ShrunkCov_det) , sp.stats.norm.ppf(1-fnr_ShrunkCov_det) , label=f"Shrunk Covariance: EER = {1-eer_score_ShrunkCov:1%}")
plt.plot(sp.stats.norm.ppf(fpr_DKE_det) , sp.stats.norm.ppf(fnr_DKE_det) , label=f"DKE: EER = {eer_score_DKE:1%}")
plt.plot(sp.stats.norm.ppf(fpr_GMM_det) , sp.stats.norm.ppf(fnr_GMM_det) , label=f"GMM: EER = {eer_score_GMM:1%}")
plt.plot(sp.stats.norm.ppf(fpr_OCSVM_det) , sp.stats.norm.ppf(fnr_OCSVM_det) , label=f"OC-SVM: EER = {eer_score_OCSVM:1%}")
plt.plot(sp.stats.norm.ppf(fpr_BSVM_det) , sp.stats.norm.ppf(fnr_BSVM_det) , label=f"B-SVM: EER = {eer_score_BSVM:1%}")



plt.xlabel('False Acceptance Rate (%)', fontsize=fontsizeaxis)
plt.ylabel('False Rejection Rate (%)', fontsize=fontsizeaxis)
plt.legend(loc='upper right')
plt.grid()
plt.axis()
# plt.savefig('SwipeFormer_DET_HuMIdb.pdf', transparent=True, bbox_inches='tight', pad_inches=0) 
plt.show()


np.save('fpr_Frank_SwipeFormer.npy', fpr_BSVM_det)
np.save('fnr_Frank_SwipeFormer.npy', fnr_BSVM_det)
np.save('eer_Frank_SwipeFormer.npy', eer_score_BSVM)
