import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sys
import time

from matplotlib import gridspec
from numba import njit, prange
from pyts.image import MarkovTransitionField
from pyts.preprocessing.discretizer import KBinsDiscretizer

import tsia.plot
import tsia.markov
import tsia.network_graph
from scipy.io import savemat
from scipy import io
from scipy.stats import entropy

import math
import matplotlib.patches as patches

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 28
fsize=32



def show_final_mtf(mtf):
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(1, 1, 1)
    _, mappable_image = tsia.plot.plot_markov_transition_field(mtf=mtf, ax=ax, colormap="GnBu", reversed_cmap=True)
    ax.set_title('Markov Transition Field')
    plt.colorbar(mappable_image)
    print(mappable_image)

    # adjust the position and size of the border to align with the image
    rect = patches.Rectangle((-0.5, -0.5), mtf.shape[1], mtf.shape[0], linewidth=1.5, edgecolor='black', facecolor='none')
    ax.add_patch(rect)
    
    plt.savefig("MTF.jpg")
    plt.show()


# get univariate state transition field matrix
def single_markov_transition_field(X_binned, X_mtm, n_timestamps):
    X_mtf = np.zeros((n_timestamps, 1))

    X_mtf[0, 0]=X_mtm[X_binned[0], X_binned[0]]
    # We loop through each timestamp twice to build a N x N matrix:
    for i in range(n_timestamps-1):
        X_mtf[i+1, 0] = X_mtm[X_binned[i], X_binned[i+1]]

    return X_mtf

# get multivariate state transition field matrix
def multi_markov_transition_field(X_binned, Y_binned, XY_mtm, n_timestamps):
    XY_mtf = np.zeros((n_timestamps, 1))

    XY_mtf[0, 0] = XY_mtm[X_binned[0], Y_binned[0]]
    # We loop through each timestamp twice to build a N x N matrix:
    for i in range(n_timestamps-1):
        XY_mtf[i+1, 0] = XY_mtm[X_binned[i], Y_binned[i+1]]

    return XY_mtf

#get single MTF matrix
def single_MTF(V_binned,n_timestamps,n_bins):
    # calculate state transition matrix
    V_mtm = tsia.markov.markov_transition_matrix(V_binned)
    V_mtm = tsia.markov.markov_transition_probabilities(V_mtm)

    # calculate state transition field matrix
    V_mtf=single_markov_transition_field(V_binned, V_mtm, n_timestamps)

    return V_mtf

# get multivariate MTF matrix
def multi_MTF(X_binned, Y_binned, n_timestamps, n_bins_X, n_bins_Y):
    # calculate state transition matrix
    XY_mtm = np.zeros((n_bins_X, n_bins_Y))
    # find the state transition frequency
    for t in prange(n_timestamps - 1):
        XY_mtm[X_binned[t], Y_binned[t + 1]] += 1
    # Normalize the state transition frequency and convert it into probability
    XY_mtm = tsia.markov.markov_transition_probabilities(XY_mtm)

    # calculate state transition field matrix
    XY_mtf = multi_markov_transition_field(X_binned, Y_binned, XY_mtm, n_timestamps)

    return XY_mtf


def entropy_optimal_bins(data, k_range=range(6, 12)):
    max_entropy = -1
    best_k = 3
    for k in k_range:
        discretizer = KBinsDiscretizer(n_bins=k, strategy='quantile')
        # discretizer = KBinsDiscretizer(n_bins=k, strategy='uniform')
        binned = discretizer.fit_transform(data)[0]
        p = np.bincount(binned.astype(int)) / len(data)
        current_entropy = entropy(p[p > 0])  # ignore empty bins
        if current_entropy > max_entropy:
            max_entropy = current_entropy
            best_k = k
    return best_k



def final_MTF(X,n_timestamps,n_bins=8): #X:(n_timestamps,38)
    Vn=X.shape[1]
    p=0.9 

    # bucket all data
    Binned=[]
    Bin_n=[]
    for i in range(Vn):
        V = X[:, i].reshape(1, -1)
        n_bins_V=entropy_optimal_bins(V)
        discretizer = KBinsDiscretizer(n_bins=n_bins_V, strategy='quantile')
        V_binned = discretizer.fit_transform(V)[0]
        Binned.append(V_binned)
        Bin_n.append(n_bins_V)

    # calculate MTF
    list_mtf=[]
    for i in range(Vn):
        self_MTF=single_MTF(Binned[i], n_timestamps, Bin_n[i])
        rest_MTF = np.zeros((n_timestamps, 1))
        for j in range(Vn):
            if j!=i:
                rest_MTF = rest_MTF + multi_MTF(Binned[i],Binned[j], n_timestamps, Bin_n[i], Bin_n[j])

        f_MTF=(self_MTF*p+rest_MTF*(1-p))/Vn
        list_mtf.append(f_MTF)

    F_MTF=np.concatenate(list_mtf,axis=1) 
    return F_MTF



# set file path
data_path='./ServerMachineDataset/'
save_path = './ServerMachineDataset/save/'
dataset_name_list = [
    # '1-1',
    # '1-2',
    # '1-3',
    '1-4',
    # '1-5',
    '1-6',
    # '1-7',
    # '1-8',
    # '2-1',
    # '2-2',
    # '2-3',
    # '2-4',
    # '2-5',
    # '2-6',
    # '2-7',
    # '2-8',
    # '2-9',
    # '3-1',
    # '3-2',
    # '3-3',
    # '3-4',
    # '3-5',
    # '3-6',
    # '3-7',
    # '3-8',
    # '3-9',
    # '3-10',
    # '3-11',
]


if __name__ == '__main__':

    N=10000
    n_timestamps =N


    start=0
    end=start+n_timestamps

    if not os.path.exists(save_path):
        os.makedirs(save_path)


    for dataset_name in dataset_name_list:
        X = np.loadtxt(data_path + 'test/machine-'+dataset_name+'.txt', dtype=np.float32, delimiter=',') 
        X1=X[start:end,:] 
        X2 = final_MTF(X1, n_timestamps) # convert the time series data matrix into an MTF matrix
        Y = np.loadtxt(data_path + 'test_label/machine-'+dataset_name+'.txt')
        Y=Y[start:end]
        savemat(save_path + 'machine-' + dataset_name + '.mat', {'A': X1,'B':X2,'C':Y}) 
        print(save_path + 'machine-'+dataset_name+'.mat')

