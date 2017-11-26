import pandas as pd
from scipy.spatial.distance import squareform, pdist
import numpy as np
import pylab as pl
from numpy.linalg import *
import matplotlib.pyplot as plt
import time
from mpl_toolkits.mplot3d import Axes3D

def generate_distance_matrix(df,e):
    start_time = time.time()
    distance_mat = pd.DataFrame(squareform(pdist(df.iloc[:, 1:])), columns=df.index, index=df.index)
    distance_mat[distance_mat > e] =np.inf
    print("Execution Time - Distance matrix : %s seconds " % (time.time() - start_time))
    return distance_mat.as_matrix()

def floydWarshall(graph):
    start_time = time.time()
    dist = graph
    length = len(dist)
    #print(dist)
    for k in range(length):
        for i in range(length):
            for j in range(length):
                dist[i][j] = min(dist[i][j],dist[i][k] + dist[k][j])
    print("Execution Time - Floyd : %s seconds " % (time.time() - start_time))
    return dist

def mds(d, dimensions = 2):
    start_time = time.time()
    n = len(d)
    d_square = d**2
    i = np.identity(n)
    ones = np.ones((n,n))
    center_mat = (i - ones)/n
    mat_to_svd = -0.5*(center_mat.dot(d_square).dot(center_mat))

    u,s,v = svd(mat_to_svd)
    reduced_mat = u * np.sqrt(s)
    print("Execution Time - MDS : %s seconds " % (time.time() - start_time))
    return reduced_mat[:,0:dimensions]

def swiss_roll(n):
    print('in swiss_roll my function')
    n_samples, n_features = n, 3
    rng = np.random.RandomState(0)
    p = rng.uniform(low=0, high=1, size=n_samples)
    q = rng.uniform(low=0, high=1, size=n_samples)
    data = np.zeros((n_samples, n_features))
    t = (3*np.pi/2)*(1+2*p)
    data[:, 0] = t * np.cos(t)
    data[:, 1] = t * np.sin(t)
    data[:, 2] = 30*q
    return (data,t)

def broken_swiss_roll():
    print('in swiss_roll my function')
    n_samples, n_features = 2000, 3
    n_turns, radius = 1.2, 1.0
    rng = np.random.RandomState(0)
    p = rng.uniform(low=0, high=1, size=n_samples)
    q = rng.uniform(low=0, high=1, size=n_samples)
    data = np.zeros((n_samples, n_features))
    for i in range(0,len(p)):
        if p[i]>2/5 and p[i]<4/5:
            print('in if')
            p[i] = np.random.uniform(low=0,high=1,size=1)
        else:
            print(p)
    # generate the 2D spiral data driven by a 1d parameter t
    t = (3*np.pi/2)*(1+2*p)
    print(type(t))
    for i in range(0,len(t)):
        if t[i]>2/5 and t[i]<4/5:
            print('in if')
            new_p = np.random.uniform(low=0,high=1,size=1)
            print(new_p)
        else:
            print(t[i])

    max_rot = n_turns * 2 * np.pi
    data[:, 0] = radius = t * np.cos(t)
    data[:, 1] = radius = t * np.sin(t)
    data[:, 2] = 30*q
    manifold = np.vstack((t * 2 - 1, data[:, 2])).T.copy()
    colors = manifold[:, 0]
    #print(colors)
    # rotate and plot original data
    sp = pl.subplot(211)
    U = np.dot(data, [[-.79, -.59, -.13],
                      [.29, -.57, .75],
                      [-.53, .56, .63]])
    sp.scatter(U[:, 1], U[:, 2], c=colors)
    sp.set_title("Original data")
    pl.show()

def helix(n):
    data = np.zeros((n, 3))
    rng = np.random.RandomState(0)
    p = rng.uniform(low=0, high=1, size=n)
    # Plot a helix along the x-axis
    theta_max = 8 * np.pi
    theta = np.linspace(0, theta_max, n)
    # data[:, 0] = theta
    # data[:, 1] = np.sin(theta)
    # data[:, 2] = np.cos(theta)
    data[:, 0] = (2+np.cos(8*p))*np.cos(p)
    data[:, 1] = (2+np.cos(8*p))*np.sin(p)
    data[:, 2] = np.sin(8*p)
    return data,p

def twin_peaks():
    n = 500
    peak_1 = np.zeros((n, 4))
    peak_2 = np.zeros((n, 4))
    rng = np.random.RandomState(0)
    p = rng.uniform(low=0, high=1, size=n)
    q = rng.uniform(low=0, high=1, size=n)
    # Plot a helix along the x-axis
    theta_max = 8 * np.pi
    theta = np.linspace(0, theta_max, n)
    peak_1[:, 0] = 1-2*p
    peak_1[:, 1] = np.sin(np.pi-2*np.pi*p)
    peak_1[:, 2] = np.tanh(3-6*q)
    manifold = np.vstack((p * 2 - 1, peak_2[:, 1])).T.copy()
    peak_1[:,3] = manifold[:,0]

    p = rng.uniform(low=0, high=1, size=n)
    q = rng.uniform(low=0, high=1, size=n)
    peak_2[:, 0] = 1 - 2 * p
    peak_2[:, 1] = np.sin(np.pi - 2 * np.pi * p)
    peak_2[:, 2] = np.tanh(3 - 6 * q)
    manifold = np.vstack((p * 2 - 1, peak_2[:, 1])).T.copy()
    peak_2[:, 3] = manifold[:, 0]


    colors = manifold[:, 0]

    sp = pl.subplot(211)
    sp.scatter(peak_1[:, 0], peak_1[:, 1], c=colors)
    sp.set_title("Original data")
    pl.show()
    return pd.DataFrame.append(peak_1,peak_2)

import random
def separete_data_for_classification(train_percentage,data):
    start_time = time.time()
    l = len(data)
    train_length = int(l*train_percentage*0.01)
    train_samples = random.sample(range(l), train_length)
    train = data[train_samples]
    all_index = np.arange(l)
    test_samples = np.setdiff1d(all_index, train_samples)
    test = data[test_samples]
    print("Saperating Data for classification, Execution Time : %s seconds " % (time.time() - start_time))
    return train,test

#helix()
#twin_peaks()
# train_samples = random.sample(range(100), 20)
#
# t = np.array([1, 1, 1, 2, 2, 3, 8, 3, 8, 8])
# print(np.nonzero(t==10))

from sklearn.neighbors import NearestNeighbors
def calculate_trustworthiness(l_data_set,h_data_set,k,n):
    start_time = time.time()
    l_nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(l_data_set)
    h_nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(h_data_set)
    l_distances, l_indices = l_nbrs.kneighbors(l_data_set)
    h_distances, h_indices = h_nbrs.kneighbors(h_data_set)
    sum = 0
    for i in range(len(l_indices)):
        l_points = l_indices[i]
        h_points = h_indices[i]
        diff = np.setdiff1d(l_points,h_points)
        for point in diff:
            itemindex = np.nonzero(l_points == point)
            sum = sum + n-itemindex[0]-k;
    rhs = 2*sum/(n*k*(2*n-3*k-1))
    trust_worthiness = 1 - rhs
    print("Execution Time - Trustworthiness : %s seconds " % (time.time() - start_time))
    print(rhs)
    print('trustworthiness',trust_worthiness[0])
    return trust_worthiness

def calculate_continuity(l_data_set,h_data_set,k,n):
    start_time = time.time()
    l_nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(l_data_set)
    h_nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(h_data_set)
    l_distances, l_indices = l_nbrs.kneighbors(l_data_set)
    h_distances, h_indices = h_nbrs.kneighbors(h_data_set)
    sum = 0
    for i in range(len(l_indices)):
        l_points = l_indices[i]
        h_points = h_indices[i]
        diff = np.setdiff1d(h_points,l_points)
        for point in diff:
            itemindex = np.nonzero(h_points == point)
            sum = sum + n-itemindex[0]-k;
    rhs = 2*sum/(n*k*(2*n-3*k-1))
    continuity = 1 - rhs
    print("Execution Time - continuity: %s seconds " % (time.time() - start_time))
    print(rhs)
    print('continuity',continuity[0])
    return continuity[0]
