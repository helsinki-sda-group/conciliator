from scipy.spatial.distance import cdist
from scipy.special import logsumexp
import numpy as np

def shepard_weights(points,coords,p=1,metric='cityblock'):
    dists = cdist(points,coords,metric=metric)
    zero_rows, zero_columns = np.where(dists == 0.0)
    if len(zero_rows) != 0:
        dists = np.delete(dists,zero_rows,axis=0)             
    log_dists = p * np.log(dists)
    log_products = np.sum(log_dists,axis=1).reshape((-1,1)) - log_dists
    log_weights = log_products - logsumexp(log_products,axis=1).reshape((-1,1))
    weights = np.exp(log_weights)
    for r in range(len(zero_rows)):
        temp = np.zeros(len(coords))
        temp[zero_columns[r]] = 1.0
        weights = np.insert(weights,zero_rows[r],temp,axis=0)
    return weights

def normal_distance_weighting(points,coords,p=1,metric='cityblock',normalize=False):
    dists = cdist(points,coords,metric=metric)
    zero_rows, zero_columns = np.where(dists == 0.0)
    weights = np.exp(-dists**p)
    for r in range(len(zero_rows)):
        weights[r,:] = np.zeros(len(coords))
        weights[r,zero_columns[r]] = 1.0
    if normalize:
        weights = weights / np.sum(weights,axis=1)
    return weights