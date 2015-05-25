# aux.py
# Misc. utility functions for hybrid_lda.py
#
# Author: Brian D. Bue (bbue@rice.edu)
# Last Modified: 5/14/12
#
# Notes: All matrices/vectors are assumed to be numpy arrays
#
# Copyright: 2010-2012 Rice University

from numpy import asarray, unique, where, sum, min, max

def loadmtx(infile):
    '''
    Reads a Matlab format matrix from file
    '''
    from scipy.io import loadmat
    # set struct_as_record to false to get rid of compatability warning
    return loadmat(infile,struct_as_record=False)
    
def randperm(*args,**kwargs):
    '''
    Returns a random permutaton of a given dimensionality
    (emulates the Matlab "randperm" function).
    '''
    from numpy.random import permutation
    return permutation(*args,**kwargs)

def vecmean(x,axis=0):
    '''
    Mean vector along provided axis
    '''
    from numpy import mean
    return mean(x,axis=axis)

def vecstd(x,axis=0):
    '''
    Standard deviation vector along provided axis
    '''
    from numpy import std
    return std(x,axis=axis)

def eucRowNorm(mtx):
    '''
    Normalizes each row of mtx by the L2 norm of that row
    '''
    from numpy.linalg import norm
    from numpy import apply_along_axis
    normvals = apply_along_axis(norm,1,mtx)
    normvals[normvals==0] = 1.0 # avoid divide by zero
    return (mtx.T/normvals).T 

def classCounts(lab):
    '''
    Returns the count of unique values in lab, sorted by value
    '''
    return asarray([sum(lab==i) for i in unique(lab)])

def classMeans(dat,lab):
    '''
    Returns a [K,d] matrix of class means, where K=len(unique(lab))
    '''
    return asarray([vecmean(dat[where(lab==i)]) for i in unique(lab)])    

def distanceMatrix(data1,data2,metric='SqEuclidean'):
    '''
    [N,M] distance matrix between N data1 vs. M data2 vectors
    data1.shape[1] and data2.shape[1] must be equal.
    '''
    from scipy.spatial.distance import cdist
    from numpy import atleast_2d
    # flat vectors cause problems with cdist, reshape
    return cdist(atleast_2d(data1),atleast_2d(data2),metric=metric)
    
def minDist(dmtx_mean,meanlab):
    '''
    Minimum distance to class means classifier

    Arguments:
    - dmtx_mean = [N,K] distanceMatrix(test_data,training_means), where
      test_data = [N,d] array and training_means = [K,d] array

    - meanlab = list of K labels for each training mean
    '''
    from numpy import argmin
    return meanlab[argmin(dmtx_mean,axis=1)]        

def accuracy(des,pred):
    '''
    Returns the percentage of equal values in des and pred
    '''
    return sum(des==pred)/float(len(des))

def stratSplit(labels, splitp=0.5, verbose=False):
    '''
    Selects a set of train/test indices given the split percentage for
    stratified sampling.
    
    Arguments: 
    - labels: n-dim vector of labels for each sample
    - splitp: split percentage (default 50/50 split)

    Returns:
    - trpos: indices in 'labels' vector of training samples
    - tepos: indices in 'labels' vector of training samples
    '''
    from numpy import where, asarray, unique, r_
    
    teidx,tridx = asarray([],int),asarray([],int)    
    for i,lab in enumerate(unique(labels)):        
        # get the indices matching label 'lab'
        lidx, = where(labels==lab)
        nl = len(lidx)
        if nl == 0:
            print 'warning: no samples for label %s, skipping'%str(lab)
            continue
        
        # randomly split into training/testing indices
        lidx = lidx[randperm(nl)]
        ntr = max([1,int(nl*splitp)]) # minimum 1 training sample/class         
        nte = nl-ntr

        tridx = r_[tridx,lidx[:ntr]]
        if nte>0:
            teidx = r_[teidx,lidx[ntr:nl]]

        if verbose:
            print 'sampled label %s (%d/%d train/test)'%(str(lab),ntr,nte)

    if verbose:
        print 'sampled %d/%d total train/test points'%(len(tridx),len(teidx))
        
    return tridx, teidx

def collectSamples(selected_labs, labs, nkeep=None, order='none'):
    """
    Selects nkeep samples of each label in selected_labs from labs 
    """
    from numpy import r_
    if nkeep is None:
        from numpy import inf
        nkeep = inf
    
    selected_idx = asarray([],int)
    for u in selected_labs:        
        class_idx, = where(labs==u)
        if order=='reverse':
            class_idx = class_idx[::-1]
        elif order=='random':
            class_idx = class_idx[randperm(len(class_idx))]            
        
        class_idx = class_idx[:min([nkeep,len(class_idx)-1])]
        selected_idx = r_[selected_idx,class_idx]

    return selected_idx
