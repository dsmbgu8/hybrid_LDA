# hybrid_lda.py
# Implementation of LDA-based hybrid metric learning
# Author: Brian D. Bue (bbue@rice.edu)
# Last modified: 5/14/12
#
# If you use this code in a publication, please cite the following paper:
# B. Bue and E. Merenyi, "An Adaptive Similarity Measure for Classification
# of Hyperspectral Signatures," IEEE Geoscience and Remote Sensing
# Letters, 2012.
#
# Copyright: 2010-2012 Rice University

from aux import *

def solveLDA(B,W,k,lambdav=0.0,convex=False):
    '''
    Calculates matrix A maximizing regularized LDA objective

    Arguments: 
    - B: D x D between class scatter matrix
    - W: D x D within class scatter matrix
    - k: rank of LDA solution
    
    Keyword arguments:
    - lambdav: regularization parameter \in [0,1]
    - convex: if true and k==1, return convex vector of A coefficients
    
    Returns:
    - A: D x k LDA transformation matrix 
    '''
    from numpy.linalg import eig, inv    
    from numpy import asmatrix, argsort, eye, sum
    
    B = asmatrix(B)
    W = asmatrix((1-lambdav)*W + lambdav*eye(W.shape[0]))
    eigvals,eigvecs = eig(inv(W)*B)
    maxi = argsort(eigvals)[-k:]
    A = eigvecs[:,maxi]

    if convex:
        if k==1:   
            # A = asmatrix(A)
            # So = (A.T*B*A) / (A.T*W*A)
            # print So, S, e1 # convex / non-convex solutions equal   
            # So = (A.T*B*A) / (A.T*W*A)   
            A /= sum(A)     # convex-ify solution
        else:
            print 'Error: can only convex-ify top eigenvector,',
            print 'returning nonconvex solution'
            
    return A

def alphaLDA(X,labels,lambdav=0.0,metrics=['SqEuclidean']):
    '''
    LDA-based hybrid metric learning. 
    
    Arguments: 
    - X: list of D representations of samples, each X[i] an n x d matrix
    - labels: list of n labels for each sample in X[i]
    
    Keyword arguments:
    - lambdav: regularization parameter \in [0,1]
    - metrics: list of 1 or D metric types for each representation
               see scipy.spatial.distance.cdist docs for possible types
    
    Returns:
    - A: \alpha coefficients        
    '''
    
    from numpy import asarray, zeros

    if lambdav < 0 or lambdav > 1:
        print "Error: lambdav must be in [0,1] range"
        exit()

    D = len(X)
    ulab = unique(labels)
    k = len(ulab)
    N = len(labels)
    Ni = classCounts(labels)

    B = zeros([D,D])
    W = zeros([D,D])

    if len(metrics)==1:
        metrics = [metrics[0]]*D
    
    for i in range(D):        
        Xi = X[i]
        Mi = classMeans(Xi,labels)
        Mui = vecmean(Mi)
        for j in range(i+1,D):
            Xj = X[j]
            Mj = classMeans(Xj,labels)
            Muj = vecmean(Mj)

            # calculate within-class scatter
            Wii, Wij, Wjj = 0,0,0
            for l in range(k):
                maskl = labels==ulab[l]
                Wdisti = distanceMatrix(Xi[maskl], Mi[l], metric=metrics[i])
                Wdistj = distanceMatrix(Xj[maskl], Mj[l], metric=metrics[i])
                Wii += sum(Wdisti**2)
                Wjj += sum(Wdistj**2)
                Wij += sum(Wdisti*Wdistj)

            # calculate between-class scatter
            Bdisti = distanceMatrix(Mi, Mui, metric=metrics[i])
            Bdistj = distanceMatrix(Mj, Muj, metric=metrics[i])
            Bii = sum(Ni*(Bdisti**2))
            Bjj = sum(Ni*(Bdistj**2))
            Bij = sum(Ni*(Bdisti*Bdistj))
            
            W[i,j] = W[j,i] = Wij
            B[i,j] = B[j,i] = Bij
            W[i,i],W[j,j] = Wii,Wjj
            B[i,i],B[j,j] = Bii,Bjj
            
    B /= N
    W /= N

    A = solveLDA(B,W,1,lambdav,convex=True)

    return asarray(A).flatten()

def minDistHybrid(X,lab,lambdas=[0],Dlab=[],verbose=False,norm='none'):
    '''
    Hybrid metric learning with minimum distance to class means classifier.
    Learns the LDA-based weight vector \alpha to combine distances
    between D representations of samples X.

    Arguments: 
    - X: list of samples in D representations, each X[i] is an n x d matrix
    - lab: n dimensional vector of labels for each sample in X
    
    Keyword arguments:
    - lambdas: list of lambda-values for regularization, default=[0] (no regularization)
      the best lambda value will be selected from this list according to its performance
      training set. 
    - verbose: print verbose output, default=False
    - Dlab: list of string labels for each of the D representations    
    - norm: normalization type ('none', 'L2' or 'std')
    '''
    
    from numpy import min,max,zeros,ones

    D = len(X)
    metrics=['SqEuclidean']*D # base metrics for each representation

    if len(Dlab) != D:
        Dlab = ['X%d'%i for i in range(D)]
    
    # 50/50 training set split
    tridx,teidx = stratSplit(lab,splitp=0.5)

    trlab,telab,ulab = lab[tridx],lab[teidx],unique(lab)
    
    trXN = [Xi[tridx] for Xi in X]
    teXN = [Xi[teidx] for Xi in X]

    # normalize each sample in each representation 
    if norm=='L2': # ...by euclidean norm
        trXN = [eucRowNorm(trXi) for trXi in trXN]
        teXN = [eucRowNorm(teXi) for teXi in teXN]
    elif norm=='std': # ...by std. dev. of each representation
        from numpy import sqrt, std
        vars = [std(Xi.flatten()) for Xi in X]
        trXN = [trXi/vari for trXi,vari in zip(trXN,vars)]
        teXN = [teXi/vari for teXi,vari in zip(teXN,vars)]        
    
    # training class means (per representation)
    trXNmu = [classMeans(trXNi,trlab) for trXNi in trXN]

    # distance matrices to training means in each representation
    trXNdmats = [distanceMatrix(trXN[i],trXNmu[i],metric=metrics[i]) 
                 for i in range(D)]
    teXNdmats = [distanceMatrix(teXN[i],trXNmu[i],metric=metrics[i]) 
                 for i in range(D)]

    bestalpha,bestlambda = ones(D)/D,-1
    trmaxacc = -1.0
    for lambdav in lambdas:
        alpha = alphaLDA(trXN,trlab,lambdav,metrics=metrics)
        if min(alpha) < 0 or max(alpha) > 1: # ill-posed eigendecomposition
            continue 

        # accumulate distance matrices for train/test points, weighted by alpha
        trHybrid_dmat = zeros(trXNdmats[0].shape)
        teHybrid_dmat = zeros(teXNdmats[0].shape)
        for i,a in enumerate(alpha):
            trHybrid_dmat += a*trXNdmats[i]
            teHybrid_dmat += a*teXNdmats[i]

        trpred = minDist(trHybrid_dmat,ulab)
        tepred = minDist(teHybrid_dmat,ulab)

        tracc = accuracy(trlab,trpred)
        teacc = accuracy(telab,tepred)

        if verbose:
            print 'lambda=',lambdav,'alpha=',alpha,
            print 'Tr hybrid acc=', tracc,
            print 'Te hybrid acc=', teacc

        if tracc > trmaxacc:
            trmaxacc = tracc
            bestalpha,bestlambda = alpha,lambdav

    if trmaxacc < 0:
        print 'Error: Could not determine a viable solution with provided data and regularization parameters'
        return

    for i,XNdmati in enumerate(trXNdmats):
        trpred = minDist(XNdmati,ulab)
        print 'Tr '+Dlab[i]+' acc=%0.3f'%accuracy(trlab,trpred),    
    print 'Tr hybrid acc=%0.3f'%trmaxacc

    for i,XNdmati in enumerate(teXNdmats):
        tepred = minDist(XNdmati,ulab)
        print 'Te '+Dlab[i]+' acc=%0.3f'%accuracy(telab,tepred),
    print 'Te hybrid acc=%0.3f'%temaxacc

    print 'best alpha=',bestalpha,
    print 'best lambda=',bestlambda
