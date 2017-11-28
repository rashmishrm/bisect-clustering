
# coding: utf-8

# In[1]:


import numpy as np

import scipy as sp
from scipy import spatial
from scipy.sparse import *
from collections import defaultdict
from random import uniform
from math import sqrt

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import silhouette_score

vectorizer = CountVectorizer()

from collections import defaultdict
from math import sqrt
import random

import matplotlib.pyplot as plt
from collections import defaultdict
from collections import Counter
from scipy.sparse import csr_matrix, find
import numpy as np
import random
from sklearn.utils import shuffle
from sklearn.metrics import calinski_harabaz_score


# In[2]:


def csr_read(fname, ftype="csr", nidx=1):
    r"""
        Read CSR matrix from a text file.

        \param fname File name for CSR/CLU matrix
        \param ftype Input format. Acceptable formats are:
            - csr - Compressed sparse row
            - clu - Cluto format, i.e., CSR + header row with "nrows ncols nnz"
        \param nidx Indexing type in CSR file. What does numbering of feature IDs start with?
    """

    with open(fname) as f:
        lines = f.readlines()

    if ftype == "clu":
        p = lines[0].split()
        nrows = int(p[0])
        ncols = int(p[1])
        nnz = long(p[2])
        lines = lines[1:]
        assert(len(lines) == nrows)
    elif ftype == "csr":
        nrows = len(lines)
        ncols = 0
        nnz = 0
        for i in xrange(nrows):
            p = lines[i].split()
            if len(p) % 2 != 0:
                raise ValueError("Invalid CSR matrix. Row %d contains %d numbers." % (i, len(p)))
            nnz += len(p)/2
            for j in xrange(0, len(p), 2):
                cid = int(p[j]) - nidx
                if cid+1 > ncols:
                    ncols = cid+1
    else:
        raise ValueError("Invalid sparse matrix ftype '%s'." % ftype)
    val = np.zeros(nnz, dtype=np.float)
    ind = np.zeros(nnz, dtype=np.int)
    ptr = np.zeros(nrows+1, dtype=np.long)
    n = 0
    for i in xrange(nrows):
        p = lines[i].split()
        for j in xrange(0, len(p), 2):
            ind[n] = int(p[j]) - nidx
            val[n] = float(p[j+1])
            n += 1
        ptr[i+1] = n

    assert(n == nnz)

    matrix = csr_matrix((val, ind, ptr), shape=(nrows, ncols), dtype=np.float)
    matrix.sort_indices()
    return matrix


# scale matrix and normalize its rows
def csr_idf(mat, copy=False, **kargs):
    r""" Scale a CSR matrix by idf.
    Returns scaling factors as dict. If copy is True,
    returns scaled matrix and scaling factors.
    """
    if copy is True:
        mat = mat.copy()
    nrows = mat.shape[0]
    nnz = mat.nnz
    ind, val, ptr = mat.indices, mat.data, mat.indptr
    # document frequency
    df = defaultdict(int)
    for i in ind:
        df[i] += 1
    # inverse document frequency
    for k,v in df.items():
        df[k] = np.log(nrows / float(v))  ## df turns to idf - reusing memory
    # scale by idf
    for i in range(0, nnz):
        val[i] *= df[ind[i]]

    return df if copy is False else mat

def csr_l2normalize(mat, copy=False, **kargs):
    r""" Normalize the rows of a CSR matrix by their L-2 norm.
    If copy is True, returns a copy of the normalized matrix.
    """
    if copy is True:
        mat = mat.copy()
    nrows = mat.shape[0]
    nnz = mat.nnz
    ind, val, ptr = mat.indices, mat.data, mat.indptr
    # normalize
    for i in range(nrows):
        rsum = 0.0
        for j in range(ptr[i], ptr[i+1]):
            rsum += val[j]**2
        if rsum == 0.0:
            continue  # do not normalize empty rows
        rsum = float(1.0/np.sqrt(rsum))
        for j in range(ptr[i], ptr[i+1]):
            val[j] *= rsum

    if copy is True:
        return mat





def initCentorids(x, k):
    x_d=x.todense()
    indices = np.random.choice(x_d.shape[0], k, replace=False)
    #x_shuffle = shuffle(x, random_state=42)
    #return x_shuffle[:k,:]
    return x[indices,:],indices

# In[103]:


def sim(x1, x2):
    #print "#####################"
    #print x1.shape
    #print x2.shape
    #print "#####################"
    sims = x1.dot(x2.T)
    return sims


def findCentroids(mat, centroids):
    idx = list()

    simsMatrix = sim(mat, centroids)

    for i in range(simsMatrix.shape[0]):
        row = simsMatrix.getrow(i).toarray()[0].ravel()

        top_indices = row.argsort()[-1]
        top_values = row[row.argsort()[-1]]
#         print top_indices
        idx.append(top_indices + 1)

    return idx


# In[106]:
def kmeans_b(k, mat_orig,indx_m, n_iter=10,epoch=10):
    print "Kmeans..."
    mat_m = list()
    mat_m = mat_orig[indx_m,:]
    #print mat_m.shape

    #mat = csr_matrix(mat_m)
    new_sse = np.inf
    min_sse = np.inf
    min_mat_indx=[None]* (k+1)
    indices_init=list()
    final_indices=list()
    for e in range(1,epoch):
        #print "epoch..."+str(e)

        centroids,indices_init = initCentorids(mat_m, k)
        old_sse = np.inf
        gain = np.inf
        mat_indx=[None]* (k+1)

        for _ in range(n_iter):
            idx = findCentroids(mat_m, centroids)
            centroids = computeMeans(mat_m, idx, k)
            if(centroids==None):
                break


            for i in range(1,k+1):
                indi = [j for j, x in enumerate(idx) if x == i]
                #print indi
                indo = list()
                for m in indi:
                    indo.append(indx_m[m])
                    mat_indx[i]=indo
            old_sse=new_sse
            new_sse=sum_sse(mat_orig,mat_indx)
            #print new_sse
            gain = old_sse - new_sse
            #print gain
            if(new_sse<min_sse):
                min_sse=new_sse
                min_mat_indx=mat_indx
                final_indices=indices_init

            if(gain < 0.01):
                break;

        #input_k, highest_sse_k,new_sse = max_sse_cluster(mat_orig,mat_indx,k)

    print min_sse
    print final_indices
    return idx, min_mat_indx





# In[108]:


def computeMeans(mat, idx, k):
    centroids = list()
    for i in range(1,k+1):
        indi = [j for j, x in enumerate(idx) if x == i]
        members = mat[indi,:]
        #print members.shape[0]
        if (members.shape[0] > 1):
            centroids.append(members.toarray().mean(0))
    #print centroids
    centroids_csr=None
    if(len(centroids)>0):
        centroids_csr = csr_matrix(centroids)
    return centroids_csr


# In[109]:


def sse(mat,idx):
    centroids = list()
    #print mat.shape
    members = mat[idx,:]
    #print members
    if (members.shape[0] > 1):
        centroids=members.toarray().mean(0)
    #print centroids
    if(len(centroids)>0):
        return np.sum(np.linalg.norm(mat - centroids, 2, 1))
    else:
        return 0


def max_sse_cluster(mat,mat_idx, k):
    #print "max_sse_cluster"
    #print "MMM"
    #print mat_idx
    sselist=list()
    #print mat_idx
    #print mat_idx[1]
    #print mat.shape
    for i in range(1,(len(mat_idx))):
        #print i
        ind=mat_idx[i]

        #print ind
        if(ind is not None and len(ind)>0):
            #print mat.shape
            sselist.append(sse(mat,ind))
    #print sselist
    ssesum= np.sum(sselist)
    arr=np.array(sselist)
    highest_sse_k=0
    if(len(sselist)>0):
        highest_sse_k = np.argsort(arr)[-1]+1
        #print highest_sse_k

    return mat_idx[highest_sse_k], highest_sse_k, ssesum


def sum_sse(mat,mat_idx):
    sselist=list()
    for i in range(1,(len(mat_idx))):
        ind=mat_idx[i]
        if(ind is not None and len(ind)>0):
            sselist.append(sse(mat,ind))
    ssesum= np.sum(sselist)
    return ssesum


# In[110]:



def bisect(data,k,n_iter=10, epoch=10):


    k_b=1

    bmat_indx = [None]*(k_b+1)
    bmat_indx[1] = np.arange(0, data.shape[0], 1)
    old_sse=np.inf
    curr_sse=np.inf
    while True:
    #max sse
        print "starting... bisect:"
    #print bmat_indx

        old_sse=curr_sse
        #print bmat_indx
        #print data.shape
        input_k, highest_sse_k,curr_sse = max_sse_cluster(data,bmat_indx,k_b)
        gain = curr_sse-old_sse

        bmat_indx.pop(highest_sse_k)
        inc=1
        idx,mat_indx = kmeans_b(2, data,input_k, n_iter,epoch)
        if(mat_indx is not None and len(mat_indx)>0):
            if( (mat_indx[1] is not None) and  (len(mat_indx[1])>0)):
                bmat_indx.append(mat_indx[1])
                inc=0
            if( (mat_indx[2] is not None) and  (len(mat_indx[2])>0)):
                bmat_indx.append(mat_indx[2])
                inc=0
            #print inc
        if(inc==1):
            k_b=k_b+1
        if(len(bmat_indx)==8 or k_b>epoch):
            break

    return bmat_indx



# In[3]:


csr_mat = csr_read("train.dat")

csridfmat = csr_idf(csr_mat, copy=True)
csrnorm = csr_l2normalize(csridfmat, copy=True)


bmat_indx = bisect(csrnorm,7,20,20)

listi=[None]*csrnorm.shape[0]

for i in range(1,8):
    for j in bmat_indx[i]:
        listi[j]=i


def printResult(idx):
    text_file = open("output7.dat", "w")
    for i in idx:

        text_file.write(str(i) +'\n')
    text_file.close()

printResult(kmeans.labels_)


print "Final Score: "
print(calinski_harabaz_score(csrnorm.toarray(), listi))
