### Set current working directory.
import os
#print(os.getcwd()) # this command will show your current directory. Ensure this is consistent across all files.
path = 'C:\\Users\\Reemon Spector\\Documents\\TuringTDA'
os.chdir(path)

### Import necessary libraries, notably GUDHI, scipy, pandas and sklearn.
from scipy.cluster.hierarchy import linkage, fcluster
import numpy as np
import pandas as pd
from gudhi.wasserstein import wasserstein_distance
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from joblib import Parallel, delayed
import json
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import squareform

### Choose the PDE, and set the maximum number of clusters appropriately.
pde = 'CIMA' # can be 'CIMA', 'CIMA20' or 'Schnak'
if pde=='CIMA' or 'CIMAsigma20':
    max_clust = 7 # determined after testing with different numbers of clusters
    linkageMethod = 'weighted' # determined after testing with different linkage methods (see below)
    zlabel = '$\\sigma$'
    if pde == 'CIMA':
        csvBarcodes = pd.read_csv(r"CIMAbarcodes3.csv")
        nodes = pd.read_csv(r"CIMAnodes3.csv")
    elif pde == 'CIMAsigma20':
        csvBarcodes = pd.read_csv(r"CIMAbarcodes.csv")
        nodes = pd.read_csv(r"CIMAnodes.csv")
elif pde == 'Schnak':
    max_clust = 9 # determined after testing with different numbers of clusters
    csvBarcodes = pd.read_csv(r"Schnakbarcodes.csv")
    nodes = pd.read_csv(r"Schnaknodes.csv")
    linkageMethod = 'average' # determined after testing with different linkage methods (see below)
    zlabel = '$\\delta$'
else:
    print("input a valid PDE. Options are 'Schnak' for Schnakenberg, 'CIMA' for CIMA, or 'CIMAsigma20' for CIMA restricted to the case where sigma == 20.")

### Convert the .CSV file barcodes obtained from CIMA.py or Schnak.py into lists.
csvnpBarcodes = csvBarcodes.to_numpy()
standardised = np.zeros_like(csvnpBarcodes)
for i in range(np.shape(csvnpBarcodes)[0]):
    for j in range(4):
        standardised[i][j] = csvnpBarcodes[i][j].replace('\n',',').replace('.0 ','.0, ') # make sure there's no funny formatting.

npbarcodes = np.array([[json.loads(standardised[i][j]) for j in range(4)] for i in range(len(standardised))],list)
N = len(npbarcodes)

### Convert the .CSV file node coordinates into a matrix.
npnodes = nodes.to_numpy()

### Define the metric used on the barcodes, namely the sum of 2-Wasserstein distances.
def wass(a,b):
    au0 = np.array(a[0])
    av0 = np.array(a[1])
    au1 = np.array(a[2])
    av1 = np.array(a[3])
    bu0 = np.array(b[0])
    bv0 = np.array(b[1])
    bu1 = np.array(b[2])
    bv1 = np.array(b[3])
    dists = [wasserstein_distance(au0,bu0,order=2.0,internal_p=2.0), wasserstein_distance(av0,bv0,order=2.0,internal_p=2.0), wasserstein_distance(au1,bu1,order=2.0,internal_p=2.0), wasserstein_distance(av1,bv1,order=2.0,internal_p=2.0)]
    return dists

### Define a parallelised algorithm for computing the pairwise distances between nodes (using the metric above).
def parallel_pdist(X, metric):
    n = len(X)
    indices = np.triu_indices(n,k=1)
    n_jobs = -2  # Use all but one of available cores (set to -1 to use all)
    distances = Parallel(n_jobs=n_jobs)(delayed(metric)(X[i], X[j]) for i, j in zip(*indices))
    dist_matrix = np.zeros(n*(n-1)//2)
    dist_matrix[:] = distances
    return dist_matrix

### Clean the barcodes of 'short' bars, as defined by the cutoffs.
clean = [[[] for j in range(4)] for i in range(len(npbarcodes))]
cutoffs = {'CIMA':[6,5], 'S':[4,3], 'GM':[0,0]} #0 ~ u0, 1 ~ v1
for i in range(len(npbarcodes)):
    for j in range(4): #0 ~ u0, 1 ~ v0, 2 ~ u1, 3 ~ v1
        for bar in npbarcodes[i][j]:
            if (bar[1]-bar[0])>=cutoffs[pde][0] and j==0:
                clean[i][j].append(np.array(bar))
            elif (bar[1]-bar[0])>=cutoffs[pde][1] and j==3:
                clean[i][j].append(np.array(bar))
            elif j==1 or j==2:
                clean[i][j].append(np.array(bar))
def d1(a,b):
    u0 = wass(a,b)[0]
    v0 = wass(a,b)[1]
    u1 = wass(a,b)[2]
    v1 = wass(a,b)[3]
    return u0+v0+u1+v1

def d2(a,b):
    u0 = wass(a,b)[0]
    v0 = wass(a,b)[1]
    u1 = wass(a,b)[2]
    v1 = wass(a,b)[3]
    return np.sqrt(u0**2+v0**2+u1**2+v1**2)

def dinf(a,b):
    u0 = wass(a,b)[0]
    v0 = wass(a,b)[1]
    u1 = wass(a,b)[2]
    v1 = wass(a,b)[3]
    return max(u0,v0,u1,v1)

#import time
#t0 = time.time()
### Compute the distance matrices for both the standard and cleaned barcodes.
distMatrixClean1 = parallel_pdist(clean, metric=d1)
distMatrix1 = parallel_pdist(npbarcodes, metric=d1)
distMatrixClean2 = parallel_pdist(clean, metric=d2)
distMatrix2 = parallel_pdist(npbarcodes, metric=d2)
distMatrixCleaninf = parallel_pdist(clean, metric=dinf)
distMatrixinf = parallel_pdist(npbarcodes, metric=dinf)
#print("finished computing distance matrix in "+str(time.time()-t0)+"s")

for i in [0,1,2]:
    distMatrix, distMatrixClean = [[distMatrix1, distMatrixClean1],[distMatrix2, distMatrixClean2],[distMatrixinf, distMatrixCleaninf]][i]
    ### To decide which linkage / whether to use the standard or cleaned barcodes.
    print(["\n1 norm","\n2 norm","\ninfinity norm"][i])
    swapped_clusters = [[1,2,3,4,5,6,7],[1,4,6,7,2,3,5],[6,4,1,7,2,5,3]][i]
    for j in [0,1]:
        for c in [7]: # in this format to allow a comparison with fewer/more clusters.
            for m in ['single','complete','average','weighted']:
                dm = [squareform(distMatrix), squareform(distMatrixClean)][j]
                linked = linkage(dm, method=m)
                clusters = fcluster(linked, c, criterion='maxclust')
                score = silhouette_score(dm, clusters, metric='precomputed')
                print(["standard, ","cleaned, "][j]+str(c)+" clusters, "+m+" linkage, score: "+str(round(score,3)))
    ### Generate the dendrogram obtained by hierarchical clustering, and define the clusters.
    linked = linkage(distMatrixClean, method=linkageMethod)
    clusters = fcluster(linked, max_clust, criterion='maxclust')
    ### Plot the 3D figures showing the clustering in the Turing space, where appropriate (i.e. for 'CIMA' and 'Schnak' but not 'CIMAsigma20').
    if pde == 'CIMA' or pde == 'Schnak':
        colours = ['black','magenta','orange','red','green','blue','brown','darkgreen','darkblue']
        fig = plt.figure(figsize=(6.875,5))
        ax = fig.add_subplot(projection='3d')
        for k in range(N):
            sizes = [6 for i in range(max_clust)]
            m = '$'+str(swapped_clusters[clusters[k]-1])+'$'
            _ = ax.plot(npnodes[k,0],npnodes[k,1],npnodes[k,2],c=colours[swapped_clusters[clusters[k]-1]-1],marker=m,markersize=sizes[clusters[k]-1])
        _ = ax.set_xlabel('$\\alpha$')
        _ = ax.set_ylabel('$\\beta$')
        _ = ax.set_zlabel(zlabel)
        ax.view_init(elev=68,azim=-142)
        plt.tight_layout()
        plt.savefig(pde+["1","2","infinity"][i]+'NormClustering.png', dpi=300)

plt.show()