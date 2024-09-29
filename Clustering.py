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
    max_clust = 7
    linkageMethod = 'weighted'
    zlabel = '$\\sigma$'
    if pde == 'CIMA':
        csvBarcodes = pd.read_csv(r"CIMAbarcodes3.csv")
        nodes = pd.read_csv(r"CIMAnodes3.csv")
    elif pde == 'CIMAsigma20':
        csvBarcodes = pd.read_csv(r"CIMAbarcodes.csv")
        nodes = pd.read_csv(r"CIMAnodes.csv")
elif pde == 'Schnak':
    max_clust = 9
    csvBarcodes = pd.read_csv(r"Schnakbarcodes.csv")
    nodes = pd.read_csv(r"Schnaknodes.csv")
    linkageMethod = 'average'
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

def d(a,b):
    u0 = wass(a,b)[0]
    v0 = wass(a,b)[1]
    u1 = wass(a,b)[2]
    v1 = wass(a,b)[3]
    return u0+v0+u1+v1

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
cutoffs = {'CIMA':[6,5], 'S':[4,3], 'CIMAsigma20':[6,5]} #0 ~ u0, 1 ~ v1
for i in range(len(npbarcodes)):
    for j in range(4): #0 ~ u0, 1 ~ v0, 2 ~ u1, 3 ~ v1
        for bar in npbarcodes[i][j]:
            if (bar[1]-bar[0])>=cutoffs[pde][0] and j==0:
                clean[i][j].append(np.array(bar))
            elif (bar[1]-bar[0])>=cutoffs[pde][1] and j==3:
                clean[i][j].append(np.array(bar))
            elif j==1 or j==2:
                clean[i][j].append(np.array(bar))

#import time
#t0 = time.time()
### Compute the distance matrices for both the standard and cleaned barcodes.
distMatrixClean = parallel_pdist(clean, metric=d)
distMatrix = parallel_pdist(npbarcodes, metric=d) #pdist(npbarcodes, metric=d)
#print("finished computing distance matrix in "+str(time.time()-t0)+"s")

### To decide which linkage / whether to use the standard or cleaned barcodes.
for j in [0,1]:
    for c in [7]:
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
colours = ['black','magenta','orange','red','green','blue','brown','darkgreen','darkblue']
if pde == 'CIMA' or pde == 'Schnak':
    fig = plt.figure(figsize=(6.875,5))
    ax = fig.add_subplot(projection='3d')
    for i in range(N):
        sizes = [6 for i in range(max_clust)]
        m = '$'+str(clusters[i])+'$'
        ax.plot(npnodes[i,0],npnodes[i,1],npnodes[i,2],c=colours[clusters[i]-1],
                marker=m,markersize=sizes[clusters[i]-1])

    ax.set_xlabel('$\\alpha$')
    ax.set_ylabel('$\\beta$')
    ax.set_zlabel(zlabel)
    ax.view_init(elev=68,azim=-142)
    plt.tight_layout()
    plt.savefig(pde+'Clustering.png', dpi=300)
#plt.show()

### Optionally, observe the dendrogram obtained by clustering, for a quick check that the clustering is "good".
# from scipy.cluster import hierarchy
# from scipy.cluster.hierarchy import dendrogram
# hierarchy.set_link_color_palette(colours)
# fig = plt.figure(figsize=(6.875,5))
# dn = dendrogram(linked, leaf_font_size=0, color_threshold=linked[-max_clust][2]+0.00001)
# plt.tight_layout()
# plt.savefig(pde+'Dendrogram.png', dpi=300)
# plt.show()

### Plot the figure showing the clustering when sigma == 20.
if pde == 'CIMAsigma20':
    fig = plt.figure()
    swapped_clusters = [1,4,7,6,5,2,3]#[5,3,6,7,1,2,4] # so that the numbering and colourscheme agrees with the rest of the paper.
    for i in range(N-1,-1,-1):
        if npnodes[i,2]==20:
            plt.plot(npnodes[i,0],npnodes[i,1],c=colours[swapped_clusters[clusters[i]-1]-1],marker='$'+str(swapped_clusters[clusters[i]-1])+'$',markersize=8)

    plt.xlabel("$\\alpha$")
    plt.ylabel("$\\beta$")
    plt.title("Clusters obtained from hierarchical clustering of barcodes")
    plt.tight_layout()
    plt.savefig('clusteringSigma20.png',dpi=300)
#plt.show()

### Plot the figure showing the difference between the minimum and maximum values of u for each node.
if pde == 'CIMA':
    npMinMax = (pd.read_csv("CIMAminmax3.csv")).to_numpy()
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for i in range(N):
        if npMinMax[i][1]-npMinMax[i][0]>2.0197: # 25th percentile, obtained using np.quantile([dfMinMax[1][i] - dfMinMax[0][i] for i in range(549)],0.25).
            col = 'green'
        elif npMinMax[i][1]-npMinMax[i][0]>0.1510: # 10th percentile, obtained using np.quantile([dfMinMax[1][i] - dfMinMax[0][i] for i in range(549)],0.25).
            col = 'orange'
        else:
            col = 'red'
        ax.plot(npnodes[i,0],npnodes[i,1],npnodes[i,2],c=col,marker='o',markersize=7)

    ax.set_xlabel('$\\alpha$')
    ax.set_ylabel('$\\beta$')
    ax.set_zlabel("$\\sigma$")
    #plt.title("Difference between maximum and minimum of $u$")
    plt.tight_layout()
    ax.view_init(elev=64,azim=-142)
    plt.savefig('CIMAmaxmindiff.png',dpi=300)

plt.show()