### Set current working directory.
import os
#print(os.getcwd()) # this command will show your current directory. Ensure this is consistent across all files.
path = 'C:\\Users\\Reemon Spector\\Documents\\TuringTDA'
os.chdir(path)

### Import necessary libraries.
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colors as mcolors

### See e.g. Chapter 2.3 of Murray's 'Mathematical Biology II: Spatial Models and Biomedical Applications' for an explanation of the functions k2, h, bTerm lambdaplus and conditions.
### Define some functions to find the maximal real part of the eigenvalues of the reaction-diffusion operator.
def k2(m,n):
    return np.pi**2 * (m**2 + n**2) / 20**2 # here Lx = Ly = 20.

def h(k2,a,b,s,d):
    u,v = a/5, 1+a**2/25
    d = 1.5
    fu = (-u**4 + u**2 * (4*v - 2) - 4*v - 1)/((1+u**2)**2)
    fv = (-4*u)/(1+u**2)
    gu = b * s * ((u**4 + u**2 * (v + 2) - v + 1)/((1+u**2)**2))
    gv = (-b * s *u)/(1+u**2)
    return d*s * k2**2 - k2 * (d*s*fu + gv) + (fu*gv-fv*gu) # Caution: this is in k^2, not in k.

def bTerm(k2,a,b,s,d): #the linear term in the quadratic dispersion relation
    u,v = a/5, 1+a**2/25
    d = 1.5
    fu = (-u**4 + u**2 * (4*v - 2) - 4*v - 1)/((1+u**2)**2)
    gv = (-b * s *u)/(1+u**2) #these are correct.
    return k2*(1+d*s) - fu - gv

def lambdaplus(k2,a,b,s,d): #the biggest (real part) of the two eigenvalues
    x = bTerm(k2,a,b,s,d)
    y = h(k2,a,b,s,d)
    if x**2 - 4*y >= 0:
        return (-x + np.sqrt(x**2 - 4*y))/2
    else:
        return -x/2

### Check if a node is in the Turing space by checking it satisfies the conditions C1-C4.
def conditions(a,b,s,d):
    u,v = a/5, 1+a**2/25
    d = 1.5
    fu = (-u**4 + u**2 * (4*v - 2) - 4*v - 1)/((1+u**2)**2)
    fv = (-4*u)/(1+u**2)
    gu = b * s * ((u**4 + u**2 * (v + 2) - v + 1)/((1+u**2)**2))
    gv = (-b * s *u)/(1+u**2)
    C1 = (fu + gv < 0)
    C2 = (fu*gv - fv*gu > 0)
    C3 = (d*s*fu + gv > 0)
    C4 = ((d*s*fu + gv)**2 - 4*d*s*(fu*gv - fv*gu) > 0)
    return C1 and C2 and C3 and C4

### Define some colours for later use
all_colors = list(mcolors.TABLEAU_COLORS.values())
num_colors = 100
distinct_colors = [all_colors[i % len(all_colors)] for i in range(num_colors)]

### Fix some parameters.
s = 20
d = 1.5

### Plot the number of unstable modes at each node.
A = np.linspace(5,20,70)
B = np.linspace(0,2,35)
fig = plt.figure(2)
colour = 'black'
w = 8
cols = []
for a in A:
    for b in B:
        ws = []
        if conditions(a,b,s,d):
            for m in range(1):
                for n in range(m,11):
                    if [m,n]!=[0,0] and h(k2(m,n),a,b,s,d)<0: # i.e. not both zero and k^2(m,n) is between k+^2 and k-^2
                        ws.append((m,n))
        if ws != []:
            ls = [lambdaplus(k2(w[0],w[1]),a,b,s,d) for w in ws if lambdaplus(k2(w[0],w[1]),a,b,s,d)>0]
            lmax = np.max(ls)
            wmax = ws[np.argmax(ls)]
            colour = distinct_colors[len(ls)]
            plt.plot(a,b,c=colour,marker='$'+str(len(ls))+'$',markersize=w)
        else:
            wmax = 0 # meaning there are no unstable wavenumbers here, or it's outside the turing space

plt.xlabel('$\\alpha$')
plt.ylabel('$\\beta$')
plt.title('Number of unstable modes')
plt.tight_layout()
#plt.show()

### Plot the wavenumber of the most unstable Fourier mode.
A = np.linspace(5,20,25)
B = np.linspace(0,2,40)
fig = plt.figure(3)
colour = 'black'
w = 22
cols = []
for a in A:
    for b in B:
        ws = []
        if conditions(a,b,s,d):
            for m in range(21):
                for n in range(m,21):
                    if [m,n]!=[0,0] and h(k2(m,n),a,b,s,d)<0: # i.e. not both zero and k^2(m,n) is between k+^2 and k-^2
                        ws.append((m,n))
        if ws != []:
            ls = [lambdaplus(k2(w[0],w[1]),a,b,s,d) for w in ws if lambdaplus(k2(w[0],w[1]),a,b,s,d)>0]
            lmax = np.max(ls)
            wmax = ws[np.argmax(ls)]
            colour = distinct_colors[wmax[0]]
            plt.plot(a,b,c=colour,marker='$'+str((wmax[0],wmax[1]))+'$',markersize=w)
        else:
            wmax = 0 # meaning there are no unstable wavenumbers here, or it's outside the turing space

plt.xlabel('$\\alpha$')
plt.ylabel('$\\beta$')
plt.title('Most unstable Fourier wavenumber')
plt.tight_layout()
#plt.show()

### Plot the percentage difference between the two largest eigenvalues
A = np.linspace(5,20,32)
B = np.linspace(0,2,36)
fig = plt.figure(4)
w = 15
cols = []
for a in A:
    for b in B:
        ws = []
        if conditions(a,b,s,d):
            for m in range(21):
                for n in range(m,21):
                    if [m,n]!=[0,0] and h(k2(m,n),a,b,s,d)<0: #i.e. not both zero and k^2(m,n) is between k+^2 and k-^2
                        ws.append((m,n))
        if ws != []:
            colour = 'g'
            ls = [lambdaplus(k2(w[0],w[1]),a,b,s,d) for w in ws if lambdaplus(k2(w[0],w[1]),a,b,s,d)>0]
            ls.sort()
            l1 = ls[-1]
            if len(ls)>1:
                l2 = ls[-2]
            else:
                l2 = l1
            if int(round(l1/l2 - 1,2)*100)>10:
                colour = 'r'
            elif int(round(l1/l2 - 1,2)*100)>5:
                colour = 'orange'
            plt.plot(a,b,c=colour,marker='$'+str(int(round(l1/l2 - 1,2)*100))+'$%',markersize=w)

plt.xlabel('$\\alpha$')
plt.ylabel('$\\beta$')
plt.title('Percentage difference between two largest eigenvalues') 
plt.tight_layout()
#plt.show()

### Optionally, test for the maximal wavenumber wmax and therefore bound the stepsize above by Lx/(2*wmax).
### Adapt this for the Schnakenberg system by changing A,B and S to the appropriate ranges [0,0.5], [0,3] and [25,45]
# A = np.linspace(5,20,30)
# B = np.linspace(0,2,100)
# S = np.linspace(5,20,100)
# for a in A:
#     for b in B:
#         for s in S:
#             ws = []
#             if conditions(a,b,s,d):
#                 for m in range(21):
#                     for n in range(m,21):
#                         if [m,n]!=[0,0] and h(k2(m,n),a,b,s,d)<0: #i.e. not both zero and k^2(m,n) is between k+^2 and k-^2
#                             ws.append((m,n))
#             if ws != []:
#                 wmax = np.max(np.asarray(ws)[:,:])
#                 if wmax>10:
#                     print(wmax)
#             else:
#                 wmax = 0 #meaning there are no unstable wavenumbers here, or it's outside the turing space

### Read the .CSV files containing the barcodes
import pandas as pd
import json
csvBarcodes = pd.read_csv("CIMAbarcodes3.csv") #change back to 4
csvnpBarcodes = csvBarcodes.to_numpy()
standardised = np.zeros_like(csvnpBarcodes)
for i in range(np.shape(csvnpBarcodes)[0]):
    for j in range(4):
        standardised[i][j] = csvnpBarcodes[i][j].replace('\n',',').replace('.0 ','.0, ') #make sure there's no funny formatting

npbarcodes = np.array([[json.loads(standardised[i][j]) for j in range(4)] for i in range(len(standardised))],list)

### To plot the four histograms of barcode length, first record the lengths.
lengths = [[],[],[],[]]
for i in range(len(npbarcodes)):
    for j in range(4):
        for bar in npbarcodes[i][j]:
            lengths[j].append(bar[1]-bar[0])

### Plot, save and show the histograms with appropriate cutoffs.
plt.figure()
plt.axvline(6, color='red', linestyle='dashed', linewidth=1,label='Cutoff')
plt.hist(lengths[0],bins=17,label='dim 0 barcode lengths of $u$')
plt.xticks(np.array([i for i in range(21)]))
plt.xlabel('Barcode length')
plt.ylabel('Count')
plt.title('Histogram of barcode length in dimension 0 for $u$')
plt.savefig('dim0uhistogram.png', dpi=300)
#plt.show()

plt.figure()
plt.hist(lengths[3],bins=15,label='dim 1 barcode lengths of v')
plt.axvline(5, color='red', linestyle='dashed', linewidth=1,label='Cutoff')
plt.xticks(np.array([i for i in range(21)]))
plt.xlabel('Barcode length')
plt.ylabel('Count')
plt.title('Histogram of barcode length in dimension 1 for $v$')
plt.savefig('dim1vhistogram.png', dpi=300)
#plt.show()

plt.figure()
plt.hist(lengths[2],bins=17,label='dim 1 barcode lengths of $u$')
plt.xticks(np.array([i for i in range(21)]))
plt.xlabel('Barcode length')
plt.ylabel('Count')
plt.title('Histogram of barcode length in dimension 1 for $u$')
plt.savefig('dim1uhistogram.png', dpi=300)
#plt.show()

plt.figure()
plt.hist(lengths[1],bins=16,label='dim 0 barcode lengths of v')
plt.xticks(np.array([i for i in range(21)]))
plt.xlabel('Barcode length')
plt.ylabel('Count')
plt.title('Histogram of barcode length in dimension 0 for $v$')
plt.savefig('dim0vhistogram.png', dpi=300)
plt.show()
