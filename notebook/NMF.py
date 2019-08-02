# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 1.0.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
from jurbey.jurbey import JURBEY
from sklearn.decomposition import NMF
import networkx as nx
import numpy as np

f = open('1558537930325.jurbey','rb')
g = JURBEY.load(f.read())

for e in g.edges(data=True):
    try:
        g[e[0]][e[1]]["speed"] = float(g[e[0]][e[1]]["data"].metadata.get("maxspeed",10))
    except ValueError:
        g[e[0]][e[1]]["speed"] = 10
# -

A = nx.adjacency_matrix(g,weight="speed")
node_list = list(g.nodes())
edge_list = list(g.edges())

from sklearn.preprocessing import MaxAbsScaler
transformer = MaxAbsScaler().fit(A)
A_norm = transformer.transform(A)

model_mu = NMF(n_components=50, init='random', random_state=0, verbose=True, solver='mu', tol=0.000001)
W_mu = model.fit_transform(A_norm)

model_cd = NMF(n_components=50, init='nndsvda', random_state=0, verbose=True, solver='cd', tol=0.0001)
W_cd = model_cd.fit_transform(A_norm)

# +
actual = A_norm[node_list.index(edge_list[1000][0]), node_list.index(edge_list[1000][1])]
pred_mu = np.dot(W_mu[node_list.index(edge_list[1000][0])],model.components_.T[node_list.index(edge_list[1000][1])])
pred_cd = np.dot(W_cd[node_list.index(edge_list[1000][0])],model.components_.T[node_list.index(edge_list[1000][1])])

print(f'actual: {actual} and pred mu: {pred_mu} and pred cd: {pred_cd}')


# -

def mf(X, K, iterations=1000, alpha=0.0002, beta=0.02):
    print(X.shape)
    m,n = X.shape
    W = np.random.rand(m,K)
    H = np.random.rand(n,K)
    H = H.T
    for _iter in range(iterations):
        for i in range(m):
            for j in range(n):
                if X[i, j] > 0 and not np.isnan(X[i, j]):
                    eij = X[i,j] - np.dot(W[i, :], H[:, j])
                    for k in range(K):
                        # multiplicative update + L2 regularization
                        W[i][k] = W[i][k] + alpha * (2 * eij * H[k][j] - beta * W[i][k])
                        H[k][j] = H[k][j] + alpha * (2 * eij * W[i][k] - beta * H[k][j])
        tol_list = list()                
        tol = 0
        for i in range(m):
            for j in range(n):
                if X[i, j] > 0 and not np.isnan(X[i, j]):
                    # Frobenius norm
                    tol = tol + pow(X[i,j] - np.dot(W[i, :], H[:, j]), 2)
                    for k in range(K):
                        tol = tol + (beta / 2) * (pow(W[i][k], 2) + pow(H[k][j], 2))
        print(f'At iter: {_iter}, tol: {tol}')
        tol_list.append(tol)
        if tol < 0.001:
            break
    return W, H.T, tol_list


import random
sampled_nodes = random.sample(g.nodes, 1000)
g_sampled = nx.DiGraph(g).subgraph(sampled_nodes)
A = nx.adjacency_matrix(g_sampled,weight="speed").todense()


print(np.shape(A))
print(A[0,1])

W,H = mf(A, 10)

print(np.shape(W))

# +
node_list = list(g_sampled.nodes())
edge_list = list(g_sampled.edges())
print(np.shape(W))
print(np.shape(H))
print(len(edge_list))
print(W[node_list.index(edge_list[1][0])])
print(H[node_list.index(edge_list[1][1])])
actual = A[node_list.index(edge_list[1][0]), node_list.index(edge_list[1][1])]
pred = np.dot(W[node_list.index(edge_list[1][0])],H[node_list.index(edge_list[1][1])])

print(f'actual: {actual} and pred: {pred}')


# +
import random
import itertools
sampled_edges = random.sample(g.edges, 100)
sampled_nodes = set(itertools.chain(*sampled_edges))
g_sampled = nx.DiGraph(g).subgraph(sampled_nodes)
node_list = list(g_sampled.nodes())
edge_list = list(g_sampled.edges())

A2 = nx.adjacency_matrix(g_sampled,weight="speed").todense()
from sklearn.preprocessing import MaxAbsScaler
transformer = MaxAbsScaler().fit(A2)
A2 = transformer.transform(A2)
gt_edges = random.sample(sampled_edges,10)
for u,v in gt_edges:
    g_sampled[u][v]["old_speed"] = A2[node_list.index(u), node_list.index(v)]
    A2[node_list.index(u), node_list.index(v)] = np.nan

# -

W,H = mf(A2, 5, iterations=3000)

# +
node_list = list(g_sampled.nodes())
edge_list = list(g_sampled.edges())
y_actual = list()
y_pred = list()
for u,v in gt_edges:
    actual = g_sampled[u][v]["old_speed"]
    pred = np.dot(W[node_list.index(u)],H[node_list.index(v)])
    print(f'actual: {actual} and pred: {pred}')
    y_actual.append(actual)
    y_pred.append(pred)
    
from sklearn.metrics import mean_squared_error
from math import sqrt

rmse = sqrt(mean_squared_error(y_actual, y_pred))
print(f'rmse: {rmse}')



    


# -

print(f'original density: {nx.density(g)} and sampled graph density: {nx.density(g_sampled)}')

sampled_nodes = random.sample(g.nodes, 5000)
g_sampled = nx.DiGraph(g).subgraph(sampled_nodes)
print(g_sampled.number_of_edges())
node_list = list(g_sampled.nodes())
edge_list = list(g_sampled.edges())

A3 = nx.adjacency_matrix(g_sampled,weight="speed").todense()
from sklearn.preprocessing import MaxAbsScaler
transformer = MaxAbsScaler().fit(A3)
A3 = transformer.transform(A3)
gt_edges = random.sample(g_sampled.edges(),10)
for u,v in gt_edges:
    g_sampled[u][v]["old_speed"] = A3[node_list.index(u), node_list.index(v)]
    A3[node_list.index(u), node_list.index(v)] = np.nan

W,H, tol_list = mf(A3, 5, iterations=100)

# +
node_list = list(g_sampled.nodes())
edge_list = list(g_sampled.edges())
y_actual = list()
y_pred = list()
for u,v in gt_edges:
    actual = g_sampled[u][v]["old_speed"]
    pred = np.dot(W[node_list.index(u)],H[node_list.index(v)])
    print(f'actual: {actual} and pred: {pred}')
    y_actual.append(actual)
    y_pred.append(pred)
    
from sklearn.metrics import mean_squared_error
from math import sqrt

rmse = sqrt(mean_squared_error(y_actual, y_pred))
print(f'rmse: {rmse}')
# -

print(f'original density: {nx.density(g)} and sampled graph density: {nx.density(g_sampled)}')


