# 2021313549 정성수
# Introduction to Deep Neural Networks - HW4
# Citation Prediction with Graph Convolutional Networks on Cora Dataset

import os
import random
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# 1. REPRODUCIBILITY
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# 2. LOAD CORA CONTENT AND CITATIONS
content_path = 'cora.content'
cites_path   = 'cora.cites'

# load node features and labels
content = pd.read_csv(content_path, sep='\t', header=None)
paper_ids = content[0].values
features  = content.iloc[:, 1:-1].values.astype(np.float32)
labels    = content.iloc[:, -1].values

# build a mapping from paper_id -> node index
id2idx = {pid: idx for idx, pid in enumerate(paper_ids)}

# load edges
edges = []
with open(cites_path) as f:
    for line in f:
        src, dst = line.strip().split()
        if src in id2idx and dst in id2idx:
            # edge direction: dst -> src, but we build undirected graph
            u = id2idx[src]
            v = id2idx[dst]
            edges.append((u, v))
            edges.append((v, u))

edges = list(set(edges))
edge_index = np.array(edges).T  # shape [2, E]

# 3. ADJACENCY AND NETWORKX GRAPH
N = features.shape[0]
adj = np.zeros((N, N), dtype=np.float32)
for u, v in edges:
    adj[u, v] = 1.0
# add self‐loops
adj[np.arange(N), np.arange(N)] = 1.0

G = nx.Graph()
G.add_nodes_from(range(N))
G.add_edges_from([(u, v) for u, v in edges if u < v])

# 4. TRAIN/VAL/TEST SPLIT ON EDGES
all_pos = np.array(edges)
# positive edges unique (undirected)
mask = all_pos[:,0] < all_pos[:,1]
pos_edges = all_pos[mask]
neg_edges = []

# create all non‐edges
while len(neg_edges) < len(pos_edges):
    u = np.random.randint(0, N)
    v = np.random.randint(0, N)
    if u != v and adj[u, v] == 0:
        neg_edges.append((u, v))
neg_edges = np.array(neg_edges)

# split
pos_train, pos_tmp, neg_train, neg_tmp = train_test_split(
    pos_edges, neg_edges, train_size=0.7, random_state=SEED
)
pos_val, pos_test, neg_val, neg_test = train_test_split(
    pos_tmp, neg_tmp, test_size=2/3, random_state=SEED
)
# note: val = 10%, test = 20%

# 5. PYTORCH DATASET HELPERS
class EdgeDataset(torch.utils.data.Dataset):
    def __init__(self, pos_edges, neg_edges, num_nodes, Q):
        self.pos = pos_edges
        self.neg = neg_edges
        self.Q = Q
        self.num_nodes = num_nodes
    def __len__(self):
        return len(self.pos)
    def __getitem__(self, idx):
        u, v_pos = self.pos[idx]
        # sample Q negative v's for u
        neg_vs = []
        while len(neg_vs) < self.Q:
            v = random.randrange(self.num_nodes)
            if adj[u, v] == 0:
                neg_vs.append(v)
        return u, v_pos, torch.tensor(neg_vs, dtype=torch.long)

# 6. GRAPHCONV LAYER
class GraphConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GraphConv, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
    def forward(self, x, adj):
        # x: [N, in_dim], adj: [N, N]
        h = torch.matmul(adj, x)
        return F.relu(self.linear(h))

# 7. GCN MODEL
class GCN(nn.Module):
    def __init__(self, in_dim, hid_dim):
        super(GCN, self).__init__()
        self.gc1 = GraphConv(in_dim, hid_dim)
        self.gc2 = GraphConv(hid_dim, hid_dim)
    def forward(self, x, adj):
        h1 = self.gc1(x, adj)
        h2 = self.gc2(h1, adj)
        return h2  # node embeddings

# 8. TRAINING SETUP
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
X = torch.tensor(features, device=device)          # [N, D]
A = torch.tensor(adj,    device=device)            # [N, N]

model = GCN(in_dim=X.size(1), hid_dim=16).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-2, weight_decay=4e-5)
Q = 2
batch_size = 256
epochs = 100
patience = 10

train_ds = EdgeDataset(pos_train, neg_train, N, Q)
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)

best_val_auc = 0.0
patience_cnt = 0

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for u, v_pos, v_negs in train_loader:
        u, v_pos, v_negs = u.to(device), v_pos.to(device), v_negs.to(device)
        z = model(X, A)  # [N, hid]
        z_u = z[u]       # [B, hid]
        z_vp = z[v_pos]  # [B, hid]
        pos_score = torch.sum(z_u * z_vp, dim=1)             # [B]
        neg_score = torch.bmm(z_u.unsqueeze(1),             # [B,1,hid]
                              z[v_negs].permute(0,2,1))    # [B,hid,Q]
        neg_score = neg_score.squeeze(1)                    # [B,Q]

        loss_pos = -F.logsigmoid(pos_score).mean()
        loss_neg = -F.logsigmoid(-neg_score).mean()
        loss = loss_pos + Q * loss_neg

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # validation AUC
    model.eval()
    with torch.no_grad():
        z = model(X, A).cpu().numpy()
        def edge_scores(edges):
            return np.sum(z[edges[:,0]] * z[edges[:,1]], axis=1)
        scores_pos = edge_scores(pos_val)
        scores_neg = edge_scores(neg_val)
        labels_auc = np.hstack([np.ones(len(scores_pos)), np.zeros(len(scores_neg))])
        scores_auc = np.hstack([scores_pos, scores_neg])
        val_auc = roc_auc_score(labels_auc, scores_auc)

    if val_auc > best_val_auc:
        best_val_auc = val_auc
        torch.save(model.state_dict(), 'best_gcn.pt')
        patience_cnt = 0
    else:
        patience_cnt += 1
        if patience_cnt >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    print(f"Epoch {epoch}  Loss {total_loss/len(train_loader):.4f}  Val AUC {val_auc:.4f}")

# load best model and test AUC
model.load_state_dict(torch.load('best_gcn.pt'))
model.eval()
with torch.no_grad():
    z = model(X, A).cpu().numpy()
    pos_scores = np.sum(z[pos_test[:,0]] * z[pos_test[:,1]], axis=1)
    neg_scores = np.sum(z[neg_test[:,0]] * z[neg_test[:,1]], axis=1)
    y_true = np.hstack([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    y_score = np.hstack([pos_scores, neg_scores])
    test_auc = roc_auc_score(y_true, y_score)
print(f"Test AUC: {test_auc:.4f}")

# 9. VISUALIZE PREDICTIONS ON THE ORIGINAL GRAPH
threshold = 0.5
pred_pos = (np.sum(z[pos_test[:,0]] * z[pos_test[:,1]], axis=1) > np.log(threshold/(1-threshold))).astype(int)

edge_colors = []
for idx, (u, v) in enumerate(pos_test):
    correct = pred_pos[idx] == 1
    edge_colors.append('blue' if correct else 'red')

plt.figure(figsize=(8,8))
pos = nx.spring_layout(G, seed=SEED)
nx.draw_networkx_nodes(G, pos, node_size=20)
nx.draw_networkx_edges(G, pos, edgelist=[tuple(e) for e in pos_test], edge_color=edge_colors, alpha=0.6)
plt.title("Test‐set Positive Edges: blue=correct, red=missed")
plt.axis('off')
plt.show()