# 2021313549 정성수
# Introduction to Deep Neural Networks - HW4
# Citation Prediction with Graph Convolutional Networks on Cora Dataset

import os
import random
from typing import Tuple, List, Dict

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import networkx as nx

# -----------------------------
# 1. SEED
# -----------------------------
# Set fixed random seed to ensure reproducible results
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

# -----------------------------
# 2. DATA LOADING AND PREPROCESSING
# -----------------------------

def load_cora_dataset(dataset_dir: str = "./cora") -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[int, int]]]:
    """
    Load Cora dataset and return feature matrix, normalized adjacency matrix, and edge list.
    
    The Cora dataset contains 2708 scientific publications (nodes) classified into 7 classes.
    Each publication is described by a 0/1-valued word vector with 1433 dimensions.
    The citation network consists of 5429 links (edges).
    
    Returns:
        features: Tensor[num_nodes, 1433] - Binary word vectors for each paper
        adj_norm: Tensor[num_nodes, num_nodes] - Normalized adjacency matrix with self-loops
        edges: List of (src_idx, dst_idx) tuples representing citations
    """
    content_file = os.path.join(dataset_dir, "cora.content")
    cites_file = os.path.join(dataset_dir, "cora.cites")

    # cora.content: <paper_id> <1433-dim word vector> <class_label>
    paper_ids = []
    features_list: List[List[int]] = []
    with open(content_file, "r") as f:
        for line in f:
            comps = line.strip().split()
            paper_ids.append(comps[0])
            features_list.append([int(x) for x in comps[1:-1]])  # 0/1 bag-of-words

    # Convert features to tensor [num_nodes, 1433]
    features = torch.tensor(features_list, dtype=torch.float32)

    # MAPPING: paper_id -> index
    id2idx = {pid: idx for idx, pid in enumerate(paper_ids)}

    edges: List[Tuple[int, int]] = []
    with open(cites_file, "r") as f:
        for line in f:
            src, dst = line.strip().split()
            if src not in id2idx or dst not in id2idx:
                continue  # RARELY OCCURRING NODES ARE REMOVED
            src_idx = id2idx[src]
            dst_idx = id2idx[dst]
            # CONVERT TO UNDIRECTED GRAPH
            edges.append((src_idx, dst_idx))
            edges.append((dst_idx, src_idx))  # Add reverse edge to make it undirected

    # Create adjacency matrix
    num_nodes = features.shape[0]
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    for u, v in edges:
        adj[u, v] = 1.0
    # ADD SELF-LOOPS to prevent information loss during message passing
    adj += torch.eye(num_nodes, dtype=torch.float32)

    # D-1/2 * A * D-1/2 normalization (symmetric normalization)
    # This prevents numerical instabilities and exploding/vanishing gradients
    deg = torch.sum(adj, dim=1)
    deg_inv_sqrt = torch.pow(deg, -0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
    D_inv_sqrt = torch.diag(deg_inv_sqrt)
    adj_norm = torch.mm(torch.mm(D_inv_sqrt, adj), D_inv_sqrt)

    return features, adj_norm, edges

# -----------------------------
# 3. GRAPH VISUALIZATION FUNCTION (OPTIONAL)
# -----------------------------

def visualize_predictions(G: nx.Graph, edge_preds: Dict[Tuple[int, int], float], threshold: float = 0.5, title: str = "Prediction Results"):
    """
    Visualize the prediction results by highlighting correctly and incorrectly predicted edges.
    
    Args:
        G: NetworkX graph containing the original citation network
        edge_preds: Dictionary mapping edges to their prediction scores
        threshold: Score threshold above which an edge is predicted as positive
        title: Plot title
    """
    plt.figure(figsize=(8, 8))
    pos = nx.spring_layout(G, seed=SEED)

    true_edges = list(G.edges())
    pred_pos_edges = [e for e, p in edge_preds.items() if p >= threshold]

    # CORRECT / INCORRECT PREDICTIONS
    # Blue edges: correctly predicted citations
    # Red edges: incorrectly predicted citations
    correct_edges = [e for e in pred_pos_edges if e in true_edges or (e[1], e[0]) in true_edges]
    incorrect_edges = [e for e in pred_pos_edges if e not in correct_edges]

    nx.draw_networkx_nodes(G, pos, node_size=50, node_color='lightgray')
    nx.draw_networkx_edges(G, pos, edgelist=true_edges, width=0.5, alpha=0.3)
    nx.draw_networkx_edges(G, pos, edgelist=correct_edges, edge_color='blue', width=1.5)
    nx.draw_networkx_edges(G, pos, edgelist=incorrect_edges, edge_color='red', width=1.5, style='dashed')

    plt.title(title)
    plt.axis('off')
    plt.show()

# -----------------------------
# 4. GRAPH CONVOLUTION LAYER IMPLEMENTATION
# -----------------------------

class GraphConv(nn.Module):
    """
    Graph Convolutional Layer as described in Kipf & Welling (2017) paper.
    
    Performs the operation: H = σ(D^(-1/2) A D^(-1/2) X W)
    where A is the adjacency matrix with self-loops, D is the degree matrix,
    X is the input feature matrix, and W is the weight matrix.
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        # Initialize weights using Xavier initialization
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        # Linear transformation
        support = torch.mm(x, self.weight)      # X W
        # Propagate messages through graph structure
        out = torch.mm(adj_norm, support)       # Ĥ X W
        return F.relu(out)  # Apply non-linear activation function

# -----------------------------
# 5. GCN MODEL DEFINITION
# -----------------------------

class GCNLinkPredictor(nn.Module):
    """
    GCN-based model for link prediction. 
    
    Architecture:
    1. Two graph convolutional layers to learn node embeddings
    2. For each potential edge (u,v), concatenate node embeddings
    3. Apply a linear classifier to predict the likelihood of a link
    """
    def __init__(self, num_features: int, hidden_dim: int = 16, dropout: float = 0.1):
        super().__init__()
        self.gc1 = GraphConv(num_features, hidden_dim)
        self.gc2 = GraphConv(hidden_dim, hidden_dim)
        self.dropout = dropout
        # CONCATENATE TWO EMBEDDINGS → 2*hidden_dim -> 1
        self.classifier = nn.Linear(2 * hidden_dim, 1)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor, edge_pairs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for link prediction.
        
        Args:
            x: Node feature matrix [num_nodes, num_features]
            adj_norm: Normalized adjacency matrix [num_nodes, num_nodes]
            edge_pairs: Edge pairs to evaluate [num_edges, 2]
            
        Returns:
            logits: Link prediction scores [num_edges]
        """
        # First graph convolutional layer
        h = self.gc1(x, adj_norm)
        h = F.dropout(h, p=self.dropout, training=self.training)
        # Second graph convolutional layer
        h = self.gc2(h, adj_norm)

        # Extract node embeddings for each edge pair
        # edge_pairs: [E, 2] (u, v) indices
        h_u = h[edge_pairs[:, 0]]  # [E, hidden]
        h_v = h[edge_pairs[:, 1]]  # [E, hidden]
        # Concatenate node embeddings of both endpoints
        edge_feats = torch.cat([h_u, h_v], dim=1)  # [E, 2*hidden]
        # Final prediction
        logits = self.classifier(edge_feats).squeeze()  # [E]
        return logits

# -----------------------------
# 6. DATA SPLITTING (train/val/test) AND NEGATIVE SAMPLING
# -----------------------------

def train_val_test_split(edges: List[Tuple[int, int]], test_ratio=0.2, val_ratio=0.1, seed: int = SEED):
    """
    Split the edges into training, validation, and test sets.
    
    Args:
        edges: List of edge tuples (src_idx, dst_idx)
        test_ratio: Proportion of edges for testing
        val_ratio: Proportion of edges for validation
        seed: Random seed for reproducibility
    
    Returns:
        train_edges, val_edges, test_edges: The split edge sets
    """
    # CONVERT TO UNDIRECTED EDGES (deduplicate and ensure consistent ordering)
    undirected_edges = list({tuple(sorted(e)) for e in edges})
    random.Random(seed).shuffle(undirected_edges)
    num_total = len(undirected_edges)
    num_test = int(num_total * test_ratio)
    num_val = int(num_total * val_ratio)
    test_edges = undirected_edges[:num_test]
    val_edges = undirected_edges[num_test:num_test + num_val]
    train_edges = undirected_edges[num_test + num_val:]
    return train_edges, val_edges, test_edges


def sample_negative_edges(num_nodes: int, num_samples: int, existing_edges_set: set) -> List[Tuple[int, int]]:
    """
    Sample negative (non-existent) edges uniformly at random.
    
    Args:
        num_nodes: Number of nodes in the graph
        num_samples: Number of negative samples to generate
        existing_edges_set: Set of existing edges to avoid
        
    Returns:
        List of negative edge tuples
    """
    negatives = []
    while len(negatives) < num_samples:
        u = random.randint(0, num_nodes - 1)
        v = random.randint(0, num_nodes - 1)
        if u == v:
            continue  # Skip self-loops
        edge = (min(u, v), max(u, v))  # Canonical edge representation
        if edge not in existing_edges_set:
            negatives.append(edge)
            existing_edges_set.add(edge)  # Prevent duplicate sampling
    return negatives

# -----------------------------
# 7. TRAINING LOOP
# -----------------------------

def edge_list_to_tensor(edge_list: List[Tuple[int, int]]) -> torch.Tensor:
    """Convert list of edge tuples to torch tensor."""
    return torch.tensor(edge_list, dtype=torch.long)


def train_model(model: nn.Module, features: torch.Tensor, adj_norm: torch.Tensor,
                train_edges: List[Tuple[int, int]], val_edges: List[Tuple[int, int]],
                num_nodes: int, epochs: int = 100, batch_size: int = 256, lr: float = 0.01,
                weight_decay: float = 4e-5, q_neg: int = 2, patience: int = 10):
    """
    Train the GCN link prediction model.
    
    Args:
        model: The GCN model
        features: Node features matrix
        adj_norm: Normalized adjacency matrix
        train_edges: Training edges
        val_edges: Validation edges
        num_nodes: Number of nodes in the graph
        epochs: Maximum number of training epochs
        batch_size: Training batch size
        lr: Learning rate
        weight_decay: L2 regularization
        q_neg: Negative sampling ratio (# negative samples per positive)
        patience: Early stopping patience
        
    Returns:
        Trained model and best validation AUC
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val_auc = 0.0
    patience_counter = 0
    existing_edge_set = set(train_edges + val_edges)

    for epoch in range(1, epochs + 1):
        model.train()
        random.shuffle(train_edges)
        epoch_losses = []
        
        # Mini-batch training
        for i in range(0, len(train_edges), batch_size):
            pos_batch = train_edges[i:i + batch_size]
            # Sample negative edges for this batch
            neg_batch = sample_negative_edges(num_nodes, len(pos_batch) * q_neg, existing_edge_set.copy())

            # Combine positive and negative edges
            edge_pairs = edge_list_to_tensor(pos_batch + neg_batch)
            # Create labels: 1 for positive edges, 0 for negative
            labels = torch.tensor([1] * len(pos_batch) + [0] * len(neg_batch), dtype=torch.float32)

            # Forward and backward pass
            optimizer.zero_grad()
            logits = model(features, adj_norm, edge_pairs)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        # Validation
        val_auc = evaluate_auc(model, features, adj_norm, val_edges, num_nodes, q_neg=q_neg)
        print(f"Epoch {epoch:03d} | Loss {sum(epoch_losses)/len(epoch_losses):.4f} | Val AUC {val_auc:.4f}")

        # Early stopping
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping!")
            break

    # Load best model state
    model.load_state_dict(best_state)
    return model, best_val_auc


# -----------------------------
# 8. EVALUATION FUNCTION (AUC)
# -----------------------------

def evaluate_auc(model: nn.Module, features: torch.Tensor, adj_norm: torch.Tensor,
                 pos_edges: List[Tuple[int, int]], num_nodes: int, q_neg: int = 2) -> float:
    """
    Evaluate model performance using Area Under ROC Curve (AUC).
    
    Args:
        model: The GCN model
        features: Node features matrix
        adj_norm: Normalized adjacency matrix
        pos_edges: Positive edges (existing links)
        num_nodes: Number of nodes
        q_neg: Negative sampling ratio
        
    Returns:
        AUC score
    """
    model.eval()
    # Sample negative edges for evaluation
    neg_edges = sample_negative_edges(num_nodes, len(pos_edges) * q_neg, set(pos_edges))
    edge_pairs = edge_list_to_tensor(pos_edges + neg_edges)
    labels = torch.tensor([1] * len(pos_edges) + [0] * len(neg_edges), dtype=torch.float32)
    
    with torch.no_grad():
        logits = model(features, adj_norm, edge_pairs)
        probs = torch.sigmoid(logits).cpu().numpy()
    
    return roc_auc_score(labels.numpy(), probs)

# -----------------------------
# 9. MAIN EXECUTION
# -----------------------------

def main():
    # DATA LOADING
    features, adj_norm, edges = load_cora_dataset()
    num_nodes, num_features = features.shape
    print(f"Dataset loaded: {num_nodes} nodes, {num_features} features, {len(set(tuple(sorted(e)) for e in edges))//2} unique edges")

    # GRAPH OBJECT (FOR VISUALIZATION)
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from({tuple(sorted(e)) for e in edges})

    # DATA SPLITTING
    train_edges, val_edges, test_edges = train_val_test_split(edges)
    print(f"Edges split: {len(train_edges)} train, {len(val_edges)} validation, {len(test_edges)} test")

    # MODEL INITIALIZATION
    model = GCNLinkPredictor(num_features=num_features, hidden_dim=16, dropout=0.1)
    print(f"Model: {model.__class__.__name__} with hidden dimension {16}")

    # TRAINING
    model, best_val_auc = train_model(model, features, adj_norm, train_edges, val_edges, num_nodes,
                                      epochs=10, batch_size=256, lr=0.01, weight_decay=4e-5, q_neg=2, patience=10)

    # TEST PERFORMANCE
    test_auc = evaluate_auc(model, features, adj_norm, test_edges, num_nodes)
    print(f"Best Val AUC: {best_val_auc:.4f} | Test AUC: {test_auc:.4f}")

    # VISUALIZATION (OPTIONAL)
    edge_scores = {}
    with torch.no_grad():
        all_edges = list(G.edges())
        edge_pairs_tensor = edge_list_to_tensor(all_edges)
        preds = torch.sigmoid(model(features, adj_norm, edge_pairs_tensor)).cpu().numpy()
    for e, p in zip(all_edges, preds):
        edge_scores[e] = p
    visualize_predictions(G, edge_scores, threshold=0.5,
                          title=f"GCN Link Prediction (Test AUC={test_auc:.3f})")


if __name__ == "__main__":
    main()