#!/usr/bin/env python3
"""
Improved Similarity Methods for Better Connectivity Preservation
"""

import numpy as np
import scipy.sparse as sp
import networkx as nx
import json
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_scipy_sparse_matrix
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

class ImprovedSimilarityMethods:
    def __init__(self, output_dir="improved_similarity"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = []
    
    def load_cora_data(self):
        """Load Cora dataset."""
        with open("dataset/cora/cora-G.json", 'r') as f:
            data_json = json.load(f)
        
        edges = data_json['links']
        n_nodes = len(data_json['nodes'])
        
        edge_list = []
        for edge in edges:
            src, tgt = edge['source'], edge['target']
            edge_list.append((src, tgt))
            edge_list.append((tgt, src))
        
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        
        try:
            features = np.load("dataset/cora/cora-feats.npy")
            x = torch.tensor(features, dtype=torch.float)
        except FileNotFoundError:
            x = torch.eye(n_nodes, dtype=torch.float)
        
        data = Data(x=x, edge_index=edge_index, num_nodes=n_nodes)
        return data
    
    def analyze_connectivity(self, adjacency_matrix):
        """Analyze connectivity."""
        try:
            if hasattr(nx, 'from_scipy_sparse_array'):
                G = nx.from_scipy_sparse_array(adjacency_matrix)
            else:
                G = nx.from_scipy_sparse_matrix(adjacency_matrix)
            
            n_components = nx.number_connected_components(G)
            components_list = list(nx.connected_components(G))
            component_sizes = sorted([len(c) for c in components_list], reverse=True)
            
            largest_component = component_sizes[0] if component_sizes else 0
            largest_component_pct = (largest_component / adjacency_matrix.shape[0]) * 100
            
            return {
                'success': True,
                'n_components': n_components,
                'largest_component': largest_component,
                'largest_component_pct': largest_component_pct
            }
            
        except Exception as e:
            return {
                'success': False,
                'n_components': -1,
                'largest_component': 0,
                'largest_component_pct': 0
            }
    
    def method_1_structural_similarity(self, data, threshold=0.5):
        """Method 1: Use structural features instead of random embeddings."""
        
        print(f"🔧 Method 1: Structural Similarity (threshold={threshold})")
        
        # Build adjacency matrix
        A = to_scipy_sparse_matrix(data.edge_index, num_nodes=data.num_nodes).tocsr()
        
        # Use actual node features if available, otherwise structural features
        if hasattr(data, 'x') and data.x.shape[1] > 1:
            # Use node features
            embeddings = data.x.cpu().numpy()
            print(f"   Using node features: {embeddings.shape}")
        else:
            # Create structural embeddings: [degree, clustering_coeff, pagerank, ...]
            G = nx.from_scipy_sparse_matrix(A)
            
            # Compute structural features
            degrees = dict(G.degree())
            clustering = nx.clustering(G)
            pagerank = nx.pagerank(G, max_iter=50)
            
            # Build feature matrix
            embeddings = np.zeros((data.num_nodes, 3))
            for i in range(data.num_nodes):
                embeddings[i, 0] = degrees.get(i, 0)
                embeddings[i, 1] = clustering.get(i, 0)
                embeddings[i, 2] = pagerank.get(i, 0)
            
            print(f"   Using structural features: {embeddings.shape}")
        
        # Compute similarities
        sim = cosine_similarity(embeddings)
        
        # Apply to edges
        edge_index = data.edge_index.cpu().numpy()
        if edge_index.ndim == 1:
            edge_index = edge_index.reshape(2, -1)
        
        rows, cols = edge_index[0], edge_index[1]
        edge_similarities = sim[rows, cols]
        weights = np.where(edge_similarities > threshold, edge_similarities, 0.0)
        
        # Build reweighted adjacency
        A_reweighted = sp.coo_matrix((weights, (rows, cols)), shape=(data.num_nodes, data.num_nodes))
        A_reweighted = A_reweighted.maximum(A_reweighted.T).tocsr()
        A_reweighted.eliminate_zeros()
        
        return A_reweighted, edge_similarities
    
    def method_2_adaptive_threshold(self, data, percentile=50):
        """Method 2: Use adaptive threshold based on similarity distribution."""
        
        print(f"🔧 Method 2: Adaptive Threshold (percentile={percentile})")
        
        # Your original spectral filtering
        A = to_scipy_sparse_matrix(data.edge_index, num_nodes=data.num_nodes).tocsr()
        
        # Build normalized Laplacian
        d = np.array(A.sum(axis=1)).flatten()
        d_safe = d + 1e-12
        d_inv_sqrt = sp.diags(1.0 / np.sqrt(d_safe))
        L = sp.diags(d) - A
        L_norm = d_inv_sqrt @ L @ d_inv_sqrt
        
        # Apply spectral filter
        np.random.seed(42)
        X = np.random.randn(data.num_nodes, 20)
        
        I = sp.identity(L_norm.shape[0], format='csr')
        filter_matrix = I - 0.5 * L_norm.tocsr()
        Y = np.zeros_like(X)
        
        for j in range(X.shape[1]):
            x = X[:, j].copy()
            power_k = x.copy()
            for _ in range(10):  # k=10
                power_k = filter_matrix @ power_k
            power_k_plus_1 = filter_matrix @ power_k
            Y[:, j] = power_k_plus_1 - power_k
        
        # Compute similarities
        sim = cosine_similarity(Y)
        
        # Adaptive threshold: use percentile of edge similarities
        edge_index = data.edge_index.cpu().numpy()
        if edge_index.ndim == 1:
            edge_index = edge_index.reshape(2, -1)
        
        rows, cols = edge_index[0], edge_index[1]
        edge_similarities = sim[rows, cols]
        
        # Use percentile as threshold
        adaptive_threshold = np.percentile(edge_similarities, percentile)
        print(f"   Adaptive threshold: {adaptive_threshold:.4f} ({percentile}th percentile)")
        
        weights = np.where(edge_similarities > adaptive_threshold, edge_similarities, 0.0)
        
        # Build reweighted adjacency
        A_reweighted = sp.coo_matrix((weights, (rows, cols)), shape=(data.num_nodes, data.num_nodes))
        A_reweighted = A_reweighted.maximum(A_reweighted.T).tocsr()
        A_reweighted.eliminate_zeros()
        
        return A_reweighted, edge_similarities
    
    def method_3_connectivity_aware_filtering(self, data, target_components=100):
        """Method 3: Iteratively adjust threshold to achieve target connectivity."""
        
        print(f"🔧 Method 3: Connectivity-Aware Filtering (target={target_components} components)")
        
        # Your original spectral filtering setup
        A = to_scipy_sparse_matrix(data.edge_index, num_nodes=data.num_nodes).tocsr()
        
        d = np.array(A.sum(axis=1)).flatten()
        d_safe = d + 1e-12
        d_inv_sqrt = sp.diags(1.0 / np.sqrt(d_safe))
        L = sp.diags(d) - A
        L_norm = d_inv_sqrt @ L @ d_inv_sqrt
        
        np.random.seed(42)
        X = np.random.randn(data.num_nodes, 20)
        
        I = sp.identity(L_norm.shape[0], format='csr')
        filter_matrix = I - 0.5 * L_norm.tocsr()
        Y = np.zeros_like(X)
        
        for j in range(X.shape[1]):
            x = X[:, j].copy()
            power_k = x.copy()
            for _ in range(10):
                power_k = filter_matrix @ power_k
            power_k_plus_1 = filter_matrix @ power_k
            Y[:, j] = power_k_plus_1 - power_k
        
        sim = cosine_similarity(Y)
        
        edge_index = data.edge_index.cpu().numpy()
        if edge_index.ndim == 1:
            edge_index = edge_index.reshape(2, -1)
        
        rows, cols = edge_index[0], edge_index[1]
        edge_similarities = sim[rows, cols]
        
        # Binary search for optimal threshold
        low_threshold = edge_similarities.min()
        high_threshold = edge_similarities.max()
        best_threshold = low_threshold
        best_components = float('inf')
        
        for iteration in range(20):  # Max 20 binary search iterations
            mid_threshold = (low_threshold + high_threshold) / 2
            
            # Test this threshold
            weights = np.where(edge_similarities > mid_threshold, edge_similarities, 0.0)
            A_test = sp.coo_matrix((weights, (rows, cols)), shape=(data.num_nodes, data.num_nodes))
            A_test = A_test.maximum(A_test.T).tocsr()
            A_test.eliminate_zeros()
            
            connectivity = self.analyze_connectivity(A_test)
            n_components = connectivity['n_components']
            
            print(f"   Iteration {iteration+1}: threshold={mid_threshold:.4f}, components={n_components}")
            
            # Update best if closer to target
            if abs(n_components - target_components) < abs(best_components - target_components):
                best_threshold = mid_threshold
                best_components = n_components
            
            # Binary search logic
            if n_components > target_components:
                high_threshold = mid_threshold  # Too many components, lower threshold
            else:
                low_threshold = mid_threshold   # Too few components, raise threshold
            
            # Stop if close enough
            if abs(n_components - target_components) <= 5:
                break
        
        print(f"   Final threshold: {best_threshold:.4f}, components: {best_components}")
        
        # Build final reweighted matrix
        weights = np.where(edge_similarities > best_threshold, edge_similarities, 0.0)
        A_reweighted = sp.coo_matrix((weights, (rows, cols)), shape=(data.num_nodes, data.num_nodes))
        A_reweighted = A_reweighted.maximum(A_reweighted.T).tocsr()
        A_reweighted.eliminate_zeros()
        
        return A_reweighted, edge_similarities
    
    def method_4_clustering_based_coarsening(self, data, n_clusters=200):
        """Method 4: Skip similarity filtering, use clustering directly."""
        
        print(f"🔧 Method 4: Direct Clustering (n_clusters={n_clusters})")
        
        # Use node features for clustering
        if hasattr(data, 'x') and data.x.shape[1] > 1:
            features = data.x.cpu().numpy()
        else:
            # Use positional encoding
            A = to_scipy_sparse_matrix(data.edge_index, num_nodes=data.num_nodes).tocsr()
            d = np.array(A.sum(axis=1)).flatten()
            d_safe = d + 1e-12
            d_inv_sqrt = sp.diags(1.0 / np.sqrt(d_safe))
            L = sp.diags(d) - A
            L_norm = d_inv_sqrt @ L @ d_inv_sqrt
            
            # Use eigenvectors as features
            from scipy.sparse.linalg import eigsh
            try:
                eigenvals, eigenvecs = eigsh(L_norm, k=20, which='SM')
                features = eigenvecs
            except:
                features = np.random.randn(data.num_nodes, 20)
        
        # K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(features)
        
        # Build coarsened adjacency directly (no similarity filtering)
        A_orig = to_scipy_sparse_matrix(data.edge_index, num_nodes=data.num_nodes).tocsr()
        
        # Build assignment matrix
        P = sp.lil_matrix((data.num_nodes, n_clusters))
        for i, cluster_id in enumerate(clusters):
            P[i, cluster_id] = 1.0
        P = P.tocsr()
        
        # Coarsen: A_coarse = P^T * A * P
        A_coarsened = P.T @ A_orig @ P
        A_coarsened.eliminate_zeros()
        
        return A_coarsened, None  # No edge similarities for this method
    
    def test_all_methods(self):
        """Test all improved similarity methods."""
        
        print("🚀 TESTING IMPROVED SIMILARITY METHODS")
        print("="*60)
        
        data = self.load_cora_data()
        A_orig = to_scipy_sparse_matrix(data.edge_index, num_nodes=data.num_nodes).tocsr()
        orig_connectivity = self.analyze_connectivity(A_orig)
        
        print(f"📊 Original graph: {orig_connectivity['n_components']} components, {orig_connectivity['largest_component_pct']:.1f}% largest")
        print()
        
        methods = [
            ("Structural Features", lambda: self.method_1_structural_similarity(data, threshold=0.5)),
            ("Adaptive Threshold (50%)", lambda: self.method_2_adaptive_threshold(data, percentile=50)),
            ("Adaptive Threshold (75%)", lambda: self.method_2_adaptive_threshold(data, percentile=75)),
            ("Connectivity-Aware (100)", lambda: self.method_3_connectivity_aware_filtering(data, target_components=100)),
            ("Connectivity-Aware (150)", lambda: self.method_3_connectivity_aware_filtering(data, target_components=150)),
            ("Direct K-means (200)", lambda: self.method_4_clustering_based_coarsening(data, n_clusters=200)),
            ("Direct K-means (150)", lambda: self.method_4_clustering_based_coarsening(data, n_clusters=150)),
        ]
        
        results = []
        
        for method_name, method_func in methods:
            print(f"\n{'='*20} {method_name} {'='*20}")
            
            try:
                A_processed, edge_similarities = method_func()
                connectivity = self.analyze_connectivity(A_processed)
                
                edge_survival = A_processed.nnz / A_orig.nnz * 100
                connectivity_degradation = connectivity['n_components'] - orig_connectivity['n_components']
                
                result = {
                    'method': method_name,
                    'success': True,
                    'original_components': orig_connectivity['n_components'],
                    'processed_components': connectivity['n_components'],
                    'connectivity_degradation': connectivity_degradation,
                    'processed_nodes': A_processed.shape[0],
                    'edge_survival_rate': edge_survival,
                    'largest_component_pct': connectivity['largest_component_pct']
                }
                
                print(f"✅ Result: {connectivity['n_components']} components, {edge_survival:.1f}% edges, {connectivity['largest_component_pct']:.1f}% largest")
                
            except Exception as e:
                print(f"❌ Failed: {e}")
                result = {
                    'method': method_name,
                    'success': False,
                    'error': str(e)
                }
            
            results.append(result)
        
        # Analyze results
        self.analyze_method_comparison(results)
        
        return results
    
    def analyze_method_comparison(self, results):
        """Compare all methods."""
        
        df = pd.DataFrame(results)
        success_df = df[df['success'] == True]
        
        if len(success_df) == 0:
            print("❌ No successful methods!")
            return
        
        print(f"\n🏆 METHOD COMPARISON RESULTS")
        print("="*80)
        print("Method                    | Components | Edge% | Largest% | Degradation")
        print("-"*80)
        
        for _, row in success_df.iterrows():
            method_name = row['method'][:25].ljust(25)
            if 'processed_components' in row:
                print(f"{method_name}| {row['processed_components']:9d} | {row['edge_survival_rate']:4.1f}% | {row['largest_component_pct']:7.1f}% | {row['connectivity_degradation']:+4d}")
        
        # Find best methods
        print(f"\n🥇 BEST METHODS:")
        print("-"*40)
        
        # Best connectivity preservation
        best_connectivity = success_df.loc[success_df['connectivity_degradation'].idxmin()]
        print(f"Best connectivity: {best_connectivity['method']}")
        print(f"   Components: {best_connectivity['original_components']} → {best_connectivity['processed_components']} ({best_connectivity['connectivity_degradation']:+d})")
        
        # Best balance
        success_df['balance_score'] = (
            100 / (success_df['connectivity_degradation'].abs() + 1) *
            success_df['edge_survival_rate'] / 100 *
            success_df['largest_component_pct'] / 100
        )
        
        best_balance = success_df.loc[success_df['balance_score'].idxmax()]
        print(f"\nBest balance: {best_balance['method']}")
        print(f"   Components: {best_balance['original_components']} → {best_balance['processed_components']} ({best_balance['connectivity_degradation']:+d})")
        print(f"   Balance score: {best_balance['balance_score']:.3f}")
        
        # Save results
        csv_file = self.output_dir / "method_comparison.csv"
        success_df.to_csv(csv_file, index=False)
        print(f"\n💾 Results saved: {csv_file}")
        
        return success_df


def main():
    """Test improved similarity methods."""
    
    analyzer = ImprovedSimilarityMethods()
    results = analyzer.test_all_methods()
    
    print(f"\n🎯 SUMMARY:")
    print("-"*50)
    print("Tested multiple approaches to improve connectivity preservation:")
    print("1. Structural features instead of random embeddings")
    print("2. Adaptive thresholds based on similarity distribution")
    print("3. Connectivity-aware threshold optimization")
    print("4. Direct clustering without similarity filtering")
    print("\nUse the best method for your CMG++ pipeline!")

if __name__ == "__main__":
    main()
