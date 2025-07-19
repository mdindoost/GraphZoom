#!/usr/bin/env python3
"""
Fixed Matrix Extractor for LAMG Output Format
Handles the specific MTX format from your LAMG implementation
"""

import numpy as np
import scipy.sparse as sp
import json
import pickle
from pathlib import Path

class FixedMatrixExtractor:
    def __init__(self, output_dir="extracted_matrices"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def load_cora_original(self, dataset_path="dataset/cora"):
        """Load original Cora dataset."""
        print("Loading original Cora dataset...")
        
        # Load cora-G.json
        with open(f"{dataset_path}/cora-G.json", 'r') as f:
            data = json.load(f)
        
        edges = data['links']
        nodes = data['nodes']
        n_nodes = len(nodes)
        
        # Build adjacency matrix
        A = sp.lil_matrix((n_nodes, n_nodes))
        for edge in edges:
            i, j = edge['source'], edge['target']
            A[i, j] = 1
            A[j, i] = 1
        
        A = A.tocsr()
        
        # Build Laplacian
        degrees = np.array(A.sum(axis=1)).flatten()
        L = sp.diags(degrees) - A
        
        print(f"Original Cora: {n_nodes} nodes, {A.nnz//2} edges")
        
        return A, L, n_nodes
    
    def load_lamg_mtx_custom(self, mtx_file):
        """Load MTX file in your custom format."""
        print(f"Loading custom MTX file: {mtx_file}")
        
        with open(mtx_file, 'r') as f:
            lines = f.readlines()
        
        # First line has dimensions
        first_line = lines[0].strip().split()
        n_rows, n_cols = int(first_line[0]), int(first_line[1])
        
        # Read coordinate data
        rows, cols, vals = [], [], []
        
        for line in lines[1:]:
            parts = line.strip().split()
            if len(parts) >= 3:
                row = int(parts[0]) - 1  # Convert to 0-indexed
                col = int(parts[1]) - 1  # Convert to 0-indexed
                val = float(parts[2])
                rows.append(row)
                cols.append(col)
                vals.append(val)
        
        # Create sparse matrix
        matrix = sp.coo_matrix((vals, (rows, cols)), shape=(n_rows, n_cols))
        
        print(f"   Shape: {matrix.shape}, NNZ: {matrix.nnz}")
        
        return matrix.tocsr()
    
    def extract_lamg_matrices_fixed(self, lamg_results_dir="reduction_results"):
        """Extract LAMG matrices with fixed format handling."""
        print("Extracting LAMG matrices (fixed format)...")
        
        lamg_path = Path(lamg_results_dir)
        if not lamg_path.exists():
            print(f"Warning: LAMG results directory {lamg_path} not found")
            return {}
        
        lamg_matrices = {}
        
        # Look for specific files
        gs_file = lamg_path / "Gs.mtx"
        proj1_file = lamg_path / "Projection_1.mtx"
        proj2_file = lamg_path / "Projection_2.mtx"
        
        # Load Gs.mtx (coarsened graph Laplacian)
        if gs_file.exists():
            try:
                print(f"\nProcessing Gs.mtx (coarsened Laplacian)...")
                L_coarse = self.load_lamg_mtx_custom(gs_file)
                
                # Convert Laplacian to adjacency: A = D - L
                # But first check if diagonal is positive (proper Laplacian)
                diagonal = L_coarse.diagonal()
                print(f"   Diagonal range: {diagonal.min():.6f} to {diagonal.max():.6f}")
                
                if np.all(diagonal >= 0):
                    # Proper Laplacian: A = D - L
                    D = sp.diags(diagonal)
                    A_coarse = D - L_coarse
                    
                    # Clean up adjacency matrix
                    A_coarse.data = np.maximum(A_coarse.data, 0)  # Remove negative values
                    A_coarse = (A_coarse + A_coarse.T) / 2  # Ensure symmetry
                    A_coarse.eliminate_zeros()
                    
                    # Rebuild clean Laplacian
                    degrees_clean = np.array(A_coarse.sum(axis=1)).flatten()
                    L_clean = sp.diags(degrees_clean) - A_coarse
                    
                else:
                    # Matrix might already be adjacency or have different format
                    print("   Warning: Unusual diagonal values, treating as adjacency")
                    A_coarse = L_coarse.copy()
                    A_coarse.data = np.abs(A_coarse.data)  # Ensure positive
                    degrees = np.array(A_coarse.sum(axis=1)).flatten()
                    L_clean = sp.diags(degrees) - A_coarse
                
                lamg_matrices['coarsened'] = {
                    'adjacency': A_coarse,
                    'laplacian': L_clean,
                    'nodes': A_coarse.shape[0],
                    'edges': A_coarse.nnz // 2,
                    'source_file': str(gs_file)
                }
                
                # Save matrices
                self.save_matrix(A_coarse, "lamg_coarsened_adjacency")
                self.save_matrix(L_clean, "lamg_coarsened_laplacian")
                
                print(f"   ✅ Extracted: {A_coarse.shape[0]} nodes, {A_coarse.nnz//2} edges")
                
            except Exception as e:
                print(f"   ❌ Error processing Gs.mtx: {e}")
        
        # Load Projection matrices (for completeness)
        for proj_file, name in [(proj1_file, "projection_1"), (proj2_file, "projection_2")]:
            if proj_file.exists():
                try:
                    print(f"\nProcessing {proj_file.name} (projection matrix)...")
                    P = self.load_lamg_mtx_custom(proj_file)
                    
                    lamg_matrices[name] = {
                        'matrix': P,
                        'shape': P.shape,
                        'source_file': str(proj_file)
                    }
                    
                    self.save_matrix(P, f"lamg_{name}")
                    print(f"   ✅ Extracted: {P.shape[0]}×{P.shape[1]} projection matrix")
                    
                except Exception as e:
                    print(f"   ❌ Error processing {proj_file.name}: {e}")
        
        return lamg_matrices
    
    def extract_cmg_matrices(self, dataset_path="dataset/cora"):
        """Extract CMG matrices using the working integration."""
        print("Extracting CMG matrices...")
        
        try:
            from filtered import cmg_filtered_clustering
            import torch
            from torch_geometric.data import Data
            
            # Load dataset as PyG data
            with open(f"{dataset_path}/cora-G.json", 'r') as f:
                data_json = json.load(f)
            
            edges = data_json['links']
            n_nodes = len(data_json['nodes'])
            
            # Build edge_index
            edge_list = []
            for edge in edges:
                src, tgt = edge['source'], edge['target']
                edge_list.append((src, tgt))
                edge_list.append((tgt, src))
            
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            
            # Load features
            try:
                features = np.load(f"{dataset_path}/cora-feats.npy")
                x = torch.tensor(features, dtype=torch.float)
            except FileNotFoundError:
                x = torch.eye(n_nodes, dtype=torch.float)
            
            data = Data(x=x, edge_index=edge_index, num_nodes=n_nodes)
            
            # Run CMG
            clusters, n_clusters, phi_stats, lambda_crit = cmg_filtered_clustering(
                data, k=10, d=10, threshold=0.1
            )
            
            # Build original adjacency
            A_orig = sp.lil_matrix((n_nodes, n_nodes))
            for edge in edges:
                i, j = edge['source'], edge['target']
                A_orig[i, j] = 1
                A_orig[j, i] = 1
            A_orig = A_orig.tocsr()
            
            # Build assignment matrix
            P = sp.lil_matrix((n_nodes, n_clusters))
            for i, cluster_id in enumerate(clusters):
                P[i, cluster_id] = 1.0
            P = P.tocsr()
            
            # Coarsen: A_coarse = P^T * A_orig * P
            A_coarse = P.T @ A_orig @ P
            A_coarse.eliminate_zeros()
            
            # Build Laplacian
            degrees = np.array(A_coarse.sum(axis=1)).flatten()
            L_coarse = sp.diags(degrees) - A_coarse
            
            cmg_result = {
                'adjacency': A_coarse,
                'laplacian': L_coarse,
                'nodes': n_clusters,
                'edges': A_coarse.nnz // 2,
                'lambda_critical': lambda_crit,
                'phi_stats': phi_stats
            }
            
            # Save matrices
            self.save_matrix(A_coarse, "cmg_coarsened_adjacency")
            self.save_matrix(L_coarse, "cmg_coarsened_laplacian")
            self.save_matrix(P, "cmg_assignment")
            
            print(f"   ✅ CMG: {n_clusters} nodes, {A_coarse.nnz//2} edges")
            print(f"   λ_critical = {lambda_crit:.6f}")
            
            return cmg_result
            
        except Exception as e:
            print(f"   ❌ CMG extraction error: {e}")
            return {}
    
    def save_matrix(self, matrix, name):
        """Save matrix with metadata."""
        base_path = self.output_dir / name
        
        # Save as pickle
        with open(f"{base_path}.pkl", 'wb') as f:
            pickle.dump(matrix, f)
        
        # Save info
        info = {
            'shape': matrix.shape,
            'nnz': matrix.nnz,
            'format': str(type(matrix)),
            'dtype': str(matrix.dtype)
        }
        with open(f"{base_path}_info.json", 'w') as f:
            json.dump(info, f, indent=2)
    
    def run_extraction(self):
        """Run complete extraction with fixed format handling."""
        print("=== Fixed Matrix Extraction ===\n")
        
        # 1. Original
        print("1. Extracting original Cora matrices...")
        A_orig, L_orig, n_orig = self.load_cora_original()
        self.save_matrix(A_orig, "cora_original_adjacency")
        self.save_matrix(L_orig, "cora_original_laplacian")
        
        # 2. LAMG (fixed format)
        print("\n2. Extracting LAMG matrices...")
        lamg_results = self.extract_lamg_matrices_fixed()
        
        # 3. CMG
        print("\n3. Extracting CMG matrices...")
        cmg_results = self.extract_cmg_matrices()
        
        # Summary
        print(f"\n=== Extraction Complete ===")
        print(f"Original: {n_orig} nodes")
        if lamg_results:
            if 'coarsened' in lamg_results:
                print(f"LAMG: {lamg_results['coarsened']['nodes']} nodes")
        if cmg_results:
            print(f"CMG: {cmg_results['nodes']} nodes")
        
        print(f"Matrices saved to: {self.output_dir}")
        
        return {
            'original': {'nodes': n_orig},
            'lamg': lamg_results,
            'cmg': cmg_results
        }

if __name__ == "__main__":
    extractor = FixedMatrixExtractor()
    results = extractor.run_extraction()
