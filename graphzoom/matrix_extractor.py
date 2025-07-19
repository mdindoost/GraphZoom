#!/usr/bin/env python3
"""
Matrix Extraction Script for LAMG vs CMG++ Spectral Analysis
Extracts coarsened graph matrices from both methods for comparison.
"""

import numpy as np
import scipy.sparse as sp
import scipy.io as sio
import networkx as nx
import json
import os
from pathlib import Path
import pickle

class MatrixExtractor:
    def __init__(self, output_dir="extracted_matrices"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def load_cora_original(self, dataset_path="dataset/cora"):
        """Load original Cora dataset for reference."""
        print("Loading original Cora dataset...")
        
        # Try different possible file names
        json_files = ["cora-G.json", "cora.json", "cora-graph.json"]
        data = None
        
        for json_file in json_files:
            try:
                with open(f"{dataset_path}/{json_file}", 'r') as f:
                    data = json.load(f)
                print(f"✅ Found dataset file: {json_file}")
                break
            except FileNotFoundError:
                continue
        
        if data is None:
            raise FileNotFoundError(f"Could not find Cora JSON file in {dataset_path}. Tried: {json_files}")
        
        # Extract edges and build adjacency matrix
        edges = data['links']
        nodes = data['nodes']
        n_nodes = len(nodes)
        
        # Build adjacency matrix
        A = sp.lil_matrix((n_nodes, n_nodes))
        for edge in edges:
            i, j = edge['source'], edge['target']
            A[i, j] = 1
            A[j, i] = 1  # Symmetric
        
        A = A.tocsr()
        
        # Build Laplacian
        degrees = np.array(A.sum(axis=1)).flatten()
        L = sp.diags(degrees) - A
        
        print(f"Original Cora: {n_nodes} nodes, {A.nnz//2} edges")
        
        # Save matrices
        self.save_matrix(A, "cora_original_adjacency")
        self.save_matrix(L, "cora_original_laplacian")
        
        return A, L, n_nodes
    
    def extract_lamg_matrices(self, lamg_results_dir="reduction_results"):
        """Extract matrices from LAMG .mtx files."""
        print("Extracting LAMG matrices...")
        
        lamg_path = Path(lamg_results_dir)
        if not lamg_path.exists():
            print(f"Warning: LAMG results directory {lamg_path} not found")
            return {}
        
        lamg_matrices = {}
        
        # Look for Gs.mtx files (coarsened graphs)
        mtx_files = list(lamg_path.glob("**/*.mtx"))
        print(f"Found {len(mtx_files)} MTX files")
        
        for mtx_file in mtx_files:
            print(f"Processing: {mtx_file}")
            try:
                # Load MTX file (should be a Laplacian matrix)
                L = sio.mmread(str(mtx_file)).tocsr()
                
                # LAMG outputs Laplacian matrices with negative off-diagonals
                # Convert to adjacency matrix: A = D - L (where D is diagonal of L)
                D = sp.diags(L.diagonal())
                A = D - L
                
                # Clean up - ensure non-negative and symmetric
                A.data = np.maximum(A.data, 0)
                A = (A + A.T) / 2
                A.eliminate_zeros()
                
                # Verify and rebuild clean Laplacian
                degrees = np.array(A.sum(axis=1)).flatten()
                L_clean = sp.diags(degrees) - A
                
                matrix_name = mtx_file.stem
                lamg_matrices[matrix_name] = {
                    'adjacency': A,
                    'laplacian': L_clean,
                    'nodes': A.shape[0],
                    'edges': A.nnz // 2,
                    'source_file': str(mtx_file)
                }
                
                # Save matrices
                self.save_matrix(A, f"lamg_{matrix_name}_adjacency")
                self.save_matrix(L_clean, f"lamg_{matrix_name}_laplacian")
                
                print(f"  → {A.shape[0]} nodes, {A.nnz//2} edges")
                
            except Exception as e:
                print(f"Error processing {mtx_file}: {e}")
        
        return lamg_matrices
    
    def extract_cmg_matrices_from_code(self, dataset_path="dataset/cora", level=2, k=10, d=10):
        """Extract matrices by running CMG++ coarsening directly."""
        print(f"Running CMG++ coarsening (level={level}, k={k}, d={d})...")
        
        try:
            # Import your CMG modules (adjust paths as needed)
            import sys
            sys.path.append('.')  # Ensure current directory is in path
            
            from filtered import cmg_filtered_clustering  # Adjust import as needed
            import torch
            from torch_geometric.data import Data
            
            # Load Cora as PyG Data object
            A_orig, L_orig, n_nodes = self.load_cora_original(dataset_path)
            
            # Convert to edge_index format
            coo = A_orig.tocoo()
            edge_index = torch.tensor(np.vstack([coo.row, coo.col]), dtype=torch.long)
            
            # Create PyG Data object
            data = Data(edge_index=edge_index, num_nodes=n_nodes)
            
            # Apply CMG filtering and clustering
            clusters, n_clusters, phi_stats, lambda_crit = cmg_filtered_clustering(
                data, k=k, d=d, threshold=0.1
            )
            
            # Build coarsened adjacency matrix using cluster assignments
            P = self.build_assignment_matrix(clusters, n_clusters, n_nodes)
            A_coarse = P.T @ A_orig @ P
            A_coarse.eliminate_zeros()
            
            # Build coarsened Laplacian
            degrees_coarse = np.array(A_coarse.sum(axis=1)).flatten()
            L_coarse = sp.diags(degrees_coarse) - A_coarse
            
            cmg_result = {
                'adjacency': A_coarse,
                'laplacian': L_coarse,
                'nodes': n_clusters,
                'edges': A_coarse.nnz // 2,
                'assignment_matrix': P,
                'clusters': clusters,
                'lambda_critical': lambda_crit,
                'phi_stats': phi_stats
            }
            
            # Save matrices
            self.save_matrix(A_coarse, f"cmg_level{level}_adjacency")
            self.save_matrix(L_coarse, f"cmg_level{level}_laplacian")
            self.save_matrix(P, f"cmg_level{level}_assignment")
            
            print(f"  → {n_clusters} nodes, {A_coarse.nnz//2} edges")
            print(f"  → λ_critical = {lambda_crit:.6f}")
            
            return cmg_result
            
        except ImportError as e:
            print(f"Error importing CMG modules: {e}")
            print("Please ensure your CMG code files are in the current directory")
            return {}
        except Exception as e:
            print(f"Error running CMG coarsening: {e}")
            return {}
    
    def build_assignment_matrix(self, clusters, n_clusters, n_nodes):
        """Build assignment matrix P from cluster assignments."""
        P = sp.lil_matrix((n_nodes, n_clusters))
        for i, cluster_id in enumerate(clusters):
            P[i, cluster_id] = 1.0
        return P.tocsr()
    
    def save_matrix(self, matrix, name):
        """Save matrix in multiple formats for convenience."""
        base_path = self.output_dir / name
        
        # Save as scipy sparse matrix (pickle)
        with open(f"{base_path}.pkl", 'wb') as f:
            pickle.dump(matrix, f)
        
        # Save as MTX file for MATLAB compatibility
        sio.mmwrite(f"{base_path}.mtx", matrix)
        
        # Save basic info as JSON
        info = {
            'shape': matrix.shape,
            'nnz': matrix.nnz,
            'format': str(type(matrix)),
            'dtype': str(matrix.dtype)
        }
        with open(f"{base_path}_info.json", 'w') as f:
            json.dump(info, f, indent=2)
    
    def load_matrix(self, name):
        """Load previously saved matrix."""
        with open(self.output_dir / f"{name}.pkl", 'rb') as f:
            return pickle.load(f)
    
    def extract_all_matrices(self, dataset_path="dataset/cora", 
                           lamg_results_dir="reduction_results",
                           cmg_level=2, cmg_k=10, cmg_d=10):
        """Extract all matrices for comparison."""
        print("=== Matrix Extraction for LAMG vs CMG++ Analysis ===\n")
        
        results = {}
        
        # 1. Original dataset
        A_orig, L_orig, n_orig = self.load_cora_original(dataset_path)
        results['original'] = {
            'adjacency': A_orig,
            'laplacian': L_orig,
            'nodes': n_orig,
            'edges': A_orig.nnz // 2
        }
        
        # 2. LAMG matrices
        lamg_results = self.extract_lamg_matrices(lamg_results_dir)
        results['lamg'] = lamg_results
        
        # 3. CMG++ matrices
        cmg_result = self.extract_cmg_matrices_from_code(
            dataset_path, level=cmg_level, k=cmg_k, d=cmg_d
        )
        if cmg_result:
            results['cmg'] = cmg_result
        
        # 4. Save summary
        summary = self.create_extraction_summary(results)
        with open(self.output_dir / "extraction_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n=== Extraction Complete ===")
        print(f"Matrices saved to: {self.output_dir}")
        print(f"Summary: {len(results)} method types extracted")
        
        return results
    
    def create_extraction_summary(self, results):
        """Create a summary of extracted matrices."""
        summary = {
            'extraction_timestamp': str(np.datetime64('now')),
            'methods': {}
        }
        
        for method, data in results.items():
            if method == 'original':
                summary['methods'][method] = {
                    'nodes': data['nodes'],
                    'edges': data['edges'],
                    'description': 'Original Cora dataset'
                }
            elif method == 'lamg':
                summary['methods'][method] = {
                    'matrices_found': len(data),
                    'matrices': {name: {'nodes': info['nodes'], 'edges': info['edges']} 
                               for name, info in data.items()},
                    'description': 'LAMG coarsened matrices from .mtx files'
                }
            elif method == 'cmg':
                summary['methods'][method] = {
                    'nodes': data['nodes'],
                    'edges': data['edges'],
                    'lambda_critical': data.get('lambda_critical', 'N/A'),
                    'description': 'CMG++ coarsened matrix from direct computation'
                }
        
        return summary

if __name__ == "__main__":
    # Configuration
    DATASET_PATH = "dataset/cora"
    LAMG_RESULTS_DIR = "reduction_results"
    CMG_LEVEL = 2
    CMG_K = 10
    CMG_D = 10
    
    # Create extractor and run
    extractor = MatrixExtractor()
    
    # Extract all matrices
    results = extractor.extract_all_matrices(
        dataset_path=DATASET_PATH,
        lamg_results_dir=LAMG_RESULTS_DIR,
        cmg_level=CMG_LEVEL,
        cmg_k=CMG_K,
        cmg_d=CMG_D
    )
    
    print("\nNext step: Run spectral analysis on extracted matrices")
    print("Use: python spectral_analyzer.py")
