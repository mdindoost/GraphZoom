#!/usr/bin/env python3
"""
Test Enhanced Two-Level Training for True Coarsened GraphSAGE

Quick validation script to ensure the enhanced approach works before integration.
"""

import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import sys
import os

# Add path for imports
sys.path.append('.')
sys.path.append('./embed_methods/graphsage/')

def test_enhanced_level1_training():
    """Test the enhanced Level 1 trainable intra-cluster GraphSAGE"""
    
    print("🧪 TESTING ENHANCED LEVEL 1 TRAINING")
    print("="*50)
    
    # Create test cluster subgraph (small triangle)
    cluster_graph = nx.Graph()
    cluster_graph.add_edges_from([(0, 1), (1, 2), (2, 0)])
    
    # Create test features
    cluster_features = np.array([
        [1.0, 0.0, 0.5],
        [0.0, 1.0, 0.5],
        [0.5, 0.5, 1.0]
    ])
    
    print(f"Test cluster: {list(cluster_graph.nodes())}")
    print(f"Test features shape: {cluster_features.shape}")
    
    # Import the enhanced training function
    try:
        from enhanced_true_coarsened_graphsage import train_intra_cluster_model
        
        # Test training
        super_feature = train_intra_cluster_model(
            cluster_graph, cluster_features,
            embed_dim=8, hidden_dim=4, epochs=10, lr=0.01
        )
        
        print(f"✅ Level 1 training successful!")
        print(f"Super-feature shape: {super_feature.shape}")
        print(f"Super-feature: {super_feature}")
        
        # Validate output
        assert super_feature.shape == (8,), f"Expected shape (8,), got {super_feature.shape}"
        assert np.isfinite(super_feature).all(), "Super-feature contains NaN/Inf"
        
        print("✅ All Level 1 tests passed!")
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Level 1 training failed: {e}")
        return False

def test_epoch_distribution():
    """Test the training epoch distribution calculations"""
    
    print("\n🧪 TESTING EPOCH DISTRIBUTION")
    print("="*50)
    
    try:
        from enhanced_true_coarsened_graphsage import (
            calculate_training_distribution,
            calculate_strategic_epochs
        )
        
        # Test parameters
        original_nodes = 2708  # Cora size
        coarsened_nodes = 1025  # Typical CMG result
        base_epochs = 1000
        
        print(f"Test scenario: {original_nodes} → {coarsened_nodes} nodes")
        
        # Test strategic epochs calculation
        strategies = ["speed_advantage", "quality_advantage", "fair_comparison"]
        
        for strategy in strategies:
            total_epochs = calculate_strategic_epochs(
                base_epochs, original_nodes, coarsened_nodes, strategy
            )
            print(f"\n{strategy}: {total_epochs} total epochs")
            
            # Test training distribution
            distributions = ["balanced", "level1_heavy", "level2_heavy", "adaptive"]
            
            for dist in distributions:
                level1_epochs, level2_epochs = calculate_training_distribution(
                    total_epochs, original_nodes, coarsened_nodes, dist
                )
                print(f"  {dist}: L1={level1_epochs}, L2={level2_epochs}")
                
                # Validate
                assert level1_epochs > 0, f"Level 1 epochs must be > 0"
                assert level2_epochs > 0, f"Level 2 epochs must be > 0"
                assert level1_epochs + level2_epochs <= total_epochs + 1, f"Total exceeds budget"
        
        print("✅ All epoch distribution tests passed!")
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Epoch distribution test failed: {e}")
        return False

def test_full_pipeline_small():
    """Test the full enhanced pipeline on a small synthetic graph"""
    
    print("\n🧪 TESTING FULL ENHANCED PIPELINE")
    print("="*50)
    
    try:
        # Create small test graph
        G = nx.karate_club_graph()
        n_nodes = G.number_of_nodes()
        
        # Create test features
        features = np.random.randn(n_nodes, 16)
        
        # Create simple clusters (divide into 3 groups)
        clusters = [
            list(range(0, 11)),      # First group
            list(range(11, 22)),     # Second group  
            list(range(22, n_nodes)) # Remaining nodes
        ]
        
        print(f"Test graph: {n_nodes} nodes, {G.number_of_edges()} edges")
        print(f"Features shape: {features.shape}")
        print(f"Clusters: {len(clusters)} clusters")
        
        # Create simple coarsened graph
        coarsened_G = nx.Graph()
        coarsened_G.add_nodes_from(range(len(clusters)))
        coarsened_G.add_edges_from([(0,1), (1,2)])  # Linear chain
        
        # Create dummy projections and laplacians
        import scipy.sparse as sp
        projection = sp.lil_matrix((n_nodes, len(clusters)))
        for node_id in range(n_nodes):
            for cluster_id, cluster in enumerate(clusters):
                if node_id in cluster:
                    projection[node_id, cluster_id] = 1.0
        
        projections = [projection.tocsr()]
        laplacians = [nx.laplacian_matrix(G)]
        
        # Test enhanced pipeline
        from enhanced_true_coarsened_graphsage import enhanced_true_coarsened_graphsage
        
        embeddings = enhanced_true_coarsened_graphsage(
            original_graph=G,
            features=features,
            clusters=clusters,
            coarsened_graph=coarsened_G,
            projections=projections,
            laplacians=laplacians,
            super_embed_dim=16,
            final_embed_dim=16,
            hidden_dim=8,
            base_epochs=100,  # Small for testing
            computational_strategy="fair_comparison",
            training_distribution="balanced"
        )
        
        print(f"✅ Full pipeline successful!")
        print(f"Output embeddings shape: {embeddings.shape}")
        
        # Validate output
        expected_shape = (n_nodes, 16)
        assert embeddings.shape == expected_shape, f"Expected {expected_shape}, got {embeddings.shape}"
        assert np.isfinite(embeddings).all(), "Embeddings contain NaN/Inf"
        
        # Check diversity
        unique_embeddings = []
        for embedding in embeddings:
            is_unique = True
            for unique_emb in unique_embeddings:
                if np.allclose(embedding, unique_emb, atol=1e-6):
                    is_unique = False
                    break
            if is_unique:
                unique_embeddings.append(embedding)
        
        diversity = len(unique_embeddings) / len(embeddings)
        print(f"Embedding diversity: {diversity:.3f}")
        
        assert diversity > 0.8, f"Low diversity: {diversity:.3f}"
        
        print("✅ All full pipeline tests passed!")
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        print("💡 Make sure enhanced_true_coarsened_graphsage.py is in the current directory")
        return False
    except Exception as e:
        print(f"❌ Full pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    
    print("🚀 TESTING ENHANCED TWO-LEVEL TRAINING")
    print("="*60)
    
    all_passed = True
    
    # Test 1: Enhanced Level 1 training
    test1_passed = test_enhanced_level1_training()
    all_passed = all_passed and test1_passed
    
    # Test 2: Epoch distribution calculations
    test2_passed = test_epoch_distribution() 
    all_passed = all_passed and test2_passed
    
    # Test 3: Full pipeline (requires existing functions)
    test3_passed = test_full_pipeline_small()
    all_passed = all_passed and test3_passed
    
    # Summary
    print(f"\n{'='*60}")
    print(f"TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Level 1 Training:     {'✅ PASS' if test1_passed else '❌ FAIL'}")
    print(f"Epoch Distribution:   {'✅ PASS' if test2_passed else '❌ FAIL'}")
    print(f"Full Pipeline:        {'✅ PASS' if test3_passed else '❌ FAIL'}")
    print(f"Overall:              {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    
    if all_passed:
        print(f"\n🎉 READY FOR INTEGRATION!")
        print(f"💡 Next steps:")
        print(f"   1. Save enhanced_true_coarsened_graphsage.py to ./embed_methods/graphsage/")
        print(f"   2. Update your main graphzoom_timed_mpaware.py with the integration code")
        print(f"   3. Test on Cora dataset")
    else:
        print(f"\n🔧 FIXES NEEDED before integration")
        print(f"💡 Address the failed tests above first")
    
    return all_passed

if __name__ == "__main__":
    main()
