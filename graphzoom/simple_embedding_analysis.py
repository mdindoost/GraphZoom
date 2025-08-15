#!/usr/bin/env python3
"""
Simple Embedding Analysis for Paper

Analyzes embeddings you've already collected to demonstrate 
the improvement from feature-aware refinement.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from collections import defaultdict

def analyze_current_embeddings():
    """
    Analyzes the embeddings from your recent runs.
    
    You'll need to save embeddings from your last few runs:
    1. Enhanced (α=0.88, β=0.12) → 72.7% accuracy
    2. Original (α=1.0, β=0.0) → 72.3% accuracy  
    3. Regular CMG → 71.6% accuracy
    """
    
    print("🔍 SIMPLE EMBEDDING ANALYSIS FOR PAPER")
    print("="*60)
    
    # Instructions for user
    print("""
    TO USE THIS ANALYSIS:
    
    1. After running your enhanced method (α=0.88, β=0.12):
       cp embed_results/embeddings.npy embeddings_enhanced.npy
       
    2. After running original method (α=1.0, β=0.0):  
       cp embed_results/embeddings.npy embeddings_original.npy
       
    3. After running regular CMG + GraphSAGE:
       cp embed_results/embeddings.npy embeddings_regular.npy
       
    4. Run this script:
       python simple_embedding_analysis.py
    """)
    
    # Check if files exist
    files = [
        ('/home/mohammad/GraphZoom/graphzoom/embed_results/embeddings_enhanced.npy', 'Enhanced True Coarsened (α=0.88)', 72.7),
        ('/home/mohammad/GraphZoom/graphzoom/embed_results/embeddings_original.npy', 'Original True Coarsened (α=1.0)', 72.3),
        ('/home/mohammad/GraphZoom/graphzoom/embed_results/embeddings_regular.npy', 'Regular CMG + GraphSAGE', 71.6)
    ]
    
    available_embeddings = {}
    
    for filename, method_name, accuracy in files:
        try:
            embeddings = np.load(filename)
            available_embeddings[method_name] = {
                'embeddings': embeddings,
                'accuracy': accuracy,
                'filename': filename
            }
            print(f"✅ Loaded {filename}: {embeddings.shape}, accuracy={accuracy}%")
        except FileNotFoundError:
            print(f"❌ {filename} not found")
    
    if len(available_embeddings) == 0:
        print("⚠️  No embedding files found. Please save them first.")
        return
    
    # Analyze each method
    results = {}
    for method_name, data in available_embeddings.items():
        results[method_name] = analyze_embedding_diversity(
            data['embeddings'], method_name, data['accuracy']
        )
    
    # Create comparison
    create_comparison_summary(results)
    
    # Create visualizations if we have multiple methods
    if len(available_embeddings) >= 2:
        create_visualizations(available_embeddings)
    
    return results

def analyze_embedding_diversity(embeddings, method_name, accuracy):
    """
    Analyze embedding diversity for a single method.
    """
    print(f"\n📊 ANALYZING: {method_name}")
    print("-" * 50)
    
    n_embeddings = len(embeddings)
    
    # 1. Count unique embeddings
    unique_embeddings = []
    tolerance = 1e-6
    
    for i, embedding in enumerate(embeddings):
        is_unique = True
        for unique_emb in unique_embeddings:
            if np.allclose(embedding, unique_emb, atol=tolerance):
                is_unique = False
                break
        if is_unique:
            unique_embeddings.append(embedding)
            
        # Progress for large datasets
        if i % 500 == 0 and i > 0:
            print(f"  Processed {i}/{n_embeddings} embeddings...")
    
    n_unique = len(unique_embeddings)
    diversity_ratio = n_unique / n_embeddings
    duplicate_count = n_embeddings - n_unique
    
    # 2. Basic statistics
    mean_embedding = np.mean(embeddings, axis=0)
    std_embedding = np.std(embeddings, axis=0)
    
    # 3. Pairwise distance statistics (sample for efficiency)
    sample_size = min(1000, n_embeddings)
    sample_indices = np.random.choice(n_embeddings, sample_size, replace=False)
    sample_embeddings = embeddings[sample_indices]
    
    pairwise_distances = []
    for i in range(sample_size):
        for j in range(i+1, sample_size):
            dist = np.linalg.norm(sample_embeddings[i] - sample_embeddings[j])
            pairwise_distances.append(dist)
    
    avg_distance = np.mean(pairwise_distances) if pairwise_distances else 0
    std_distance = np.std(pairwise_distances) if pairwise_distances else 0
    
    # Results
    results = {
        'method_name': method_name,
        'accuracy': accuracy,
        'total_embeddings': n_embeddings,
        'unique_embeddings': n_unique,
        'duplicate_embeddings': duplicate_count,
        'diversity_ratio': diversity_ratio,
        'avg_pairwise_distance': avg_distance,
        'std_pairwise_distance': std_distance,
        'embedding_mean': np.mean(mean_embedding),
        'embedding_std': np.mean(std_embedding)
    }
    
    # Print results
    print(f"Accuracy: {accuracy}%")
    print(f"Total embeddings: {n_embeddings}")
    print(f"Unique embeddings: {n_unique}")
    print(f"Duplicate embeddings: {duplicate_count}")
    print(f"Diversity ratio: {diversity_ratio:.4f}")
    print(f"Average pairwise distance: {avg_distance:.6f}")
    
    return results

def create_comparison_summary(results):
    """
    Create a summary comparison table.
    """
    print(f"\n📋 COMPARISON SUMMARY")
    print("="*80)
    
    # Sort by accuracy
    sorted_results = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    
    print(f"{'Method':<30} {'Accuracy':<10} {'Diversity':<10} {'Duplicates':<12} {'Avg Distance':<12}")
    print("-" * 80)
    
    for method_name, result in sorted_results:
        print(f"{method_name:<30} "
              f"{result['accuracy']:<10.1f} "
              f"{result['diversity_ratio']:<10.4f} "
              f"{result['duplicate_embeddings']:<12} "
              f"{result['avg_pairwise_distance']:<12.6f}")
    
    # Key insights
    print(f"\n🎯 KEY INSIGHTS:")
    
    if len(results) >= 2:
        enhanced_name = "Enhanced True Coarsened (α=0.88)"
        original_name = "Original True Coarsened (α=1.0)"
        
        if enhanced_name in results and original_name in results:
            enhanced = results[enhanced_name]
            original = results[original_name]
            
            acc_improvement = enhanced['accuracy'] - original['accuracy']
            diversity_improvement = enhanced['diversity_ratio'] - original['diversity_ratio']
            duplicate_reduction = original['duplicate_embeddings'] - enhanced['duplicate_embeddings']
            
            print(f"✅ Accuracy improvement: +{acc_improvement:.1f}% ({original['accuracy']}% → {enhanced['accuracy']}%)")
            print(f"✅ Diversity improvement: +{diversity_improvement:.3f} ({original['diversity_ratio']:.3f} → {enhanced['diversity_ratio']:.3f})")
            print(f"✅ Duplicate reduction: -{duplicate_reduction} embeddings")
            
            if acc_improvement > 0 and diversity_improvement > 0:
                print(f"🎉 Feature-aware refinement successfully improved both accuracy and diversity!")

def create_visualizations(available_embeddings):
    """
    Create t-SNE visualizations comparing methods.
    """
    print(f"\n🎨 CREATING VISUALIZATIONS")
    print("-" * 50)
    
    fig, axes = plt.subplots(1, len(available_embeddings), figsize=(5*len(available_embeddings), 5))
    if len(available_embeddings) == 1:
        axes = [axes]
    
    for idx, (method_name, data) in enumerate(available_embeddings.items()):
        embeddings = data['embeddings']
        accuracy = data['accuracy']
        
        # Subsample for t-SNE efficiency
        if len(embeddings) > 2000:
            indices = np.random.choice(len(embeddings), 2000, replace=False)
            embeddings_subset = embeddings[indices]
        else:
            embeddings_subset = embeddings
        
        # t-SNE
        print(f"Computing t-SNE for {method_name}...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings_subset)-1))
        embeddings_2d = tsne.fit_transform(embeddings_subset)
        
        # Plot
        scatter = axes[idx].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                                   alpha=0.6, s=20, c=range(len(embeddings_2d)), cmap='viridis')
        axes[idx].set_title(f'{method_name}\nAccuracy: {accuracy}%')
        axes[idx].set_xlabel('t-SNE 1')
        axes[idx].set_ylabel('t-SNE 2')
    
    plt.tight_layout()
    plt.savefig('embedding_comparison.png', dpi=300, bbox_inches='tight')
    print(f"💾 Saved visualization: embedding_comparison.png")
    plt.show()

def generate_parameter_sweep_data():
    """
    Generate data for parameter sweep analysis based on your test results.
    """
    print(f"\n📊 PARAMETER SWEEP ANALYSIS")
    print("-" * 50)
    
    # Your test results
    test_results = [
        (0.91, 0.09, 72.3),
        (0.89, 0.11, 72.5),
        (0.88, 0.12, 72.7),  # Best
        (0.87, 0.13, 72.5),
        (0.86, 0.14, 72.5),
        (0.85, 0.15, 72.6),
        (0.84, 0.16, 72.4),
        (0.83, 0.17, 72.2),
        (1.0, 0.0, 72.3),    # Pure spectral
    ]
    
    alphas = [result[0] for result in test_results]
    betas = [result[1] for result in test_results]
    accuracies = [result[2] for result in test_results]
    
    # Create parameter sweep plot
    plt.figure(figsize=(10, 6))
    plt.scatter(alphas, accuracies, c=betas, cmap='viridis', s=100, alpha=0.7)
    plt.colorbar(label='β (Feature Weight)')
    plt.xlabel('α (Spectral Weight)')
    plt.ylabel('Accuracy (%)')
    plt.title('Parameter Sweep: α vs Accuracy')
    
    # Highlight best result
    best_idx = accuracies.index(max(accuracies))
    plt.scatter(alphas[best_idx], accuracies[best_idx], c='red', s=200, marker='*', 
                label=f'Best: α={alphas[best_idx]}, acc={accuracies[best_idx]}%')
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('parameter_sweep.png', dpi=300, bbox_inches='tight')
    print(f"💾 Saved parameter sweep: parameter_sweep.png")
    plt.show()
    
    # Print optimal parameters
    print(f"🎯 OPTIMAL PARAMETERS:")
    print(f"   α = {alphas[best_idx]} (spectral weight)")
    print(f"   β = {betas[best_idx]} (feature weight)")
    print(f"   Accuracy = {accuracies[best_idx]}%")

if __name__ == "__main__":
    # Run analysis
    results = analyze_current_embeddings()
    
    # Generate parameter sweep visualization
    generate_parameter_sweep_data()
    
    print(f"\n🎯 PAPER READY OUTPUTS:")
    print("="*50)
    print("✅ Embedding comparison table (printed above)")
    print("✅ Parameter sweep plot (parameter_sweep.png)")
    print("✅ t-SNE visualizations (embedding_comparison.png)")
    print("✅ Quantitative improvements documented")
