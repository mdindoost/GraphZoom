#!/usr/bin/env python3
"""
Corrected analysis focusing on k-NN effects, clustering, and compression
"""

import pandas as pd
import numpy as np

def analyze_corrected_tests():
    """Analyze the corrected test results with clustering focus"""
    try:
        df = pd.read_csv('corrected_test_results.csv')
        df = df[df['accuracy'] != 'FAILED'].copy()
        df['accuracy'] = pd.to_numeric(df['accuracy'], errors='coerce')
        df['time'] = pd.to_numeric(df['time'], errors='coerce')
        
        # Handle cluster and compression data
        df['clusters_found'] = pd.to_numeric(df['clusters_found'], errors='coerce')
        df['compression_ratio'] = pd.to_numeric(df['compression_ratio'], errors='coerce')
        
        df = df.dropna(subset=['accuracy'])
        
        print("🔍 CORRECTED TEST ANALYSIS - CLUSTERING & COMPRESSION FOCUS")
        print("=" * 65)
        print(f"✅ Successful tests: {len(df)}")
        print()
        
        # Test 1: K-NN Effect Analysis (Koutis's actual suggestion)
        print("📊 TEST 1: K-NN Effect in GraphZoom Fusion")
        print("-" * 45)
        knn_tests = df[df['test_name'].str.contains('knn')]
        if not knn_tests.empty:
            print("K-NN\tClusters\tCompression\tAccuracy\tTime(s)\tEfficiency")
            print("-" * 60)
            for _, row in knn_tests.iterrows():
                knn = int(row['knn_neighbors']) if pd.notna(row['knn_neighbors']) else 0
                clusters = int(row['clusters_found']) if pd.notna(row['clusters_found']) else 'N/A'
                compression = f"{row['compression_ratio']:.1f}x" if pd.notna(row['compression_ratio']) else 'N/A'
                acc = row['accuracy']
                time = row['time']
                efficiency = acc / time * 1000 if pd.notna(time) and time > 0 else 0
                print(f"{knn}\t{clusters}\t\t{compression}\t\t{acc:.3f}\t\t{time:.1f}\t{efficiency:.1f}")
            
            # K-NN insights
            best_knn = knn_tests.loc[knn_tests['accuracy'].idxmax()]
            print(f"\n🏆 Best K-NN: {int(best_knn['knn_neighbors'])} (Accuracy: {best_knn['accuracy']:.3f})")
            
            # Compression vs accuracy trade-off
            if knn_tests['compression_ratio'].notna().any():
                print(f"📊 Compression range: {knn_tests['compression_ratio'].min():.1f}x - {knn_tests['compression_ratio'].max():.1f}x")
        
        # Test 2: CMG Parameter Efficiency (smaller d)
        print(f"\n🎯 TEST 2: CMG Parameter Efficiency (Smaller d)")
        print("-" * 45)
        param_tests = df[df['test_name'].str.contains('current_best|smaller_d')]
        if not param_tests.empty:
            print("Config\t\td\tClusters\tCompression\tAccuracy\tTime(s)")
            print("-" * 55)
            for _, row in param_tests.iterrows():
                config = row['test_name'].replace('_', ' ').title()[:12]
                d = int(row['cmg_d']) if pd.notna(row['cmg_d']) else 0
                clusters = int(row['clusters_found']) if pd.notna(row['clusters_found']) else 'N/A'
                compression = f"{row['compression_ratio']:.1f}x" if pd.notna(row['compression_ratio']) else 'N/A'
                acc = row['accuracy']
                time = row['time']
                print(f"{config:<12}\t{d}\t{clusters}\t\t{compression}\t\t{acc:.3f}\t\t{time:.1f}")
            
            # Efficiency analysis
            baseline = param_tests[param_tests['test_name'] == 'current_best']
            if not baseline.empty:
                baseline_acc = baseline.iloc[0]['accuracy']
                print(f"\n📈 Parameter Efficiency Analysis:")
                for _, row in param_tests.iterrows():
                    if row['test_name'] != 'current_best':
                        improvement = row['accuracy'] - baseline_acc
                        d_reduction = 20 - row['cmg_d']
                        print(f"   d={int(row['cmg_d'])}: {improvement:+.3f} accuracy, {d_reduction} dimensions saved")
        
        # Test 3: CMG vs Simple Clustering Comparison
        print(f"\n⚖️ TEST 3: CMG vs Simple Clustering Comparison")
        print("-" * 45)
        comparison_tests = df[df['test_name'].str.contains('baseline|simple_knn|cmg_knn')]
        if not comparison_tests.empty:
            print("Method\tK-NN\tClusters\tCompression\tAccuracy\tTime(s)\tSpeedup")
            print("-" * 65)
            
            simple_baseline_time = None
            for _, row in comparison_tests.iterrows():
                method = row['method']
                knn = int(row['knn_neighbors']) if pd.notna(row['knn_neighbors']) else 0
                clusters = int(row['clusters_found']) if pd.notna(row['clusters_found']) else 'N/A'
                compression = f"{row['compression_ratio']:.1f}x" if pd.notna(row['compression_ratio']) else 'N/A'
                acc = row['accuracy']
                time = row['time']
                
                if 'simple_baseline' in row['test_name']:
                    simple_baseline_time = time
                    speedup = "baseline"
                else:
                    speedup = f"{simple_baseline_time / time:.1f}x" if simple_baseline_time and time > 0 else 'N/A'
                
                print(f"{method}\t{knn}\t{clusters}\t\t{compression}\t\t{acc:.3f}\t\t{time:.1f}\t{speedup}")
            
            # Clustering comparison
            simple_clusters = comparison_tests[comparison_tests['method'] == 'simple']['clusters_found']
            cmg_clusters = comparison_tests[comparison_tests['method'] == 'cmg']['clusters_found']
            
            if not simple_clusters.empty and not cmg_clusters.empty:
                simple_avg = simple_clusters.mean()
                cmg_avg = cmg_clusters.mean()
                print(f"\n📊 Clustering Comparison:")
                print(f"   Simple coarsening: {simple_avg:.0f} clusters on average")
                print(f"   CMG coarsening: {cmg_avg:.0f} clusters on average")
                print(f"   CMG creates {((cmg_avg - simple_avg) / simple_avg * 100):+.1f}% {'more' if cmg_avg > simple_avg else 'fewer'} clusters")
        
        # Test 4: Multi-level Compression Analysis
        print(f"\n🔄 TEST 4: Multi-level Compression Analysis")
        print("-" * 40)
        multilevel_tests = df[df['test_name'].str.contains('compression')]
        if not multilevel_tests.empty:
            print("Level\tClusters\tCompression\tAccuracy\tTime(s)\tSpeedup vs L1")
            print("-" * 55)
            
            level1_time = None
            for _, row in multilevel_tests.iterrows():
                level = int(row['level'])
                clusters = int(row['clusters_found']) if pd.notna(row['clusters_found']) else 'N/A'
                compression = f"{row['compression_ratio']:.1f}x" if pd.notna(row['compression_ratio']) else 'N/A'
                acc = row['accuracy']
                time = row['time']
                
                if level == 1:
                    level1_time = time
                    speedup = "baseline"
                else:
                    speedup = f"{level1_time / time:.1f}x" if level1_time and time > 0 else 'N/A'
                
                print(f"{level}\t{clusters}\t\t{compression}\t\t{acc:.3f}\t\t{time:.1f}\t{speedup}")
            
            # Multi-level insights
            if len(multilevel_tests) > 1:
                best_compression = multilevel_tests.loc[multilevel_tests['compression_ratio'].idxmax()]
                fastest = multilevel_tests.loc[multilevel_tests['time'].idxmin()]
                print(f"\n📈 Multi-level Insights:")
                print(f"   Best compression: Level {int(best_compression['level'])} ({best_compression['compression_ratio']:.1f}x)")
                print(f"   Fastest: Level {int(fastest['level'])} ({fastest['time']:.1f}s)")
        
        # Overall insights and recommendations
        print(f"\n💡 KEY INSIGHTS:")
        print("-" * 15)
        
        # Best configurations
        if not df.empty:
            best_overall = df.loc[df['accuracy'].idxmax()]
            fastest_overall = df.loc[df['time'].idxmin()]
            
            print(f"🏆 Best accuracy: {best_overall['test_name']} ({best_overall['accuracy']:.3f})")
            print(f"⚡ Fastest: {fastest_overall['test_name']} ({fastest_overall['time']:.1f}s)")
            
            if pd.notna(best_overall['clusters_found']):
                print(f"📊 Best config clusters: {int(best_overall['clusters_found'])} ({best_overall['compression_ratio']:.1f}x compression)")
        
        # K-NN effect summary
        knn_tests = df[df['test_name'].str.contains('knn')]
        if not knn_tests.empty:
            knn_effect = knn_tests.groupby('knn_neighbors')['accuracy'].mean()
            print(f"\n📈 K-NN Effect Summary:")
            for knn, acc in knn_effect.items():
                print(f"   K-NN={int(knn)}: {acc:.3f} average accuracy")
        
        # Parameter efficiency
        param_tests = df[df['test_name'].str.contains('smaller_d')]
        if not param_tests.empty:
            print(f"\n🎯 Parameter Efficiency (Koutis's suggestion):")
            current_best = df[df['test_name'] == 'current_best']
            if not current_best.empty:
                baseline_acc = current_best.iloc[0]['accuracy']
                best_smaller_d = param_tests.loc[param_tests['accuracy'].idxmax()]
                print(f"   Current d=20: {baseline_acc:.3f}")
                print(f"   Best smaller d={int(best_smaller_d['cmg_d'])}: {best_smaller_d['accuracy']:.3f}")
                print(f"   Koutis hypothesis: {'✅ CONFIRMED' if best_smaller_d['accuracy'] > baseline_acc else '❌ NOT CONFIRMED'}")
        
        print(f"\n📋 RECOMMENDATIONS:")
        print("-" * 20)
        print("1. Optimal K-NN value for graph fusion")
        print("2. CMG vs Simple clustering efficiency comparison")
        print("3. Best compression vs accuracy trade-off")
        print("4. Validate smaller d efficiency on larger datasets")
        
    except FileNotFoundError:
        print("❌ corrected_test_results.csv not found. Run the corrected test script first.")
    except Exception as e:
        print(f"❌ Error analyzing results: {e}")

if __name__ == "__main__":
    analyze_corrected_tests()
