#!/usr/bin/env python3
"""
Quick analysis of manual tests for Koutis's ideas
"""

import pandas as pd
import numpy as np

def analyze_quick_tests():
    """Analyze the quick test results"""
    try:
        df = pd.read_csv('quick_test_results.csv')
        df = df[df['accuracy'] != 'FAILED'].copy()
        df['accuracy'] = pd.to_numeric(df['accuracy'], errors='coerce')
        df['time'] = pd.to_numeric(df['time'], errors='coerce')
        df = df.dropna()
        
        print("🔍 QUICK TEST ANALYSIS")
        print("=" * 50)
        print(f"✅ Successful tests: {len(df)}")
        print()
        
        # Test 1: Multi-level analysis
        print("📊 TEST 1: Multi-level Effect")
        print("-" * 30)
        level_tests = df[df['test_name'].str.contains('level')]
        if not level_tests.empty:
            print("Level\tAccuracy\tTime(s)\tImprovement")
            baseline_acc = None
            baseline_time = None
            
            for _, row in level_tests.iterrows():
                level = row['level']
                acc = row['accuracy']
                time = row['time']
                
                if 'level1' in row['test_name']:
                    baseline_acc = acc
                    baseline_time = time
                    improvement = "baseline"
                else:
                    improvement = f"{acc - baseline_acc:+.3f}" if baseline_acc else "N/A"
                
                print(f"{level}\t{acc:.3f}\t\t{time:.1f}\t{improvement}")
        
        # Test 2: Parameter hypothesis
        print("\n🎯 TEST 2: Koutis Parameter Hypothesis")
        print("-" * 40)
        param_tests = df[df['test_name'].str.contains('current_best|koutis|moderate')]
        if not param_tests.empty:
            print("Config\t\tk\td\tAccuracy\tTime(s)\tEfficiency")
            for _, row in param_tests.iterrows():
                name = row['test_name'].replace('_', ' ').title()
                k = int(row['k']) if pd.notna(row['k']) else 0
                d = int(row['d']) if pd.notna(row['d']) else 0
                acc = row['accuracy']
                time = row['time']
                efficiency = acc / time * 1000  # accuracy per second * 1000
                print(f"{name:<15}\t{k}\t{d}\t{acc:.3f}\t\t{time:.1f}\t{efficiency:.1f}")
        
        # Test 3: Stability analysis
        print("\n🔄 TEST 3: Stability Analysis")
        print("-" * 30)
        stability_tests = df[df['test_name'].str.contains('stability')]
        if not stability_tests.empty:
            accuracies = stability_tests['accuracy'].values
            times = stability_tests['time'].values
            
            print(f"Accuracy: Mean={np.mean(accuracies):.3f}, Std={np.std(accuracies):.4f}")
            print(f"Time: Mean={np.mean(times):.1f}s, Std={np.std(times):.1f}s")
            print(f"Coefficient of Variation: Acc={np.std(accuracies)/np.mean(accuracies)*100:.1f}%, Time={np.std(times)/np.mean(times)*100:.1f}%")
        
        # Test 4: Method comparison
        print("\n⚖️ TEST 4: Simple vs CMG Comparison")
        print("-" * 35)
        comparison_tests = df[df['test_name'].str.contains('simple|cmg.*comp')]
        if not comparison_tests.empty:
            print("Method\tLevel\tAccuracy\tTime(s)\tSpeedup vs Simple")
            simple_times = {}
            
            for _, row in comparison_tests.iterrows():
                method = row['method']
                level = row['level']
                acc = row['accuracy']
                time = row['time']
                
                if method == 'simple':
                    simple_times[level] = time
                    speedup = "baseline"
                else:
                    speedup = f"{simple_times.get(level, time) / time:.1f}x" if level in simple_times else "N/A"
                
                print(f"{method}\t{level}\t{acc:.3f}\t\t{time:.1f}\t{speedup}")
        
        # Overall insights
        print("\n💡 KEY INSIGHTS:")
        print("-" * 15)
        
        # Best performing config
        best_config = df.loc[df['accuracy'].idxmax()]
        print(f"🏆 Best accuracy: {best_config['test_name']} ({best_config['accuracy']:.3f})")
        
        # Fastest config
        fastest_config = df.loc[df['time'].idxmin()]
        print(f"⚡ Fastest: {fastest_config['test_name']} ({fastest_config['time']:.1f}s)")
        
        # Parameter insights
        if 'koutis' in df['test_name'].str.cat():
            koutis_tests = df[df['test_name'].str.contains('koutis')]
            current_test = df[df['test_name'].str.contains('current_best')]
            
            if not koutis_tests.empty and not current_test.empty:
                koutis_best = koutis_tests.loc[koutis_tests['accuracy'].idxmax()]
                current_best = current_test.iloc[0]
                
                print(f"📈 Koutis hypothesis result: {koutis_best['accuracy']:.3f} vs {current_best['accuracy']:.3f}")
                print(f"   Parameter trade-off: k={koutis_best['k']}, d={koutis_best['d']} vs k={current_best['k']}, d={current_best['d']}")
        
        print("\n📋 RECOMMENDATIONS:")
        print("-" * 20)
        print("1. Test multi-level CMG if level 2+ shows improvement")
        print("2. Consider Koutis's parameter suggestions if they outperform current")
        print("3. CMG stability appears good (low variance)")
        print("4. Focus on configurations with best accuracy/time trade-off")
        
    except FileNotFoundError:
        print("❌ quick_test_results.csv not found. Run the test script first.")
    except Exception as e:
        print(f"❌ Error analyzing results: {e}")

if __name__ == "__main__":
    analyze_quick_tests()
