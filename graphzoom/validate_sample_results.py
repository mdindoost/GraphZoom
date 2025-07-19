#!/usr/bin/env python3
"""
Sample Results Validator
Quick validation of sample test CSV output before running full study
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

def validate_sample_csv(csv_path):
    """Validate sample CSV output and check data quality"""
    print("🔍 SAMPLE CSV VALIDATION")
    print("=" * 30)
    
    try:
        df = pd.read_csv(csv_path)
        print(f"✅ Successfully loaded CSV: {csv_path}")
        print(f"📊 Total rows: {len(df)}")
        
        # Check columns
        expected_cols = [
            'experiment_type', 'method', 'dataset', 'embedding', 'run_id',
            'dimension', 'k_param', 'level', 'beta', 'reduce_ratio', 'search_ratio',
            'accuracy', 'total_time', 'fusion_time', 'reduction_time', 'embedding_time',
            'refinement_time', 'clustering_time', 'memory_mb', 'original_nodes',
            'final_clusters', 'compression_ratio', 'speedup_vs_vanilla', 'notes'
        ]
        
        missing_cols = [col for col in expected_cols if col not in df.columns]
        if missing_cols:
            print(f"⚠️  Missing columns: {missing_cols}")
        else:
            print("✅ All expected columns present")
        
        # Check data quality
        print(f"\n📋 DATA QUALITY CHECK:")
        print(f"   Completed tests: {len(df[df['notes'] == 'completed'])}")
        print(f"   Failed tests: {len(df[df['notes'] == 'failed'])}")
        
        # Check each experiment type
        print(f"\n📊 EXPERIMENT TYPES:")
        for exp_type in df['experiment_type'].unique():
            exp_data = df[df['experiment_type'] == exp_type]
            completed = len(exp_data[exp_data['notes'] == 'completed'])
            total = len(exp_data)
            print(f"   {exp_type}: {completed}/{total} completed")
        
        # Check methods
        print(f"\n🔬 METHODS TESTED:")
        for method in df['method'].unique():
            method_data = df[df['method'] == method]
            completed = len(method_data[method_data['notes'] == 'completed'])
            total = len(method_data)
            print(f"   {method}: {completed}/{total} completed")
        
        # Check LAMG specifically
        lamg_data = df[df['method'] == 'lamg']
        if not lamg_data.empty:
            print(f"\n⚡ LAMG SPECIFIC VALIDATION:")
            lamg_completed = lamg_data[lamg_data['notes'] == 'completed']
            print(f"   LAMG tests completed: {len(lamg_completed)}")
            
            if not lamg_completed.empty:
                print(f"   LAMG timing data:")
                for _, row in lamg_completed.iterrows():
                    print(f"     Total time: {row['total_time']}s")
                    print(f"     Clustering time: {row['clustering_time']}s")
                    print(f"     Final clusters: {row['final_clusters']}")
                    print(f"     Compression: {row['compression_ratio']}x")
        else:
            print(f"\n❌ No LAMG tests found")
        
        # Check data ranges
        print(f"\n📈 DATA RANGES:")
        completed_data = df[df['notes'] == 'completed']
        if not completed_data.empty:
            numeric_cols = ['accuracy', 'total_time', 'compression_ratio']
            for col in numeric_cols:
                if col in completed_data.columns:
                    values = pd.to_numeric(completed_data[col], errors='coerce').dropna()
                    if not values.empty:
                        print(f"   {col}: {values.min():.3f} - {values.max():.3f}")
        
        # Sample data preview
        print(f"\n📋 SAMPLE DATA (first few completed tests):")
        completed_sample = df[df['notes'] == 'completed'].head(3)
        if not completed_sample.empty:
            for _, row in completed_sample.iterrows():
                print(f"   {row['experiment_type']} | {row['method']} | {row['dataset']} | "
                      f"acc={row['accuracy']} | time={row['total_time']}s | "
                      f"clusters={row['final_clusters']}")
        
        # Validation summary
        total_tests = len(df)
        completed_tests = len(df[df['notes'] == 'completed'])
        success_rate = (completed_tests / total_tests * 100) if total_tests > 0 else 0
        
        print(f"\n🎯 VALIDATION SUMMARY:")
        print(f"   Success rate: {success_rate:.1f}% ({completed_tests}/{total_tests})")
        
        if success_rate >= 80:
            print("   ✅ VALIDATION PASSED - Ready for full study")
            return True
        elif success_rate >= 50:
            print("   🟡 PARTIAL SUCCESS - Some issues but can proceed")
            return True
        else:
            print("   ❌ VALIDATION FAILED - Too many failures")
            return False
            
    except Exception as e:
        print(f"❌ Error validating CSV: {e}")
        return False

def check_koutis_requirements(csv_path):
    """Check if sample covers all of Koutis's specific requirements"""
    print(f"\n🎯 KOUTIS REQUIREMENTS CHECK")
    print("=" * 30)
    
    try:
        df = pd.read_csv(csv_path)
        completed_data = df[df['notes'] == 'completed']
        
        requirements = {
            "Vanilla baselines": len(completed_data[completed_data['experiment_type'] == 'vanilla_baseline']) > 0,
            "Dimension scaling": len(completed_data[completed_data['experiment_type'] == 'dimension_scaling']) > 0,
            "Hyperparameter robustness": len(completed_data[completed_data['experiment_type'] == 'hyperparameter_robustness']) > 0,
            "Multilevel comparison": len(completed_data[completed_data['experiment_type'] == 'multilevel_comparison']) > 0,
            "Computational efficiency": len(completed_data[completed_data['experiment_type'] == 'computational_efficiency']) > 0,
            "CMG++ method": len(completed_data[completed_data['method'] == 'cmg']) > 0,
            "LAMG method": len(completed_data[completed_data['method'] == 'lamg']) > 0,
            "Multiple dimensions": len(completed_data['dimension'].unique()) > 1,
            "Timing data": completed_data['total_time'].notna().sum() > 0,
            "Clustering data": completed_data['final_clusters'].notna().sum() > 0
        }
        
        print("📋 Requirements coverage:")
        all_passed = True
        for req, passed in requirements.items():
            status = "✅" if passed else "❌"
            print(f"   {status} {req}")
            if not passed:
                all_passed = False
        
        print(f"\n🎯 Koutis requirements: {'✅ ALL MET' if all_passed else '❌ SOME MISSING'}")
        return all_passed
        
    except Exception as e:
        print(f"❌ Error checking requirements: {e}")
        return False

def generate_sample_insights(csv_path):
    """Generate quick insights from sample data"""
    print(f"\n📊 SAMPLE INSIGHTS")
    print("=" * 20)
    
    try:
        df = pd.read_csv(csv_path)
        completed_data = df[df['notes'] == 'completed']
        
        if completed_data.empty:
            print("❌ No completed tests to analyze")
            return
        
        # Convert to numeric
        numeric_cols = ['accuracy', 'total_time', 'compression_ratio']
        for col in numeric_cols:
            completed_data[col] = pd.to_numeric(completed_data[col], errors='coerce')
        
        # Method comparison
        print("🔬 Method Performance (sample):")
        for method in completed_data['method'].unique():
            method_data = completed_data[completed_data['method'] == method]
            if not method_data.empty:
                avg_acc = method_data['accuracy'].mean()
                avg_time = method_data['total_time'].mean()
                avg_comp = method_data['compression_ratio'].mean()
                
                acc_pct = avg_acc * 100 if avg_acc < 1 else avg_acc
                comp_str = f"{avg_comp:.2f}x" if avg_comp > 0 else "N/A"
                
                print(f"   {method.upper()}: {acc_pct:.1f}% accuracy, {avg_time:.1f}s, {comp_str}")
        
        # Dimension scaling preview
        dimension_data = completed_data[completed_data['experiment_type'] == 'dimension_scaling']
        if not dimension_data.empty:
            print(f"\n📈 Dimension Scaling Preview:")
            for dim in sorted(dimension_data['dimension'].unique()):
                dim_data = dimension_data[dimension_data['dimension'] == dim]
                methods = []
                for method in dim_data['method'].unique():
                    method_data = dim_data[dim_data['method'] == method]
                    if not method_data.empty:
                        acc = method_data['accuracy'].mean()
                        acc_pct = acc * 100 if acc < 1 else acc
                        methods.append(f"{method}={acc_pct:.1f}%")
                print(f"   d={dim}: {', '.join(methods)}")
        
        # Efficiency preview
        efficiency_data = completed_data[completed_data['experiment_type'] == 'computational_efficiency']
        if not efficiency_data.empty:
            print(f"\n⚡ Efficiency Preview:")
            for method in efficiency_data['method'].unique():
                method_data = efficiency_data[efficiency_data['method'] == method]
                if not method_data.empty:
                    avg_time = method_data['total_time'].mean()
                    avg_cluster_time = method_data['clustering_time'].mean()
                    cluster_time_str = f", clustering={avg_cluster_time:.1f}s" if avg_cluster_time > 0 else ""
                    print(f"   {method.upper()}: total={avg_time:.1f}s{cluster_time_str}")
        
        # LAMG specific insights
        lamg_data = completed_data[completed_data['method'] == 'lamg']
        if not lamg_data.empty:
            print(f"\n🔬 LAMG Specific Insights:")
            print(f"   LAMG tests completed: {len(lamg_data)}")
            
            # Check timing extraction
            lamg_with_time = lamg_data[lamg_data['total_time'] > 0]
            print(f"   LAMG with valid timing: {len(lamg_with_time)}")
            
            # Check clustering extraction  
            lamg_with_clusters = lamg_data[lamg_data['final_clusters'] > 0]
            print(f"   LAMG with cluster data: {len(lamg_with_clusters)}")
            
            if not lamg_with_clusters.empty:
                avg_compression = lamg_with_clusters['compression_ratio'].mean()
                print(f"   LAMG avg compression: {avg_compression:.2f}x")
        
    except Exception as e:
        print(f"❌ Error generating insights: {e}")

def main():
    """Main validation function"""
    if len(sys.argv) != 2:
        print("Usage: python validate_sample_results.py <sample_csv_path>")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    if not Path(csv_path).exists():
        print(f"❌ Sample CSV file not found: {csv_path}")
        sys.exit(1)
    
    print("🧪 KOUTIS SAMPLE VALIDATION ANALYSIS")
    print("=" * 40)
    print(f"📁 Analyzing sample results: {csv_path}")
    
    # Run all validations
    csv_valid = validate_sample_csv(csv_path)
    requirements_met = check_koutis_requirements(csv_path)
    generate_sample_insights(csv_path)
    
    print(f"\n🎯 FINAL VALIDATION RESULT:")
    print("=" * 30)
    
    if csv_valid and requirements_met:
        print("✅ SAMPLE VALIDATION SUCCESSFUL!")
        print("🚀 Ready to run full Koutis efficiency study")
        print("")
        print("📋 Next Steps:")
        print("1. Review sample results above")
        print("2. Fix any issues if needed")
        print("3. Run full test suite: ./koutis_test_suite.sh") 
        print("4. Scale up experiment parameters as needed")
    else:
        print("❌ SAMPLE VALIDATION FAILED!")
        print("🔍 Issues found:")
        if not csv_valid:
            print("   - CSV output problems")
        if not requirements_met:
            print("   - Missing Koutis requirements")
        print("")
        print("🛠️  Fix Issues:")
        print("1. Check GraphZoom integration")
        print("2. Verify MATLAB MCR setup for LAMG")
        print("3. Review log files for errors")
        print("4. Test individual methods manually")

if __name__ == "__main__":
    main()
