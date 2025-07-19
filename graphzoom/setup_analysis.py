#!/usr/bin/env python3
"""
Analysis Setup and Quick Start Script
====================================

This script helps set up the spectral analysis environment and provides
quick commands to run the analysis with your specific setup.
"""

import os
import sys
from pathlib import Path
import argparse

def check_file_structure():
    """Check and report the current file structure"""
    
    print("📁 CURRENT FILE STRUCTURE:")
    print("=" * 30)
    
    # Expected directories and files
    expected_structure = {
        "./": ["filtered.py", "estimate_k_and_clusters.py", "core.py", "torch_interface.py"],
        "./dataset/cora/": ["cora.json", "cora-feats.npy"],
        "./reduction_results/": ["*.mtx"],
        "./cmg_extracted_graphs/": ["*_laplacian.mtx", "*_metadata.json"],
    }
    
    missing_items = []
    found_items = []
    
    for directory, files in expected_structure.items():
        dir_path = Path(directory)
        
        print(f"\n📂 {directory}:")
        
        if not dir_path.exists():
            print(f"   ❌ Directory does not exist")
            missing_items.append(directory)
            continue
        
        for file_pattern in files:
            if '*' in file_pattern:
                # Handle wildcards
                matching_files = list(dir_path.glob(file_pattern))
                if matching_files:
                    print(f"   ✅ Found {len(matching_files)} files matching {file_pattern}")
                    found_items.extend(matching_files)
                else:
                    print(f"   ❌ No files matching {file_pattern}")
                    missing_items.append(f"{directory}{file_pattern}")
            else:
                file_path = dir_path / file_pattern
                if file_path.exists():
                    print(f"   ✅ {file_pattern}")
                    found_items.append(file_path)
                else:
                    print(f"   ❌ {file_pattern}")
                    missing_items.append(str(file_path))
    
    return found_items, missing_items

def create_directory_structure():
    """Create necessary directories"""
    
    directories = [
        "./cmg_extracted_graphs",
        "./analysis_output", 
        "./dataset/cora"
    ]
    
    print("\n📁 Creating directory structure...")
    
    for directory in directories:
        dir_path = Path(directory)
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"   ✅ Created: {directory}")
        else:
            print(f"   ✅ Exists: {directory}")

def generate_quick_commands():
    """Generate quick command examples"""
    
    print("\n🚀 QUICK START COMMANDS:")
    print("=" * 25)
    
    commands = [
        ("1. Extract CMG++ graphs:", 
         "python cmg_graph_extractor.py"),
        
        ("2. Run complete analysis:", 
         "python run_spectral_comparison.py"),
        
        ("3. Extract only (if files missing):", 
         "python run_spectral_comparison.py --extract-only"),
        
        ("4. Analyze existing graphs only:", 
         "python run_spectral_comparison.py --analyze-only"),
        
        ("5. Extract custom CMG++ graph:", 
         "python cmg_graph_extractor.py --k 10 --d 20 --name custom_test"),
        
        ("6. Check dependencies:", 
         "python -c \"from run_spectral_comparison import check_dependencies; check_dependencies()\""),
    ]
    
    for description, command in commands:
        print(f"\n{description}")
        print(f"   {command}")

def check_python_packages():
    """Check required Python packages"""
    
    print("\n📦 CHECKING PYTHON PACKAGES:")
    print("=" * 30)
    
    required_packages = [
        "numpy", "scipy", "matplotlib", "pandas", 
        "networkx", "torch", "sklearn", "seaborn"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n💡 Install missing packages:")
        print(f"   pip install {' '.join(missing_packages)}")
    
    return missing_packages

def run_quick_test():
    """Run a quick test to verify everything works"""
    
    print("\n🧪 RUNNING QUICK TEST:")
    print("=" * 20)
    
    try:
        # Test imports
        print("Testing imports...")
        import numpy as np
        import scipy.sparse as sp
        import matplotlib.pyplot as plt
        print("   ✅ Core packages imported")
        
        # Test CMG modules
        try:
            from filtered import cmg_filtered_clustering
            from estimate_k_and_clusters import estimate_k
            print("   ✅ CMG modules imported")
        except ImportError as e:
            print(f"   ❌ CMG modules: {e}")
            return False
        
        # Test dataset loading
        dataset_path = Path("./dataset/cora/cora.json")
        if dataset_path.exists():
            print("   ✅ Dataset file found")
        else:
            print("   ❌ Dataset file missing")
            return False
        
        # Test LAMG files
        lamg_path = Path("./reduction_results")
        mtx_files = list(lamg_path.glob("*.mtx")) if lamg_path.exists() else []
        if mtx_files:
            print(f"   ✅ Found {len(mtx_files)} LAMG files")
        else:
            print("   ⚠️  No LAMG files found (will extract CMG++ only)")
        
        print("\n✅ Quick test passed! Ready to run analysis.")
        return True
        
    except Exception as e:
        print(f"\n❌ Quick test failed: {e}")
        return False

def show_analysis_workflow():
    """Show the complete analysis workflow"""
    
    print("\n🔄 COMPLETE ANALYSIS WORKFLOW:")
    print("=" * 35)
    
    workflow_steps = [
        "1. 📂 Check file structure and dependencies",
        "2. 🔄 Extract CMG++ graphs from your pipeline", 
        "3. 📁 Load LAMG matrices from reduction_results/",
        "4. 🔍 Compute spectral properties (eigenvalues, Fiedler values)",
        "5. 📊 Analyze connectivity vs accuracy correlations",
        "6. 📈 Generate comparison visualizations",
        "7. 📝 Create comprehensive analysis report",
        "8. 🎯 Provide definitive evidence for accuracy differences"
    ]
    
    for step in workflow_steps:
        print(f"   {step}")
    
    print(f"\n📋 EXPECTED OUTPUTS:")
    print(f"   📊 spectral_analysis_results.csv - Complete data table")
    print(f"   📈 eigenvalue_spectra_comparison.png - Eigenvalue plots")
    print(f"   📈 connectivity_vs_accuracy.png - Correlation analysis")
    print(f"   📈 method_comparison_bars.png - Bar chart comparisons")
    print(f"   📝 spectral_analysis_report.txt - Full text report")

def customize_accuracy_data():
    """Help user customize accuracy data"""
    
    print("\n⚙️  CUSTOMIZING ACCURACY DATA:")
    print("=" * 30)
    
    print("Current default accuracy mappings:")
    default_accuracy = {
        'reduce_ratio_3': 79.5,
        'level_2': 74.8,
        'k10_d10': 74.8,
        'k10_d20': 75.2,
    }
    
    for key, accuracy in default_accuracy.items():
        print(f"   {key}: {accuracy}%")
    
    print(f"\n💡 To customize accuracy data:")
    print(f"   1. Edit the 'accuracy_data' dictionary in run_spectral_comparison.py")
    print(f"   2. Map your method names to actual accuracy values")
    print(f"   3. Use lowercase keys that match parts of your method names")
    
    print(f"\n📝 Example custom mapping:")
    print(f"   accuracy_data = {{")
    print(f"       'lamg_reduce_3': 79.5,    # Your LAMG result")
    print(f"       'cmg_level_2': 74.8,      # Your CMG++ result")
    print(f"       'custom_method': 76.2,    # Additional methods")
    print(f"   }}")

def main():
    """Main setup function"""
    
    print("🔬 SPECTRAL ANALYSIS SETUP")
    print("=" * 26)
    print("Setting up LAMG vs CMG++ comparison analysis")
    
    # Check file structure
    found_items, missing_items = check_file_structure()
    
    # Create directories
    create_directory_structure()
    
    # Check packages
    missing_packages = check_python_packages()
    
    # Show workflow
    show_analysis_workflow()
    
    # Generate commands
    generate_quick_commands()
    
    # Customize accuracy
    customize_accuracy_data()
    
    # Summary
    print(f"\n📋 SETUP SUMMARY:")
    print(f"=" * 15)
    print(f"   Found {len(found_items)} required files")
    print(f"   Missing {len(missing_items)} items")
    print(f"   Missing {len(missing_packages)} Python packages")
    
    if missing_items:
        print(f"\n⚠️  MISSING ITEMS:")
        for item in missing_items[:5]:  # Show first 5
            print(f"   ❌ {item}")
        if len(missing_items) > 5:
            print(f"   ... and {len(missing_items) - 5} more")
    
    if missing_packages:
        print(f"\n💡 INSTALL PACKAGES:")
        print(f"   pip install {' '.join(missing_packages)}")
    
    # Run quick test if requested
    if len(missing_items) == 0 and len(missing_packages) == 0:
        print(f"\n🧪 Running quick test...")
        if run_quick_test():
            print(f"\n🎉 READY TO RUN ANALYSIS!")
            print(f"   python run_spectral_comparison.py")
        else:
            print(f"\n⚠️  Please resolve test failures before running analysis")
    else:
        print(f"\n⚠️  Please resolve missing dependencies before running analysis")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Setup spectral analysis environment")
    parser.add_argument("--test-only", action="store_true", help="Run quick test only")
    parser.add_argument("--check-only", action="store_true", help="Check structure only")
    
    args = parser.parse_args()
    
    if args.test_only:
        run_quick_test()
    elif args.check_only:
        check_file_structure()
        check_python_packages()
    else:
        main()
