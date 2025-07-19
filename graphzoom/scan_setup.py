#!/usr/bin/env python3
"""
Setup Scanner - Check available files and directory structure
"""

import os
import json
from pathlib import Path

def scan_directory_structure():
    """Scan and report the current directory structure."""
    
    print("=== Directory Structure Scan ===\n")
    
    current_dir = Path(".")
    print(f"Current directory: {current_dir.absolute()}")
    print()
    
    # 1. Check for CMG Python files
    print("1. CMG Implementation Files:")
    cmg_files = ["filtered.py", "cmg_coarsening_timed.py", "graphzoom_timed.py", 
                 "utils.py", "core.py", "torch_interface.py"]
    
    found_cmg_files = []
    for file in cmg_files:
        if Path(file).exists():
            print(f"   ✅ {file}")
            found_cmg_files.append(file)
        else:
            print(f"   ❌ {file}")
    
    print(f"   Found {len(found_cmg_files)}/{len(cmg_files)} CMG files")
    print()
    
    # 2. Check for dataset files
    print("2. Dataset Files:")
    dataset_dir = Path("dataset/cora")
    if dataset_dir.exists():
        print(f"   ✅ Dataset directory: {dataset_dir}")
        
        for file in dataset_dir.iterdir():
            if file.is_file():
                size_mb = file.stat().st_size / (1024*1024)
                print(f"      📁 {file.name} ({size_mb:.1f} MB)")
        
        # Try to load the JSON file to see its structure
        json_files = ["cora-G.json", "cora.json"]
        graph_json = None
        
        for json_file in json_files:
            json_path = dataset_dir / json_file
            if json_path.exists():
                print(f"\n   📋 Graph JSON file structure preview:")
                try:
                    with open(json_path, 'r') as f:
                        data = json.load(f)
                    
                    print(f"      File: {json_file}")
                    print(f"      Keys: {list(data.keys())}")
                    
                    if 'nodes' in data:
                        print(f"      Nodes: {len(data['nodes'])}")
                    if 'links' in data:
                        print(f"      Links: {len(data['links'])}")
                    elif 'edges' in data:
                        print(f"      Edges: {len(data['edges'])}")
                    
                    graph_json = json_file
                    break
                        
                except Exception as e:
                    print(f"      ❌ Error reading JSON: {e}")
        
        if graph_json is None:
            print("   ⚠️  No graph JSON file found (need cora-G.json or cora.json)")
        
        # Also check class mapping file
        class_map_path = dataset_dir / "cora-class_map.json"
        if class_map_path.exists():
            print(f"\n   📋 Class mapping file found: cora-class_map.json")
    else:
        print(f"   ❌ Dataset directory not found: {dataset_dir}")
    
    print()
    
    # 3. Check for LAMG results
    print("3. LAMG Results:")
    lamg_dirs = ["reduction_results", "lamg_results", "results", "output"]
    
    found_lamg_dir = None
    for dir_name in lamg_dirs:
        lamg_dir = Path(dir_name)
        if lamg_dir.exists():
            print(f"   ✅ Found results directory: {lamg_dir}")
            found_lamg_dir = lamg_dir
            
            # List MTX files
            mtx_files = list(lamg_dir.glob("**/*.mtx"))
            print(f"      MTX files found: {len(mtx_files)}")
            for mtx_file in mtx_files[:5]:  # Show first 5
                size_kb = mtx_file.stat().st_size / 1024
                print(f"         📄 {mtx_file.relative_to(lamg_dir)} ({size_kb:.1f} KB)")
            if len(mtx_files) > 5:
                print(f"         ... and {len(mtx_files) - 5} more")
            break
    
    if found_lamg_dir is None:
        print("   ❌ No LAMG results directory found")
        print(f"      Searched: {lamg_dirs}")
    
    print()
    
    # 4. Check Python environment
    print("4. Python Environment:")
    try:
        import torch
        print(f"   ✅ PyTorch: {torch.__version__}")
    except ImportError:
        print("   ❌ PyTorch not available")
    
    try:
        import torch_geometric
        print(f"   ✅ PyTorch Geometric: {torch_geometric.__version__}")
    except ImportError:
        print("   ❌ PyTorch Geometric not available")
    
    try:
        import scipy
        print(f"   ✅ SciPy: {scipy.__version__}")
    except ImportError:
        print("   ❌ SciPy not available")
    
    try:
        import networkx
        print(f"   ✅ NetworkX: {networkx.__version__}")
    except ImportError:
        print("   ❌ NetworkX not available")
    
    print()
    
    # 5. Test CMG imports
    print("5. CMG Module Import Test:")
    if "filtered.py" in found_cmg_files:
        try:
            from filtered import cmg_filtered_clustering
            print("   ✅ Successfully imported cmg_filtered_clustering")
        except ImportError as e:
            print(f"   ❌ Import error: {e}")
        except Exception as e:
            print(f"   ⚠️  Import warning: {e}")
    else:
        print("   ❌ filtered.py not found")
    
    print()
    
    # 6. Generate recommendations
    print("6. Recommendations:")
    
    if len(found_cmg_files) < len(cmg_files):
        missing = set(cmg_files) - set(found_cmg_files)
        print(f"   📋 Missing CMG files: {missing}")
        print("      → Please copy these files to the current directory")
    
    if not dataset_dir.exists():
        print("   📋 Dataset setup needed:")
        print("      → Ensure dataset/cora/ directory exists")
        print("      → Copy cora-G.json and cora-feats.npy to dataset/cora/")
    
    if found_lamg_dir is None:
        print("   📋 LAMG results needed:")
        print("      → Run LAMG coarsening to generate .mtx files")
        print("      → Or point to existing LAMG results directory")
    
    print("\n=== Scan Complete ===")
    
    return {
        'cmg_files': found_cmg_files,
        'dataset_available': dataset_dir.exists(),
        'lamg_results': found_lamg_dir,
        'ready_for_analysis': len(found_cmg_files) > 0 and dataset_dir.exists()
    }

def suggest_next_steps(scan_results):
    """Suggest next steps based on scan results."""
    
    print("\n=== Next Steps ===")
    
    if scan_results['ready_for_analysis']:
        print("🎉 Ready for analysis!")
        print("   1. Run: python cmg_adapter.py")
        print("   2. Run: python matrix_extractor.py") 
        print("   3. Run: python spectral_analyzer.py")
    else:
        print("⚠️  Setup needed before analysis:")
        
        if not scan_results['cmg_files']:
            print("   1. Copy CMG implementation files to current directory")
        
        if not scan_results['dataset_available']:
            print("   2. Set up dataset directory with Cora files")
        
        if scan_results['lamg_results'] is None:
            print("   3. Generate LAMG results or specify results directory")
        
        print("   4. Then run the analysis scripts")

if __name__ == "__main__":
    scan_results = scan_directory_structure()
    suggest_next_steps(scan_results)
