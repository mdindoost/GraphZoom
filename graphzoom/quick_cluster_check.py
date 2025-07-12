#!/usr/bin/env python3
"""
Quick script to manually check cluster info from a single run
"""

import subprocess
import re
import sys

def run_single_test_and_extract():
    """Run one test and extract detailed cluster information"""
    
    print("🔍 Running single test to check cluster extraction...")
    
    # Run a single CMG test
    cmd = ["python", "graphzoom.py", "--dataset", "cora", "--coarse", "cmg", 
           "--level", "1", "--cmg_k", "10", "--cmg_d", "20", "--num_neighs", "2", 
           "--embed_method", "deepwalk"]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            output = result.stdout
            
            print("📊 EXTRACTING CLUSTER INFORMATION:")
            print("-" * 40)
            
            # Look for various cluster indicators
            patterns = [
                (r'CMG found (\d+) clusters', 'CMG clusters found'),
                (r'Coarsened to (\d+) nodes', 'Coarsened nodes'),
                (r'Final graph: (\d+) nodes', 'Final graph nodes'),
                (r'Num of nodes: (\d+)', 'Coarsening nodes'),
                (r'nc = (\d+)', 'nc variable'),
                (r'(\d+) clusters', 'Generic clusters')
            ]
            
            found_info = False
            for pattern, description in patterns:
                matches = re.findall(pattern, output)
                if matches:
                    print(f"✅ {description}: {matches}")
                    found_info = True
            
            if not found_info:
                print("❌ No cluster information found in output")
                print("\n📋 Full output for debugging:")
                print("-" * 30)
                print(output[-1000:])  # Last 1000 chars
            
            # Extract accuracy and time for comparison
            acc_match = re.search(r'Test Accuracy:\s+([\d.]+)', output)
            time_match = re.search(r'Total Time.*?= ([\d.]+)', output)
            
            if acc_match and time_match:
                print(f"\n📈 Performance: Accuracy={acc_match.group(1)}, Time={time_match.group(1)}s")
            
            # Calculate compression if we found clusters
            cora_nodes = 2708
            cluster_patterns = [r'CMG found (\d+) clusters', r'Final graph: (\d+) nodes', r'Coarsened to (\d+) nodes']
            
            for pattern in cluster_patterns:
                match = re.search(pattern, output)
                if match:
                    clusters = int(match.group(1))
                    compression = cora_nodes / clusters
                    print(f"🎯 Compression: {cora_nodes} → {clusters} = {compression:.2f}x")
                    break
            
            return True
            
        else:
            print(f"❌ Command failed with return code: {result.returncode}")
            print(f"Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Command timed out")
        return False
    except Exception as e:
        print(f"❌ Error running command: {e}")
        return False

def check_log_file():
    """Check if there's a recent log file we can analyze"""
    import os
    
    log_files = ['temp_log.txt']
    for log_file in log_files:
        if os.path.exists(log_file):
            print(f"\n📁 Checking existing log file: {log_file}")
            print("-" * 30)
            
            with open(log_file, 'r') as f:
                content = f.read()
            
            # Look for cluster info
            patterns = [
                r'CMG found (\d+) clusters',
                r'Coarsened to (\d+) nodes',
                r'Final graph: (\d+) nodes',
                r'Num of nodes: (\d+)'
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, content)
                if matches:
                    print(f"Found pattern '{pattern}': {matches}")
            
            return True
    
    return False

if __name__ == "__main__":
    print("🔧 CLUSTER EXTRACTION DEBUG TOOL")
    print("=" * 35)
    
    # First check if there's an existing log
    if not check_log_file():
        print("No existing log files found.")
    
    # Run a fresh test
    print("\n🚀 Running fresh test...")
    success = run_single_test_and_extract()
    
    if success:
        print("\n✅ Test completed successfully!")
    else:
        print("\n❌ Test failed - check your GraphZoom setup")
