#!/usr/bin/env python3
"""
Quick test script to verify node count extraction on existing logs
"""
import os
import re
import glob

def test_extraction(log_file):
    """Test node count extraction on a single log file"""
    filename = os.path.basename(log_file).replace('.log', '')
    parts = filename.split('_')
    method_type = parts[0] if parts else "unknown"
    config_name = f"{parts[0]}_{parts[1]}" if len(parts) >= 2 else parts[0]
    
    with open(log_file, 'r') as f:
        content = f.read()
    
    coarsening_info = ""
    
    if "cmg" in config_name.lower():
        # CMG pattern: "Final graph: 254 nodes"
        nodes_match = re.search(r'Final graph:\s+(\d+)\s+nodes', content)
        if nodes_match:
            coarsening_info = f"Final: {nodes_match.group(1)} nodes"
            
    elif "lamg" in config_name.lower():
        # LAMG pattern: "531 train nodes" 
        nodes_match = re.search(r'(\d+)\s+train\s+nodes', content)
        if nodes_match:
            coarsening_info = f"Final: {nodes_match.group(1)} nodes"
            
    elif "simple" in config_name.lower():
        # Simple pattern: Find LAST occurrence of "Num of nodes: XXXX"
        nodes_matches = re.findall(r'Num of nodes:\s+(\d+)', content)
        if nodes_matches:
            # Show progression for debugging
            progression = " -> ".join(nodes_matches)
            final_nodes = nodes_matches[-1]
            coarsening_info = f"Final: {final_nodes} nodes"
            return f"{config_name:<12} {coarsening_info:<20} (Progression: {progression})"
    
    return f"{config_name:<12} {coarsening_info if coarsening_info else 'NOT FOUND':<20}"

def main():
    logs_dir = "."
    log_files = sorted(glob.glob("*.log"))
    
    print("Node Count Extraction Test")
    print("=" * 60)
    print(f"{'Config':<12} {'Result':<20} {'Details'}")
    print("-" * 60)
    
    for log_file in log_files:
        result = test_extraction(log_file)
        print(result)

if __name__ == "__main__":
    main()
