#!/usr/bin/env python3
"""
Debug script to check LAMG output format and fix cluster extraction
"""

import subprocess
import re
import sys

def debug_lamg_output():
    """Run LAMG and examine its output format"""
    print("🔍 Debugging LAMG output format...")
    
    # Run a quick LAMG test
    cmd = ["python", "graphzoom.py", "--dataset", "cora", "--coarse", "lamg", 
           "--reduce_ratio", "2", "--search_ratio", "12", 
           "--mcr_dir", "/home/mohammad/matlab/R2018a", "--embed_method", "deepwalk"]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            output = result.stdout
            
            print("📊 SEARCHING FOR LAMG CLUSTER PATTERNS:")
            print("-" * 50)
            
            # Look for various LAMG-specific patterns
            lamg_patterns = [
                (r'Gs\.mtx.*?(\d+)', 'Gs.mtx pattern'),
                (r'coarse.*?(\d+).*?nodes', 'Coarse nodes pattern'),
                (r'reduced.*?(\d+)', 'Reduced pattern'),
                (r'NumLevels.*?(\d+)', 'NumLevels pattern'),
                (r'Graph.*?(\d+).*?nodes', 'Graph nodes pattern'),
                (r'level.*?(\d+)', 'Level pattern'),
                (r'Mapping.*?(\d+)', 'Mapping pattern'),
                (r'(\d+)\s+nodes', 'Generic nodes pattern'),
            ]
            
            found_patterns = []
            for pattern, description in lamg_patterns:
                matches = re.findall(pattern, output, re.IGNORECASE)
                if matches:
                    print(f"✅ {description}: {matches}")
                    found_patterns.append((pattern, matches))
                else:
                    print(f"❌ {description}: No matches")
            
            if not found_patterns:
                print("\n❌ No LAMG cluster patterns found!")
                print("\n📋 FULL LAMG OUTPUT FOR MANUAL INSPECTION:")
                print("-" * 50)
                print(output)
            else:
                print(f"\n✅ Found {len(found_patterns)} potential patterns")
            
            # Look for file-based information
            print(f"\n📁 CHECKING FOR LAMG OUTPUT FILES:")
            print("-" * 35)
            
            import os
            lamg_files = [
                'reduction_results/Gs.mtx',
                'reduction_results/NumLevels.txt', 
                'reduction_results/Mapping.mtx',
                'reduction_results/CPUtime.txt'
            ]
            
            for file_path in lamg_files:
                if os.path.exists(file_path):
                    print(f"✅ Found: {file_path}")
                    try:
                        with open(file_path, 'r') as f:
                            content = f.read()[:200]  # First 200 chars
                            print(f"   Content preview: {content.strip()}")
                    except:
                        print(f"   Could not read file")
                else:
                    print(f"❌ Missing: {file_path}")
            
        else:
            print(f"❌ LAMG test failed with return code: {result.returncode}")
            print(f"Error output: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        print("❌ LAMG test timed out")
    except Exception as e:
        print(f"❌ Error running LAMG test: {e}")

if __name__ == "__main__":
    debug_lamg_output()
