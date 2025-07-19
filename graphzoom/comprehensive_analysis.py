#!/usr/bin/env python3
"""
Comprehensive Multi-Level Analysis: LAMG vs CMG++
- CMG++: levels 1, 2, 3 (multiple runs)
- LAMG: reduce_ratio 2, 3, 4, 5, 6
- Eigenvalue analysis (up to 12 eigenvalues)
- NetworkX connectivity verification
"""

import os
import subprocess
from pathlib import Path
import scipy.sparse as sp
import numpy as np
import pickle
import json
import time

class ComprehensiveAnalyzer:
    def __init__(self, output_dir="comprehensive_analysis"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def load_lamg_mtx_custom(self, mtx_file):
        with open(mtx_file, 'r') as f:
            lines = f.readlines()

        n_rows, n_cols = map(int, lines[0].strip().split())
        rows, cols, vals = [], [], []

        for line in lines[1:]:
            parts = line.strip().split()
            if len(parts) >= 3:
                row, col, val = int(parts[0])-1, int(parts[1])-1, float(parts[2])
                rows.append(row)
                cols.append(col)
                vals.append(val)

        matrix = sp.coo_matrix((vals, (rows, cols)), shape=(n_rows, n_cols))
        return matrix.tocsr()

    def run_lamg_multiratio(self, reduce_ratios=[3, 4, 5, 6]):
        print("\n🚀 Running LAMG with Multiple Ratios using Shell Environment")
        matlab_mcr_root = "/home/mohammad/matlab/R2018a"
        lamg_results = {}

        for ratio in reduce_ratios:
            print(f"\n📊 LAMG Reduce Ratio {ratio}:")
            try:
                # Clean previous results
                reduction_dir = Path("reduction_results")
                if reduction_dir.exists():
                    import shutil
                    shutil.rmtree(reduction_dir)
                    print("   🧹 Cleaned previous results")

                time.sleep(1)  # Filesystem sync

                cmd = f"python graphzoom_timed.py --dataset cora --coarse lamg --reduce_ratio {ratio} --mcr_dir {matlab_mcr_root} --embed_method deepwalk --seed 42"
                print(f"   🔧 Command: {cmd}")
                result = subprocess.run(cmd, shell=True)

                if result.returncode != 0:
                    print(f"   ❌ LAMG failed at ratio {ratio} (return code {result.returncode})")
                    continue

                gs_file = Path("reduction_results/Gs.mtx")
                if not gs_file.exists():
                    print(f"   ❌ Gs.mtx not found after run")
                    available = list(Path("reduction_results").glob("*.mtx"))
                    if available:
                        print(f"      Found other MTX files: {[f.name for f in available]}")
                    continue

                print(f"   ✅ Found Gs.mtx")
                L = self.load_lamg_mtx_custom(gs_file)
                print(f"   📐 Matrix loaded: {L.shape[0]} nodes, {L.nnz // 2} edges")

                with open(self.output_dir / f"lamg_ratio{ratio}_laplacian.pkl", 'wb') as f:
                    pickle.dump(L, f)

                lamg_results[ratio] = {
                    'nodes': L.shape[0],
                    'edges': L.nnz // 2
                }

            except Exception as e:
                print(f"   ❌ Exception at ratio {ratio}: {e}")

        print("\n✅ LAMG multi-ratio test complete")
        return lamg_results

if __name__ == "__main__":
    analyzer = ComprehensiveAnalyzer()
    analyzer.run_lamg_multiratio(reduce_ratios=[3, 4, 5, 6])
