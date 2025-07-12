#!/usr/bin/env python3
"""
Quick fix script to add CMG parameters and basic timing to graphzoom.py
"""

def fix_graphzoom():
    """Add CMG parameters and basic timing to graphzoom.py"""
    
    with open('graphzoom.py', 'r') as f:
        content = f.read()
    
    # 1. Add CMG parameters to argument parser
    if '--cmg_k' not in content:
        # Find the argument parser section and add CMG parameters
        cmg_args = '''parser.add_argument("--cmg_k", type=int, default=10, \\
        help="CMG filter order (only for CMG coarsening)")
parser.add_argument("--cmg_d", type=int, default=20, \\
        help="CMG embedding dimension (only for CMG coarsening)")  
parser.add_argument("--cmg_threshold", type=float, default=0.1, \\
        help="CMG cosine similarity threshold (only for CMG coarsening)")
parser.add_argument("--seed", type=int, default=42, \\
        help="Random seed for reproducibility")
'''
        
        # Add after the existing arguments
        insert_point = 'parser.add_argument("-g", "--sage_model"'
        if insert_point in content:
            content = content.replace(insert_point, cmg_args + '    ' + insert_point)
    
    # 2. Add CMG case to the coarsening section
    if 'elif args.coarse == "cmg":' not in content:
        cmg_case = '''
    elif args.coarse == "cmg":
        G, projections, laplacians, level = cmg_coarse(
            laplacian, args.level, args.cmg_k, args.cmg_d, args.cmg_threshold
        )
        reduce_time = time.process_time() - reduce_start
'''
        
        # Add after the LAMG case
        lamg_case_end = 'projections, laplacians = construct_proj_laplacian(laplacian, level, reduce_results)'
        if lamg_case_end in content:
            content = content.replace(
                lamg_case_end,
                lamg_case_end + cmg_case
            )
    
    # 3. Add basic timing import
    if 'import time' not in content:
        content = content.replace('import sys', 'import sys\nimport time')
    
    # 4. Add CMG import
    if 'from cmg_coarsening import' not in content:
        content = content.replace(
            'from scoring import lr',
            'from scoring import lr\nfrom cmg_coarsening import cmg_coarse'
        )
    
    # 5. Add seed setting
    if 'np.random.seed' not in content:
        seed_setting = '''
    # Set random seed for reproducibility
    np.random.seed(args.seed)
    import random
    random.seed(args.seed)
'''
        # Add after args parsing
        content = content.replace(
            'args = parser.parse_args()',
            'args = parser.parse_args()' + seed_setting
        )
    
    # 6. Add basic timing prints (simple version)
    timing_replacements = [
        ('print("%%%%%% Starting Graph Reduction %%%%%%")',
         'print("%%%%%% Starting Graph Reduction %%%%%%")\n    print(f"[TIMING] Original graph: {laplacian.shape[0]} nodes")'),
        
        ('print("%%%%%% Starting Graph Embedding %%%%%%")',
         'print("%%%%%% Starting Graph Embedding %%%%%%")\n    print(f"[TIMING] Coarsened graph: {len(G.nodes())} nodes, {len(G.edges())} edges")'),
    ]
    
    for old, new in timing_replacements:
        content = content.replace(old, new)
    
    # Write the fixed file
    with open('graphzoom.py', 'w') as f:
        f.write(content)
    
    print("✅ Fixed graphzoom.py with CMG parameters and basic timing")

def main():
    print("🔧 QUICK FIX: Adding CMG parameters to GraphZoom")
    print("="*50)
    
    fix_graphzoom()
    
    print("\n✅ FIXED! Now test with:")
    print("python graphzoom.py --dataset cora --coarse cmg --embed_method deepwalk --cmg_k 10 --cmg_d 20")
    
    print("\n📝 Added parameters:")
    print("  --cmg_k (filter order)")
    print("  --cmg_d (embedding dimension)")  
    print("  --cmg_threshold (cosine similarity threshold)")
    print("  --seed (random seed)")

if __name__ == "__main__":
    main()
