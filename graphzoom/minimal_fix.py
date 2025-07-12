#!/usr/bin/env python3
"""
Minimal fix: Just add CMG parameters to graphzoom.py without complex timing
"""

def minimal_fix():
    """Add only the essential CMG parameters"""
    
    # Fix graphzoom.py
    with open('graphzoom.py', 'r') as f:
        content = f.read()
    
    # Add CMG parameters to argument parser (find a safe insertion point)
    if '--cmg_k' not in content:
        # Find the help argument and add before it
        help_arg = 'parser.add_argument("-f", "--fusion"'
        if help_arg in content:
            cmg_params = '''parser.add_argument("--cmg_k", type=int, default=10, help="CMG filter order")
parser.add_argument("--cmg_d", type=int, default=20, help="CMG embedding dimension")
parser.add_argument("--cmg_threshold", type=float, default=0.1, help="CMG threshold")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
    '''
            content = content.replace(help_arg, cmg_params + help_arg)
    
    # Add CMG import if not present
    if 'from cmg_coarsening import' not in content:
        content = content.replace(
            'from scoring import lr',
            'from scoring import lr\nfrom cmg_coarsening import cmg_coarse'
        )
    
    # Add CMG case in the coarsening section
    if 'elif args.coarse == "cmg":' not in content:
        # Find the else clause and add CMG case before it
        else_clause = '    else:\n        raise NotImplementedError'
        if else_clause in content:
            cmg_case = '''    elif args.coarse == "cmg":
        G, projections, laplacians, level = cmg_coarse(
            laplacian, args.level, args.cmg_k, args.cmg_d, args.cmg_threshold
        )
        reduce_time = time.process_time() - reduce_start

'''
            content = content.replace(else_clause, cmg_case + else_clause)
    
    # Add seed setting
    if 'np.random.seed(args.seed)' not in content:
        # Add after argument parsing
        args_line = 'args = parser.parse_args()'
        if args_line in content:
            seed_code = '''args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)
    import random
    random.seed(args.seed)'''
            content = content.replace(args_line, seed_code)
    
    with open('graphzoom.py', 'w') as f:
        f.write(content)
    
    print("✅ Minimal fix applied to graphzoom.py")

def test_syntax():
    """Test if all files have valid syntax"""
    import py_compile
    
    files_to_test = ['graphzoom.py', 'cmg_coarsening.py', 'filtered.py']
    
    for file_name in files_to_test:
        try:
            py_compile.compile(file_name, doraise=True)
            print(f"✅ {file_name} - syntax OK")
        except py_compile.PyCompileError as e:
            print(f"❌ {file_name} - syntax error: {e}")
            return False
    
    return True

def main():
    print("🔧 MINIMAL FIX: Adding only essential CMG parameters")
    print("="*60)
    
    minimal_fix()
    
    print("\n🔍 Testing syntax...")
    if test_syntax():
        print("\n✅ ALL FILES HAVE VALID SYNTAX!")
        print("\n🧪 Test with:")
        print("python graphzoom.py --dataset cora --coarse cmg --embed_method deepwalk --cmg_k 10 --cmg_d 20")
    else:
        print("\n❌ Syntax errors found - check the files manually")

if __name__ == "__main__":
    main()
