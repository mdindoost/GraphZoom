#!/usr/bin/env python3
"""
Fix the missing seed parameter in graphzoom.py
"""

def fix_seed():
    with open('graphzoom.py', 'r') as f:
        content = f.read()
    
    # Check if seed parameter exists in parser
    if 'add_argument("--seed"' not in content:
        # Find a safe place to add it (after the last add_argument)
        last_arg = 'parser.add_argument("-w", "--sage_weighted"'
        if last_arg in content:
            seed_arg = '''parser.add_argument("--seed", type=int, default=42, \\
            help="Random seed for reproducibility")
    '''
            content = content.replace(last_arg, last_arg + '\n    ' + seed_arg)
    
    # Also remove the seed usage if it exists but parameter doesn't
    if 'np.random.seed(args.seed)' in content and 'add_argument("--seed"' not in content:
        # Comment out the seed usage for now
        content = content.replace(
            'np.random.seed(args.seed)',
            '# np.random.seed(args.seed)  # Seed parameter not available'
        )
        content = content.replace(
            'random.seed(args.seed)',
            '# random.seed(args.seed)  # Seed parameter not available'
        )
    
    with open('graphzoom.py', 'w') as f:
        f.write(content)
    
    print("✅ Fixed seed parameter issue")

def test_basic_run():
    import subprocess
    
    print("🧪 Testing basic CMG run...")
    try:
        result = subprocess.run([
            'python', 'graphzoom.py', '--help'
        ], capture_output=True, text=True, timeout=10)
        
        if '--cmg_k' in result.stdout:
            print("✅ CMG parameters found in help")
        else:
            print("❌ CMG parameters NOT found in help")
            
        if '--seed' in result.stdout:
            print("✅ Seed parameter found in help")
        else:
            print("❌ Seed parameter NOT found in help")
            
    except Exception as e:
        print(f"❌ Error testing: {e}")

def main():
    print("🔧 FIXING SEED PARAMETER")
    print("=" * 30)
    
    fix_seed()
    test_basic_run()
    
    print("\n🧪 Now test with:")
    print("python graphzoom.py --dataset cora --coarse cmg --embed_method deepwalk --cmg_k 10 --cmg_d 20 --cmg_threshold 0.1")

if __name__ == "__main__":
    main()
