import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def load_results():
    """Load all spectral analysis results"""
    results = {}
    
    for file in glob.glob("spectral_results_*.txt"):
        method_name = file.replace("spectral_results_", "").replace(".txt", "")
        
        data = {}
        with open(file, 'r') as f:
            for line in f:
                if ':' in line:
                    key, value = line.strip().split(':', 1)
                    if key in ['Nodes', 'Edges']:
                        data[key] = int(value)
                    elif key in ['Fiedler_value', 'Spectral_gap']:
                        data[key] = float(value)
                    elif key == 'Eigenvalues':
                        try:
                            data[key] = [float(x) for x in value.split(',')]
                        except:
                            data[key] = []
                    else:
                        data[key] = value.strip()
        
        results[method_name] = data
    
    return results

def generate_report(results):
    """Generate spectral comparison report"""
    print("🔬 SPECTRAL ANALYSIS REPORT")
    print("=" * 50)
    
    if not results:
        print("❌ No results found")
        return
    
    # Create DataFrame for easy analysis
    df_data = []
    for method, data in results.items():
        df_data.append({
            'Method': method,
            'Nodes': data.get('Nodes', 0),
            'Edges': data.get('Edges', 0),
            'Fiedler_Value': data.get('Fiedler_value', 0),
            'Spectral_Gap': data.get('Spectral_gap', 0),
            'Compression': 2708 / data.get('Nodes', 1) if data.get('Nodes', 0) > 0 else 0
        })
    
    df = pd.DataFrame(df_data)
    df = df.sort_values('Nodes', ascending=False)
    
    print(f"\n{'Method':<15} {'Nodes':<8} {'Edges':<8} {'Compression':<12} {'Fiedler':<12} {'Spectral Gap':<12}")
    print("-" * 80)
    
    for _, row in df.iterrows():
        print(f"{row['Method']:<15} {row['Nodes']:<8} {row['Edges']:<8} {row['Compression']:<12.2f}x {row['Fiedler_Value']:<12.6f} {row['Spectral_Gap']:<12.6f}")
    
    # Key insights
    print("\n🎯 KEY SPECTRAL INSIGHTS:")
    print("=" * 50)
    
    # Find best spectral preservation
    best_fiedler = df.loc[df['Fiedler_Value'].idxmax()]
    best_gap = df.loc[df['Spectral_Gap'].idxmax()]
    
    print(f"✅ Best Fiedler value preservation: {best_fiedler['Method']} ({best_fiedler['Fiedler_Value']:.6f})")
    print(f"✅ Best spectral gap preservation: {best_gap['Method']} ({best_gap['Spectral_Gap']:.6f})")
    
    # Compare LAMG methods
    lamg_methods = df[df['Method'].str.contains('lamg', case=False, na=False)]
    if len(lamg_methods) > 1:
        print(f"\n📊 LAMG COMPRESSION vs SPECTRAL QUALITY:")
        print("=" * 50)
        for _, row in lamg_methods.iterrows():
            print(f"{row['Method']}: {row['Compression']:.1f}x compression, λ₂={row['Fiedler_Value']:.6f}")
        
        # Check if there's a trade-off
        fiedler_trend = lamg_methods['Fiedler_Value'].corr(lamg_methods['Compression'])
        print(f"\n💡 Fiedler-Compression correlation: {fiedler_trend:.3f}")
        if fiedler_trend < -0.5:
            print("   Strong negative correlation: Higher compression → Lower connectivity")
        elif fiedler_trend > 0.5:
            print("   Strong positive correlation: Higher compression → Higher connectivity (unusual)")
        else:
            print("   Weak correlation: LAMG maintains connectivity across compression levels")
    
    return df

if __name__ == "__main__":
    results = load_results()
    
    if results:
        df = generate_report(results)
        
        # Save summary
        if not df.empty:
            df.to_csv('spectral_summary.csv', index=False)
            print(f"\n📁 Summary saved to: spectral_summary.csv")
    else:
        print("❌ No spectral analysis results found")
