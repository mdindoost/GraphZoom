#!/bin/bash
# Quick script to check what arguments GraphZoom supports

echo "🔍 CHECKING GRAPHZOOM ARGUMENTS"
echo "==============================="

if [ ! -f "graphzoom.py" ]; then
    echo "❌ graphzoom.py not found. Please run from GraphZoom directory."
    exit 1
fi

echo "📋 GraphZoom help output:"
echo "-------------------------"
python graphzoom.py --help

echo ""
echo "🔍 Checking specific arguments we need:"
echo "---------------------------------------"

# Test CMG arguments
echo -n "CMG support: "
if python graphzoom.py --help 2>&1 | grep -q "cmg"; then
    echo "✅ Found"
else
    echo "❌ Not found"
fi

# Test LAMG arguments  
echo -n "LAMG support: "
if python graphzoom.py --help 2>&1 | grep -q "lamg"; then
    echo "✅ Found"
else
    echo "❌ Not found"
fi

# Check for CMG-specific parameters
echo -n "CMG k parameter: "
if python graphzoom.py --help 2>&1 | grep -q "cmg_k"; then
    echo "✅ Found"
else
    echo "❌ Not found"
fi

echo -n "CMG d parameter: "
if python graphzoom.py --help 2>&1 | grep -q "cmg_d"; then
    echo "✅ Found"
else
    echo "❌ Not found"
fi

# Check coarsening methods
echo ""
echo "🔬 Available coarsening methods:"
python graphzoom.py --help 2>&1 | grep -A 10 -B 5 "coarse\|method" | head -20

echo ""
echo "📊 Testing a simple run to see output format:"
echo "---------------------------------------------"
echo "Running: python graphzoom.py --dataset cora --embed_method deepwalk"
echo ""

# Run a quick test to see output format
timeout 30 python graphzoom.py --dataset cora --embed_method deepwalk 2>&1 | head -20

echo ""
echo "✅ GraphZoom argument check complete!"
echo "💡 Use this information to update the test scripts with correct arguments."
