#!/bin/bash

# LAMG Matrix Generation Script
# Generates coarsened matrices for multiple reduction ratios

echo "🚀 LAMG MATRIX GENERATION PIPELINE"
echo "=" | tr '\n' '=' | head -c 50; echo

# Configuration
MATLAB_MCR_ROOT="/home/mohammad/matlab/R2018a"
DATASET="cora"
EMBED_METHOD="deepwalk"
SEED="42"
RATIOS=(2 3 4 5 6)

# Create output directory
OUTPUT_DIR="lamg_matrices"
mkdir -p $OUTPUT_DIR

# Function to run LAMG for a specific ratio
run_lamg_ratio() {
    local ratio=$1
    echo
    echo "📊 LAMG Reduce Ratio $ratio"
    echo "----------------------------------------"
    
    # Clean previous results
    rm -rf reduction_results/
    mkdir -p reduction_results/
    
    # Run GraphZoom with LAMG
    echo "Running: python graphzoom_timed.py --dataset $DATASET --coarse lamg --reduce_ratio $ratio --mcr_dir $MATLAB_MCR_ROOT --embed_method $EMBED_METHOD --seed $SEED"
    
    python graphzoom_timed.py \
        --dataset $DATASET \
        --coarse lamg \
        --reduce_ratio $ratio \
        --mcr_dir $MATLAB_MCR_ROOT \
        --embed_method $EMBED_METHOD \
        --seed $SEED
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "✅ LAMG ratio $ratio completed successfully"
        
        # Check for required files
        if [ -f "reduction_results/Gs.mtx" ]; then
            # Copy results to output directory
            cp reduction_results/Gs.mtx $OUTPUT_DIR/lamg_ratio${ratio}_Gs.mtx
            echo "   📄 Saved: lamg_ratio${ratio}_Gs.mtx"
            
            # Copy other files if they exist
            for file in Mapping.mtx Projection_1.mtx Projection_2.mtx; do
                if [ -f "reduction_results/$file" ]; then
                    cp "reduction_results/$file" "$OUTPUT_DIR/lamg_ratio${ratio}_$file"
                    echo "   📄 Saved: lamg_ratio${ratio}_$file"
                fi
            done
            
            # Extract info about the matrix
            local nodes=$(head -1 reduction_results/Gs.mtx | cut -d' ' -f1)
            local reduction_ratio=$(echo "scale=2; 2708 / $nodes" | bc -l)
            echo "   📊 Matrix: $nodes nodes (${reduction_ratio}x reduction)"
            
            # Log success
            echo "ratio_${ratio}:success:${nodes}:${reduction_ratio}" >> $OUTPUT_DIR/generation_log.txt
            
        else
            echo "❌ Gs.mtx not found for ratio $ratio"
            echo "ratio_${ratio}:failed:no_matrix" >> $OUTPUT_DIR/generation_log.txt
        fi
    else
        echo "❌ LAMG ratio $ratio failed (exit code: $exit_code)"
        echo "ratio_${ratio}:failed:exit_code_${exit_code}" >> $OUTPUT_DIR/generation_log.txt
    fi
    
    # Clean up
    rm -rf reduction_results/
}

# Main execution
echo "Configuration:"
echo "  MATLAB_MCR_ROOT: $MATLAB_MCR_ROOT"
echo "  Dataset: $DATASET"
echo "  Ratios: ${RATIOS[*]}"
echo "  Output: $OUTPUT_DIR/"
echo

# Initialize log
echo "# LAMG Generation Log - $(date)" > $OUTPUT_DIR/generation_log.txt

# Run LAMG for each ratio
for ratio in "${RATIOS[@]}"; do
    run_lamg_ratio $ratio
done

echo
echo "📊 LAMG GENERATION SUMMARY"
echo "=" | tr '\n' '=' | head -c 50; echo

# Parse results from log
successful=0
failed=0

while IFS=':' read -r ratio_name status nodes reduction; do
    if [[ $status == "success" ]]; then
        echo "✅ $ratio_name: $nodes nodes (${reduction}x reduction)"
        ((successful++))
    else
        echo "❌ $ratio_name: $status"
        ((failed++))
    fi
done < $OUTPUT_DIR/generation_log.txt

echo
echo "📈 Results: $successful successful, $failed failed"
echo "📁 Matrices saved to: $OUTPUT_DIR/"

# List generated files
echo
echo "📄 Generated files:"
ls -la $OUTPUT_DIR/*.mtx 2>/dev/null || echo "   No .mtx files generated"

echo
if [ $successful -gt 0 ]; then
    echo "🎯 Ready for analysis! Run: python analyze_matrices.py"
else
    echo "⚠️  No matrices generated. Check errors above."
fi

echo "=" | tr '\n' '=' | head -c 50; echo
