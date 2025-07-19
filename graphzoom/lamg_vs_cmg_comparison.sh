#!/bin/bash
# Definitive LAMG vs CMG Comparison

echo "🔥 DEFINITIVE LAMG vs CMG COMPARISON"
echo "==================================="

# Create results file
echo "method,dataset,embedding,accuracy,time,clusters,compression,config" > lamg_vs_cmg_results.csv

run_comparison_test() {
    local method=$1
    local config_name=$2
    local extra_params=$3
    
    echo "🔄 Running: $method ($config_name)"
    
    if [ "$method" = "lamg" ]; then
        timeout 400 python graphzoom.py --dataset cora --coarse lamg --mcr_dir $MATLAB_MCR_ROOT --embed_method deepwalk $extra_params > ${method}_${config_name}.log 2>&1
    else
        timeout 400 python graphzoom.py --dataset cora --coarse cmg --embed_method deepwalk $extra_params > ${method}_${config_name}.log 2>&1
    fi
    
    if [ $? -eq 0 ]; then
        accuracy=$(grep "Test Accuracy:" ${method}_${config_name}.log | awk '{print $3}')
        time=$(grep "Total Time" ${method}_${config_name}.log | awk '{print $NF}')
        
        # Extract cluster info
        if [ "$method" = "cmg" ]; then
            clusters=$(grep "Final graph:" ${method}_${config_name}.log | tail -1 | awk '{print $4}')
        else
            # For LAMG, try to extract from reduction results
            clusters=$(grep -i "coarsen\|nodes" ${method}_${config_name}.log | grep -o "[0-9]\+")
        fi
        
        # Calculate compression
        if [ -n "$clusters" ] && [ "$clusters" -gt 0 ]; then
            compression=$(echo "scale=2; 2708 / $clusters" | bc -l)
        else
            compression="unknown"
        fi
        
        echo "$method,cora,deepwalk,$accuracy,$time,$clusters,$compression,$config_name" >> lamg_vs_cmg_results.csv
        echo "✅ $method ($config_name): Acc=$accuracy, Time=${time}s, Clusters=$clusters"
    else
        echo "❌ $method ($config_name): FAILED"
        echo "$method,cora,deepwalk,FAILED,TIMEOUT,unknown,unknown,$config_name" >> lamg_vs_cmg_results.csv
    fi
}

echo ""
echo "📊 ROUND 1: Standard Configurations"
echo "-----------------------------------"

# LAMG baseline
run_comparison_test "lamg" "standard" "--level 1"

# CMG configurations
run_comparison_test "cmg" "default" "--level 1 --cmg_k 10 --cmg_d 20"
run_comparison_test "cmg" "best_params" "--level 1 --cmg_k 10 --cmg_d 10"  # Best from your tests
run_comparison_test "cmg" "multilevel2" "--level 2 --cmg_k 10 --cmg_d 10"

echo ""
echo "📊 ROUND 2: Head-to-Head Comparison"
echo "-----------------------------------"

# Multiple runs for reliability
for i in {1..3}; do
    echo "🔄 Run $i/3:"
    run_comparison_test "lamg" "run$i" "--level 1"
    run_comparison_test "cmg" "run$i" "--level 1 --cmg_k 10 --cmg_d 10"
done

echo ""
echo "✅ All comparisons completed!"
echo "📁 Results saved in: lamg_vs_cmg_results.csv"

# Quick analysis
echo ""
echo "📊 QUICK ANALYSIS:"
echo "=================="

if [ -f lamg_vs_cmg_results.csv ]; then
    echo "🏆 ACCURACY COMPARISON:"
    echo "Method    | Config      | Accuracy | Time     | Compression"
    echo "----------|-------------|----------|----------|------------"
    
    tail -n +2 lamg_vs_cmg_results.csv | while IFS=',' read -r method dataset embedding accuracy time clusters compression config; do
        if [ "$accuracy" != "FAILED" ]; then
            printf "%-9s | %-11s | %-8s | %-8s | %s\n" "$method" "$config" "$accuracy" "${time}s" "${compression}x"
        fi
    done
    
    echo ""
    echo "📈 SUMMARY STATISTICS:"
    
    # Best accuracy for each method
    lamg_best=$(tail -n +2 lamg_vs_cmg_results.csv | grep "lamg" | grep -v "FAILED" | cut -d',' -f4 | sort -nr | head -1)
    cmg_best=$(tail -n +2 lamg_vs_cmg_results.csv | grep "cmg" | grep -v "FAILED" | cut -d',' -f4 | sort -nr | head -1)
    
    # Average times
    lamg_avg_time=$(tail -n +2 lamg_vs_cmg_results.csv | grep "lamg" | grep -v "TIMEOUT" | cut -d',' -f5 | awk '{sum+=$1; count++} END {if(count>0) print sum/count; else print "N/A"}')
    cmg_avg_time=$(tail -n +2 lamg_vs_cmg_results.csv | grep "cmg" | grep -v "TIMEOUT" | cut -d',' -f5 | awk '{sum+=$1; count++} END {if(count>0) print sum/count; else print "N/A"}')
    
    echo "🎯 Best Accuracy:"
    echo "   LAMG: $lamg_best"
    echo "   CMG:  $cmg_best"
    
    if [ -n "$lamg_best" ] && [ -n "$cmg_best" ]; then
        winner=$(python3 -c "
lamg = float('$lamg_best')
cmg = float('$cmg_best')
diff = cmg - lamg
if diff > 0.01:
    print(f'🏆 CMG WINS by {diff:.3f}!')
elif diff < -0.01:
    print(f'🏆 LAMG WINS by {-diff:.3f}!')
else:
    print('🤝 TIE - Similar performance!')
print(f'Difference: {diff:+.3f}')
")
        echo "$winner"
    fi
    
    echo ""
    echo "⚡ Average Speed:"
    echo "   LAMG: ${lamg_avg_time}s"
    echo "   CMG:  ${cmg_avg_time}s"
    
    if [ "$lamg_avg_time" != "N/A" ] && [ "$cmg_avg_time" != "N/A" ]; then
        speedup=$(python3 -c "print(f'{float('$lamg_avg_time') / float('$cmg_avg_time'):.1f}x')")
        echo "   CMG Speedup: $speedup"
    fi
fi
