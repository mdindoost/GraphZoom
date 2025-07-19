#!/bin/bash
# Script to clean up Phase 2 results and fix the comprehensive test suite

echo "🧹 CLEANING UP PHASE 2 RESULTS AND FIXING TEST SUITE"
echo "===================================================="

# Find the results CSV file
RESULTS_CSV=$(find . -name "comprehensive_results_*.csv" -type f | head -1)

if [ -z "$RESULTS_CSV" ]; then
    echo "❌ No comprehensive results CSV file found"
    exit 1
fi

echo "📁 Found results file: $RESULTS_CSV"

# Backup original file
cp "$RESULTS_CSV" "${RESULTS_CSV}.backup"
echo "💾 Backup created: ${RESULTS_CSV}.backup"

# Count original Phase 2 tests
original_phase2_count=$(grep "^phase2," "$RESULTS_CSV" | wc -l)
echo "📊 Original Phase 2 tests: $original_phase2_count"

# Remove all Phase 2 entries from CSV
echo "🗑️  Removing all Phase 2 entries from results..."
grep -v "^phase2," "$RESULTS_CSV" > "${RESULTS_CSV}.temp"
mv "${RESULTS_CSV}.temp" "$RESULTS_CSV"

# Count remaining tests
remaining_tests=$(wc -l < "$RESULTS_CSV")
echo "📊 Remaining tests after cleanup: $((remaining_tests - 1))"  # -1 for header

# Clean up Phase 2 log files
echo "🗑️  Removing Phase 2 log files..."
LOGS_DIR=$(dirname "$RESULTS_CSV")/logs
if [ -d "$LOGS_DIR" ]; then
    removed_logs=$(find "$LOGS_DIR" -name "phase2_*.log" -type f | wc -l)
    find "$LOGS_DIR" -name "phase2_*.log" -type f -delete
    echo "📊 Removed $removed_logs Phase 2 log files"
else
    echo "📁 No logs directory found"
fi

echo ""
echo "✅ Phase 2 cleanup completed!"
echo "📁 Updated results file: $RESULTS_CSV"
echo "💾 Backup available: ${RESULTS_CSV}.backup"
echo ""
echo "Next step: Run the fixed comprehensive test suite"
