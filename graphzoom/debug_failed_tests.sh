#!/bin/bash
# Debug Failed Tests - Find out why all tests failed

echo "🔍 DEBUGGING FAILED TESTS"
echo "========================="

# Check if results directory exists
RESULTS_DIR="koutis_sample_validation"
if [ ! -d "$RESULTS_DIR" ]; then
    echo "❌ Results directory not found: $RESULTS_DIR"
    exit 1
fi

# Check CSV file
CSV_FILE=$(find $RESULTS_DIR -name "koutis_sample_results_*.csv" | head -1)
if [ ! -f "$CSV_FILE" ]; then
    echo "❌ No CSV results file found"
    exit 1
fi

echo "📁 Found CSV file: $CSV_FILE"
echo "📊 CSV contents:"
echo "================"
cat "$CSV_FILE"

echo ""
echo "🔍 Checking log files..."
echo "========================"

# Find log files
LOG_DIR="$RESULTS_DIR/logs"
if [ ! -d "$LOG_DIR" ]; then
    echo "❌ No logs directory found"
    exit 1
fi

LOG_FILES=$(find $LOG_DIR -name "*.log" | head -5)
if [ -z "$LOG_FILES" ]; then
    echo "❌ No log files found"
    exit 1
fi

echo "📄 Found log files:"
ls -la $LOG_DIR/*.log | head -5

echo ""
echo "🔍 Analyzing first few failed tests..."
echo "====================================="

for log_file in $(find $LOG_DIR -name "*.log" | head -3); do
    echo ""
    echo "📄 Log file: $(basename $log_file)"
    echo "-------------------------------------------"
    echo "📝 Content (first 20 lines):"
    head -20 "$log_file"
    echo ""
    echo "📝 Last 10 lines:"
    tail -10 "$log_file"
    echo "==========================================="
done

echo ""
echo "🧪 MANUAL TEST - Let's try one command manually"
echo "==============================================="

echo "Testing basic GraphZoom command..."
echo "Command: python graphzoom.py --dataset cora --embed_method deepwalk"
echo ""

# Run a simple test manually to see what happens
timeout 60 python graphzoom.py --dataset cora --embed_method deepwalk

echo ""
echo "🔍 DIAGNOSIS COMPLETE"
echo "===================="
echo "Check the output above to identify the issue:"
echo "1. Look at log file contents for error messages"
echo "2. Check if manual test worked"
echo "3. Identify common failure patterns"
