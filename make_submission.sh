#!/usr/bin/env bash
set -e

if [ "$#" -ne 1 ]; then
    echo "Usage: bash make_submission.sh <roll_number>"
    echo "Example: bash make_submission.sh 24100123"
    exit 1
fi

ROLL_NUMBER=$1
ARCHIVE_NAME="${ROLL_NUMBER}_pa3.zip"

echo "Creating submission archive: ${ARCHIVE_NAME}"

# Verify pipeline notebook or script exists
PIPELINE_FILE=""
if [ -f "run_pipeline.ipynb" ]; then
    PIPELINE_FILE="run_pipeline.ipynb"
elif [ -f "run_all.py" ]; then
    PIPELINE_FILE="run_all.py"
else
    echo "Error: Neither run_pipeline.ipynb nor run_all.py found in repository root."
    echo "Please provide your runnable pipeline notebook or script."
    exit 1
fi

# Verify required files and directories exist
for req in "submission_adapter.py" "src" "outputs/per_problem.jsonl" "outputs/results_summary.json" "outputs/aggregation_ablation.json" "outputs/sc_scaling_curve.json" "REPORT.pdf"; do
    if [ ! -e "$req" ]; then
        echo "Error: Required submission file or directory '$req' not found."
        echo "Please ensure you have run the evaluation scripts and placed REPORT.pdf in the repository root."
        exit 1
    fi
done

# Run public smoke tests
echo "Running public smoke tests..."
python tests/test_public_adapter_smoke.py || echo "Warning: Some tests failed. Proceeding with packaging."

# Create zip archive
zip -r "${ARCHIVE_NAME}" \
    src/ \
    submission_adapter.py \
    "${PIPELINE_FILE}" \
    outputs/per_problem.jsonl \
    outputs/results_summary.json \
    outputs/aggregation_ablation.json \
    outputs/sc_scaling_curve.json \
    REPORT.pdf \
    -x "__pycache__/*" "src/__pycache__/*" "*.pyc" "_reference/*"

echo "=========================================="
echo "Successfully generated: ${ARCHIVE_NAME}"
echo "Please verify its contents and upload to the LMS."
echo "=========================================="
