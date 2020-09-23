#!/bin/bash

echo "Checking code style"
echo ">>>>>>>>>>>>>>>>>>>>>>>"

format_occurred=false
declare -a black_dirs=("models/" "optimizers/" "utils/" "visualizations/" "./")
for black_dir in "${black_dirs[@]}"; do
    echo ">>> Checking $black_dir"
    black --check "$black_dir"

    if [ $? -ne 0 ]; then
        echo ">>> Failed, reformatting now!"
        black "$black_dir"
        format_occurred=true
    fi
done

# exit needed because reformatted files are unstaged
if [ "$format_occurred" = true ]; then
    echo "At least one file was formatted, exiting before tests"
    exit 1
fi
