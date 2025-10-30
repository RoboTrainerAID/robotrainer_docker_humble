#!/bin/bash
# Description:
#   Filters all .bag files in the current directory to remove topics containing 'camera'
#   Saves the filtered files inside a "filtered_bags" folder with "_cutted" appended.
#   Displays total execution time at the end.

set -e  # Exit immediately if a command exits with a non-zero status

# Record start time (in seconds)
start_time=$(date +%s)

# Define output directory
OUTPUT_DIR="filtered_bags"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Enable nullglob so that *.bag expands to nothing if no files exist
shopt -s nullglob
for bagfile in *.bag; do
    [ -e "$bagfile" ] || { echo "No .bag files found."; exit 0; }

    base="${bagfile%.bag}"
    output="${OUTPUT_DIR}/${base}_cutted.bag"

    echo "Processing file: $bagfile"
    echo "Output file:     $output"

    rosbag filter "$bagfile" "$output" "'camera' not in topic"

    echo "Completed: $output"
    echo
done

# Record end time and compute duration
end_time=$(date +%s)
elapsed=$(( end_time - start_time ))

# Format duration as minutes and seconds
minutes=$(( elapsed / 60 ))
seconds=$(( elapsed % 60 ))

echo "All .bag files processed successfully!"
echo "Filtered files saved in: $OUTPUT_DIR/"
echo "Total execution time: ${minutes}m ${seconds}s"