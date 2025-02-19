#!/bin/bash
# Enable nullglob so that non-matching globs expand to nothing.
shopt -s nullglob

# Exit if no file extension arguments are provided.
if [ "$#" -eq 0 ]; then
  echo "Usage: $0 ext1 [ext2 ...]"
  exit 1
fi

output_file="output.md"
# Clear (or create) the output file.
> "$output_file"

# Use an associative array to avoid duplicate file processing.
declare -A processed

# Loop over each file extension argument.
for ext in "$@"; do
  # Remove a leading dot if present.
  ext="${ext#.}"
  # Loop over all files with the given extension.
  for file in *."$ext"; do
    if [ -f "$file" ] && [ -z "${processed[$file]}" ]; then
      # Append a markdown header with the file's path.
      echo "## $file" >> "$output_file"
      echo >> "$output_file"
      # Append the file's content.
      cat "$file" >> "$output_file"
      echo >> "$output_file"
      echo >> "$output_file"
      processed["$file"]=1
    fi
  done
done

echo "Merged content saved in $output_file"

