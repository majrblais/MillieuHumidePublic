#!/bin/bash

# Check if a file argument is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <file>"
    exit 1
fi

FILE="$1.tex"

# Check if the file exists
if [ ! -f "$FILE" ]; then
    echo "File $FILE not found!"
    exit 1
fi

# Run indefinitely
while true; do 
    # Wait for the file to be modified
    inotifywait -e modify "$FILE"

    # Compile the LaTeX file
    pdflatex -shell-escape "$FILE"
    bibtex "${FILE%.tex}"  # Remove the .tex extension for bibtex
    pdflatex -shell-escape "$FILE"
    pdflatex -shell-escape "$FILE"
done
