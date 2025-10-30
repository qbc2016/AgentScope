#!/bin/bash

set -e

# Clean old build files
rm -rf build/ doctrees/

# Generate the API rst files
sphinx-apidoc -o api ../../../src/agentscope -t ../_templates -e

# Key: Put doctrees outside build directory to avoid deployment
sphinx-build -M html ./ build -d ../doctrees_en

# Remove temporary files (double insurance)
rm -rf build/html/.doctrees
rm -f build/html/.buildinfo
find build/html -name "*.pickle" -delete
find build/html -name "__pycache__" -delete
find build/html -name "*.pyc" -delete

echo "✅ English docs built successfully, temporary files cleaned"