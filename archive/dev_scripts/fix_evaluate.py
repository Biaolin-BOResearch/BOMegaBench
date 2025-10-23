#!/usr/bin/env python3
"""
Script to fix _evaluate method names to _evaluate_true for BoTorch compatibility.
"""

import os
import re

def fix_file(filepath):
    """Fix _evaluate methods in a file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace def _evaluate( with def _evaluate_true(
    content = re.sub(r'def _evaluate\(', 'def _evaluate_true(', content)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Fixed {filepath}")

def main():
    """Fix all function files."""
    files_to_fix = [
        'bomegabench/functions/classical.py',
        'bomegabench/functions/botorch_functions.py',
    ]
    
    for filepath in files_to_fix:
        if os.path.exists(filepath):
            fix_file(filepath)
        else:
            print(f"File not found: {filepath}")

if __name__ == "__main__":
    main() 