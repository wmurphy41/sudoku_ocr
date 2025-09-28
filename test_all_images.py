#!/usr/bin/env python3
"""
Comprehensive test script for Sudoku OCR using all test images.
Tests the current heuristics against expected outputs and provides detailed analysis.
"""

import sys
import os
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple
import re

# Add src to path
sys.path.append('src')

from sudoku_ocr.ocr import ocr_cells, to_grid
from sudoku_ocr.grid import find_and_warp
from sudoku_ocr.cells import split_into_cells
from sudoku_ocr.preprocess import to_binary
import cv2

def parse_test_data() -> Dict[str, str]:
    """Parse the TestData.txt file to extract expected outputs."""
    test_data_path = Path("data/raw/TestData.txt")
    
    if not test_data_path.exists():
        print(f"Error: Test data file not found: {test_data_path}")
        return {}
    
    expected_outputs = {}
    
    with open(test_data_path, 'r') as f:
        lines = f.readlines()
    
    for line_num, line in enumerate(lines, 1):
        line = line.strip()
        
        # Skip empty lines and comments
        if not line or line.startswith('#'):
            continue
        
        # Parse format: filename|expected_grid_81_chars
        if '|' in line:
            parts = line.split('|', 1)
            if len(parts) == 2:
                filename = parts[0].strip()
                grid_data = parts[1].strip()
                
                if len(grid_data) == 81:
                    expected_outputs[filename] = grid_data
                    print(f"Found test case: {filename}")
                else:
                    print(f"Warning: Invalid grid length for {filename} (line {line_num}): {len(grid_data)}")
            else:
                print(f"Warning: Invalid format on line {line_num}: {line}")
        else:
            print(f"Warning: Missing '|' separator on line {line_num}: {line}")
    
    return expected_outputs

def run_ocr_pipeline(image_path: str) -> Tuple[List[List[int]], List[int]]:
    """Run the complete OCR pipeline on an image."""
    try:
        # Load and preprocess image
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            print(f"Error: Could not load image {image_path}")
            return None, None
        
        # Apply preprocessing
        binary = to_binary(img_bgr, apply_clahe=False)
        
        # Find and warp grid
        artifacts = find_and_warp(img_bgr, size=450, apply_clahe=False)
        if artifacts is None:
            print(f"Error: Could not find grid in {image_path}")
            return None, None
        
        warped = artifacts['warped']
        
        # Extract cells
        cells = split_into_cells(warped, pad=4)
        if len(cells) != 81:
            print(f"Error: Expected 81 cells, got {len(cells)} for {image_path}")
            return None, None
        
        # Perform OCR
        digits = ocr_cells(cells, conf_thresh=0.45)
        
        # Convert to grid format
        grid = to_grid(digits)
        
        return grid, digits
        
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None, None

def compare_grids(recognized: List[int], expected: str) -> Dict[str, any]:
    """Compare recognized digits with expected output."""
    if len(recognized) != 81:
        return {"error": f"Invalid recognized length: {len(recognized)}"}
    
    if len(expected) != 81:
        return {"error": f"Invalid expected length: {len(expected)}"}
    
    # Convert expected string to list of ints
    expected_digits = [int(c) for c in expected]
    
    # Calculate metrics
    total_cells = 81
    correct = 0
    wrong = 0
    missed = 0
    false_positives = 0
    
    errors = []
    
    for i in range(81):
        row = i // 9
        col = i % 9
        recognized_digit = recognized[i]
        expected_digit = expected_digits[i]
        
        if expected_digit == 0:  # Empty cell
            if recognized_digit != 0:
                false_positives += 1
                errors.append(f"FP r{row}c{col}: expected empty, got {recognized_digit}")
        else:  # Non-empty cell
            if recognized_digit == 0:
                missed += 1
                errors.append(f"MISS r{row}c{col}: expected {expected_digit}, got empty")
            elif recognized_digit == expected_digit:
                correct += 1
            else:
                wrong += 1
                errors.append(f"WRONG r{row}c{col}: expected {expected_digit}, got {recognized_digit}")
    
    accuracy = correct / (total_cells - expected.count('0')) if expected.count('0') < total_cells else 0
    precision = correct / (correct + wrong) if (correct + wrong) > 0 else 0
    recall = correct / (correct + missed) if (correct + missed) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "total_cells": total_cells,
        "filled_cells": total_cells - expected.count('0'),
        "correct": correct,
        "wrong": wrong,
        "missed": missed,
        "false_positives": false_positives,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "errors": errors
    }

def print_grid_comparison(recognized: List[List[int]], expected: str, filename: str):
    """Print a side-by-side comparison of grids."""
    print(f"\n{'='*60}")
    print(f"GRID COMPARISON: {filename}")
    print(f"{'='*60}")
    
    expected_grid = [[int(expected[i*9 + j]) for j in range(9)] for i in range(9)]
    
    print("RECOGNIZED vs EXPECTED")
    print("+-------------------------+  +-------------------------+")
    
    for row in range(9):
        rec_line = "|"
        exp_line = "|"
        
        for col in range(9):
            rec_digit = recognized[row][col] if recognized[row][col] != 0 else " "
            exp_digit = expected_grid[row][col] if expected_grid[row][col] != 0 else " "
            
            # Add spacing
            rec_line += f" {rec_digit} "
            exp_line += f" {exp_digit} "
            
            # Add vertical separators
            if col in [2, 5]:
                rec_line += "|"
                exp_line += "|"
        
        rec_line += "|"
        exp_line += "|"
        
        print(f"{rec_line}  {exp_line}")
        
        # Add horizontal separators
        if row in [2, 5]:
            print("+-------------------------+  +-------------------------+")
    
    print("+-------------------------+  +-------------------------+")

def main():
    """Main test function."""
    print("Comprehensive Sudoku OCR Test")
    print("=" * 50)
    
    # Parse test data
    print("Parsing test data...")
    expected_outputs = parse_test_data()
    print(f"Parsed {len(expected_outputs)} test cases")
    
    if not expected_outputs:
        print("No test data found!")
        return
    
    print(f"Found {len(expected_outputs)} test images:")
    for filename in expected_outputs.keys():
        print(f"  - {filename}")
    
    # Test each image
    results = {}
    total_metrics = {
        "total_correct": 0,
        "total_wrong": 0,
        "total_missed": 0,
        "total_false_positives": 0,
        "total_filled_cells": 0
    }
    
    for filename, expected in expected_outputs.items():
        image_path = f"data/raw/{filename}"
        
        if not os.path.exists(image_path):
            print(f"Warning: Image file not found: {image_path}")
            continue
        
        print(f"\nTesting: {filename}")
        print("-" * 40)
        
        # Run OCR pipeline
        grid, digits = run_ocr_pipeline(image_path)
        
        if grid is None or digits is None:
            print(f"Failed to process {filename}")
            continue
        
        # Compare with expected
        metrics = compare_grids(digits, expected)
        
        if "error" in metrics:
            print(f"Error: {metrics['error']}")
            continue
        
        # Store results
        results[filename] = {
            "grid": grid,
            "digits": digits,
            "expected": expected,
            "metrics": metrics
        }
        
        # Print comparison
        print_grid_comparison(grid, expected, filename)
        
        # Print metrics
        print(f"\nMETRICS for {filename}:")
        print(f"  Accuracy:  {metrics['accuracy']:.3f} ({metrics['correct']}/{metrics['filled_cells']})")
        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  Recall:    {metrics['recall']:.3f}")
        print(f"  F1-Score:  {metrics['f1_score']:.3f}")
        print(f"  Errors:    {len(metrics['errors'])}")
        
        # Show first few errors
        if metrics['errors']:
            print(f"  First 5 errors:")
            for error in metrics['errors'][:5]:
                print(f"    {error}")
        
        # Update totals
        total_metrics["total_correct"] += metrics["correct"]
        total_metrics["total_wrong"] += metrics["wrong"]
        total_metrics["total_missed"] += metrics["missed"]
        total_metrics["total_false_positives"] += metrics["false_positives"]
        total_metrics["total_filled_cells"] += metrics["filled_cells"]
    
    # Print overall summary
    print(f"\n{'='*60}")
    print("OVERALL SUMMARY")
    print(f"{'='*60}")
    
    if total_metrics["total_filled_cells"] > 0:
        overall_accuracy = total_metrics["total_correct"] / total_metrics["total_filled_cells"]
        overall_precision = total_metrics["total_correct"] / (total_metrics["total_correct"] + total_metrics["total_wrong"]) if (total_metrics["total_correct"] + total_metrics["total_wrong"]) > 0 else 0
        overall_recall = total_metrics["total_correct"] / (total_metrics["total_correct"] + total_metrics["total_missed"]) if (total_metrics["total_correct"] + total_metrics["total_missed"]) > 0 else 0
        overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
        
        print(f"Total filled cells: {total_metrics['total_filled_cells']}")
        print(f"Correct:           {total_metrics['total_correct']}")
        print(f"Wrong:             {total_metrics['total_wrong']}")
        print(f"Missed:            {total_metrics['total_missed']}")
        print(f"False positives:   {total_metrics['total_false_positives']}")
        print(f"")
        print(f"Overall Accuracy:  {overall_accuracy:.3f}")
        print(f"Overall Precision: {overall_precision:.3f}")
        print(f"Overall Recall:    {overall_recall:.3f}")
        print(f"Overall F1-Score:  {overall_f1:.3f}")
    
    # Save detailed results
    results_file = "test_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to: {results_file}")

if __name__ == "__main__":
    main()
