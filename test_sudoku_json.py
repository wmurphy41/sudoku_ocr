#!/usr/bin/env python3
"""
Test function for Sudoku OCR that compares expected outputs with JSON results.
"""

import sys
import json
import subprocess
from pathlib import Path
from typing import List, Tuple, Dict, Any
import tempfile
import shutil

def parse_test_data(test_data_file: str) -> List[Tuple[str, str]]:
    """
    Parse test data file to extract image filenames and expected outputs.
    
    Args:
        test_data_file: Path to TestData.txt file
        
    Returns:
        List of tuples (image_filename, expected_grid_string)
    """
    test_cases = []
    
    with open(test_data_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                parts = line.split('|')
                if len(parts) == 2:
                    image_filename = parts[0].strip()
                    expected_grid = parts[1].strip()
                    test_cases.append((image_filename, expected_grid))
    
    return test_cases

def run_sudoku_cli(image_path: str, output_dir: str) -> bool:
    """
    Run sudoku_ocr CLI on an image.
    
    Args:
        image_path: Path to input image
        output_dir: Directory to save output files
        
    Returns:
        True if successful, False otherwise
    """
    try:
        cmd = [
            sys.executable, '-m', 'sudoku_ocr.cli',
            '--image', image_path,
            '--out', output_dir
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            print(f"CLI error output: {result.stderr}")
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print(f"Timeout running CLI on {image_path}")
        return False
    except Exception as e:
        print(f"Error running CLI on {image_path}: {e}")
        return False

def load_json_output(output_dir: str, image_filename: str) -> List[List[int]]:
    """
    Load the JSON output file for an image.
    
    Args:
        output_dir: Output directory path
        image_filename: Image filename (without extension)
        
    Returns:
        9x9 grid as list of lists, or None if file not found
    """
    json_file = Path(output_dir) / f"{image_filename}_puzzle.json"
    
    if not json_file.exists():
        return None
    
    try:
        with open(json_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading JSON file {json_file}: {e}")
        return None

def expected_string_to_grid(expected_string: str) -> List[List[int]]:
    """
    Convert expected string (81 characters) to 9x9 grid.
    
    Args:
        expected_string: String of 81 characters representing the grid
        
    Returns:
        9x9 grid as list of lists
    """
    if len(expected_string) != 81:
        raise ValueError(f"Expected string must be 81 characters, got {len(expected_string)}")
    
    grid = []
    for i in range(9):
        row = []
        for j in range(9):
            char = expected_string[i * 9 + j]
            if char == '0':
                row.append(0)
            else:
                row.append(int(char))
        grid.append(row)
    
    return grid

def compare_grids(expected: List[List[int]], actual: List[List[int]]) -> Dict[str, Any]:
    """
    Compare expected and actual grids.
    
    Args:
        expected: Expected 9x9 grid
        actual: Actual 9x9 grid
        
    Returns:
        Dictionary with comparison results
    """
    if len(expected) != 9 or len(actual) != 9:
        return {"error": "Invalid grid dimensions"}
    
    correct = 0
    wrong = 0
    missed = 0
    false_positives = 0
    errors = []
    
    for i in range(9):
        for j in range(9):
            exp_val = expected[i][j]
            act_val = actual[i][j]
            
            if exp_val == 0 and act_val == 0:
                # Both empty - correct
                pass
            elif exp_val != 0 and act_val == exp_val:
                # Both have same non-zero value - correct
                correct += 1
            elif exp_val == 0 and act_val != 0:
                # Expected empty, got value - false positive
                false_positives += 1
                errors.append(f"FP r{i}c{j}: expected empty, got {act_val}")
            elif exp_val != 0 and act_val == 0:
                # Expected value, got empty - missed
                missed += 1
                errors.append(f"MISS r{i}c{j}: expected {exp_val}, got empty")
            else:
                # Different non-zero values - wrong
                wrong += 1
                errors.append(f"WRONG r{i}c{j}: expected {exp_val}, got {act_val}")
    
    total_filled = sum(1 for row in expected for val in row if val != 0)
    
    accuracy = correct / total_filled if total_filled > 0 else 0
    precision = correct / (correct + false_positives) if (correct + false_positives) > 0 else 0
    recall = correct / (correct + missed) if (correct + missed) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "correct": correct,
        "wrong": wrong,
        "missed": missed,
        "false_positives": false_positives,
        "total_filled": total_filled,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "errors": errors[:10]  # First 10 errors
    }

def test_sudoku_ocr(test_data_file: str, images_dir: str, output_dir: str = None) -> Dict[str, Any]:
    """
    Test Sudoku OCR on multiple images and compare with expected outputs.
    
    Args:
        test_data_file: Path to TestData.txt file
        images_dir: Directory containing test images
        output_dir: Output directory (if None, uses temporary directory)
        
    Returns:
        Dictionary with test results
    """
    # Parse test data
    test_cases = parse_test_data(test_data_file)
    print(f"Found {len(test_cases)} test cases")
    
    # Create output directory
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="sudoku_test_")
        cleanup_output = True
    else:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        cleanup_output = False
    
    results = {
        "test_cases": [],
        "overall_stats": {
            "total_tests": len(test_cases),
            "successful_tests": 0,
            "failed_tests": 0,
            "total_correct": 0,
            "total_wrong": 0,
            "total_missed": 0,
            "total_false_positives": 0,
            "overall_accuracy": 0,
            "overall_precision": 0,
            "overall_recall": 0,
            "overall_f1_score": 0
        }
    }
    
    print(f"\nTesting Sudoku OCR...")
    print("=" * 50)
    
    for i, (image_filename, expected_string) in enumerate(test_cases):
        print(f"\nTest {i+1}/{len(test_cases)}: {image_filename}")
        print("-" * 40)
        
        # Construct image path
        image_path = Path(images_dir) / image_filename
        if not image_path.exists():
            print(f"  ERROR: Image file not found: {image_path}")
            results["test_cases"].append({
                "image": image_filename,
                "status": "failed",
                "error": "Image file not found"
            })
            results["overall_stats"]["failed_tests"] += 1
            continue
        
        # Extract filename without extension
        image_stem = Path(image_filename).stem
        
        # Run CLI
        print(f"  Running CLI on {image_filename}...")
        success = run_sudoku_cli(str(image_path), output_dir)
        
        if not success:
            print(f"  ERROR: CLI failed for {image_filename}")
            results["test_cases"].append({
                "image": image_filename,
                "status": "failed",
                "error": "CLI execution failed"
            })
            results["overall_stats"]["failed_tests"] += 1
            continue
        
        # Load JSON output
        actual_grid = load_json_output(output_dir, image_stem)
        if actual_grid is None:
            print(f"  ERROR: Could not load JSON output for {image_filename}")
            results["test_cases"].append({
                "image": image_filename,
                "status": "failed",
                "error": "Could not load JSON output"
            })
            results["overall_stats"]["failed_tests"] += 1
            continue
        
        # Convert expected string to grid
        try:
            expected_grid = expected_string_to_grid(expected_string)
        except Exception as e:
            print(f"  ERROR: Invalid expected string for {image_filename}: {e}")
            results["test_cases"].append({
                "image": image_filename,
                "status": "failed",
                "error": f"Invalid expected string: {e}"
            })
            results["overall_stats"]["failed_tests"] += 1
            continue
        
        # Compare grids
        comparison = compare_grids(expected_grid, actual_grid)
        
        print(f"  Results:")
        print(f"    Accuracy: {comparison['accuracy']:.3f} ({comparison['correct']}/{comparison['total_filled']})")
        print(f"    Precision: {comparison['precision']:.3f}")
        print(f"    Recall: {comparison['recall']:.3f}")
        print(f"    F1-Score: {comparison['f1_score']:.3f}")
        print(f"    Errors: {comparison['wrong'] + comparison['missed'] + comparison['false_positives']}")
        
        if comparison['errors']:
            print(f"    First few errors:")
            for error in comparison['errors'][:3]:
                print(f"      {error}")
        
        # Store results
        test_result = {
            "image": image_filename,
            "status": "success",
            "comparison": comparison
        }
        results["test_cases"].append(test_result)
        results["overall_stats"]["successful_tests"] += 1
        
        # Update overall stats
        results["overall_stats"]["total_correct"] += comparison["correct"]
        results["overall_stats"]["total_wrong"] += comparison["wrong"]
        results["overall_stats"]["total_missed"] += comparison["missed"]
        results["overall_stats"]["total_false_positives"] += comparison["false_positives"]
    
    # Calculate overall metrics
    total_correct = results["overall_stats"]["total_correct"]
    total_wrong = results["overall_stats"]["total_wrong"]
    total_missed = results["overall_stats"]["total_missed"]
    total_false_positives = results["overall_stats"]["total_false_positives"]
    
    if total_correct + total_missed > 0:
        results["overall_stats"]["overall_accuracy"] = total_correct / (total_correct + total_missed)
    if total_correct + total_false_positives > 0:
        results["overall_stats"]["overall_precision"] = total_correct / (total_correct + total_false_positives)
    if total_correct + total_missed > 0:
        results["overall_stats"]["overall_recall"] = total_correct / (total_correct + total_missed)
    
    precision = results["overall_stats"]["overall_precision"]
    recall = results["overall_stats"]["overall_recall"]
    if precision + recall > 0:
        results["overall_stats"]["overall_f1_score"] = 2 * (precision * recall) / (precision + recall)
    
    # Print overall summary
    print(f"\n" + "=" * 50)
    print(f"OVERALL SUMMARY")
    print(f"=" * 50)
    print(f"Total tests: {results['overall_stats']['total_tests']}")
    print(f"Successful: {results['overall_stats']['successful_tests']}")
    print(f"Failed: {results['overall_stats']['failed_tests']}")
    print(f"Total correct: {results['overall_stats']['total_correct']}")
    print(f"Total wrong: {results['overall_stats']['total_wrong']}")
    print(f"Total missed: {results['overall_stats']['total_missed']}")
    print(f"Total false positives: {results['overall_stats']['total_false_positives']}")
    print(f"Overall Accuracy: {results['overall_stats']['overall_accuracy']:.3f}")
    print(f"Overall Precision: {results['overall_stats']['overall_precision']:.3f}")
    print(f"Overall Recall: {results['overall_stats']['overall_recall']:.3f}")
    print(f"Overall F1-Score: {results['overall_stats']['overall_f1_score']:.3f}")
    
    # Cleanup temporary directory
    if cleanup_output:
        shutil.rmtree(output_dir)
    
    return results

def main():
    """Main function to run the test."""
    if len(sys.argv) < 3:
        print("Usage: python test_sudoku_json.py <test_data_file> <images_dir> [output_dir]")
        print("Example: python test_sudoku_json.py data/raw/TestData.txt data/raw")
        sys.exit(1)
    
    test_data_file = sys.argv[1]
    images_dir = sys.argv[2]
    output_dir = sys.argv[3] if len(sys.argv) > 3 else None
    
    results = test_sudoku_ocr(test_data_file, images_dir, output_dir)
    
    # Save results to JSON file
    results_file = "test_json_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to: {results_file}")

if __name__ == "__main__":
    main()
