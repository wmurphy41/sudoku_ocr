"""
Command-line interface for the Sudoku OCR pipeline.

This module provides the main entry point for processing Sudoku images
through the complete pipeline: preprocessing -> grid detection -> cell extraction.
"""

import argparse
import sys
from pathlib import Path
import cv2
import numpy as np

from .preprocess import to_binary, largest_contour
from .grid import find_and_warp, approx_to_quad, order_corners, warp_to_square, create_quad_overlay, GridNotFoundError
from .cells import split_into_cells
from .ocr import ocr_cells, to_grid, print_grid


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Extract Sudoku puzzle from image",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m sudoku_ocr.cli --image data/raw/example.jpg --out out --save-cells
  python -m sudoku_ocr.cli --image data/raw/example.jpg --out out --save-cells --debug
  python -m sudoku_ocr.cli --image data/raw/example.jpg --out out --size 600 --pad 6
        """
    )
    
    parser.add_argument(
        "--image",
        required=True,
        help="Path to input Sudoku image"
    )
    
    parser.add_argument(
        "--out",
        default="out",
        help="Output directory for processed images (default: out)"
    )
    
    parser.add_argument(
        "--save-cells",
        action="store_true",
        help="Save all 81 individual cells to data/processed/cells/"
    )
    
    parser.add_argument(
        "--size",
        type=int,
        default=450,
        help="Size of the warped square output (default: 450)"
    )
    
    parser.add_argument(
        "--pad",
        type=int,
        default=4,
        help="Inner padding around each cell to avoid grid lines (default: 4)"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Save extra intermediate debug images"
    )
    
    parser.add_argument(
        "--debug-first-cell",
        action="store_true",
        help="Debug mode: process only the first cell and exit"
    )
    
    parser.add_argument(
        "--apply-clahe",
        action="store_true",
        help="Apply CLAHE contrast enhancement before thresholding"
    )
    
    parser.add_argument(
        "--ocr",
        action="store_true",
        default=True,
        help="Perform OCR digit recognition on extracted cells (default: True)"
    )
    parser.add_argument(
        "--no-ocr",
        dest="ocr",
        action="store_false",
        help="Skip OCR digit recognition"
    )
    
    parser.add_argument(
        "--ocr-conf",
        type=float,
        default=0.45,
        help="OCR confidence threshold for digit recognition (default: 0.45)"
    )
    
    
    parser.add_argument(
        "--tesseract-conf",
        type=float,
        default=0.45,
        help="Tesseract confidence threshold for digit recognition (default: 0.45)"
    )
    
    args = parser.parse_args()
    
    # Validate input file
    input_path = Path(args.image)
    if not input_path.exists():
        print(f"Error: Input file '{input_path}' not found", file=sys.stderr)
        sys.exit(1)
    
    # Extract filename without extension for output file naming
    input_filename = input_path.stem
    
    # Create output directory
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load image
        print(f"Loading image: {input_path}")
        image = cv2.imread(str(input_path))
        if image is None:
            print(f"Error: Could not load image '{input_path}'", file=sys.stderr)
            sys.exit(1)
        
        # Stage 1: Preprocessing and grid detection
        print("Stage 1: Preprocessing and grid detection...")
        try:
            artifacts = find_and_warp(image, size=args.size, apply_clahe=args.apply_clahe)
        except GridNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            print("Please ensure the image contains a clear Sudoku puzzle with visible grid lines", file=sys.stderr)
            sys.exit(2)
        
        print("[OK] Grid detected successfully!")
        
        # Stage 2: Cell extraction
        print("Stage 2: Extracting cells...")
        try:
            cells = split_into_cells(artifacts['warped'], pad=args.pad)
            print(f"[OK] Extracted {len(cells)} cells")
        except ValueError as e:
            print(f"Error extracting cells: {e}", file=sys.stderr)
            sys.exit(1)
        
        # Save required artifacts
        print("Saving artifacts...")
        
        # binary.png
        cv2.imwrite(str(output_dir / f"{input_filename}_binary.png"), artifacts['binary'])
        print(f"  Saved: {output_dir / f'{input_filename}_binary.png'}")
        
        # contour_mask.png (white fill of largest contour)
        cv2.imwrite(str(output_dir / f"{input_filename}_contour_mask.png"), artifacts['contour_mask'])
        print(f"  Saved: {output_dir / f'{input_filename}_contour_mask.png'}")
        
        # quad_overlay.png (original + quad with circles and lines)
        quad_overlay = create_quad_overlay(image, artifacts['quad'])
        cv2.imwrite(str(output_dir / f"{input_filename}_quad_overlay.png"), quad_overlay)
        print(f"  Saved: {output_dir / f'{input_filename}_quad_overlay.png'}")
        
        # warped.png
        cv2.imwrite(str(output_dir / f"{input_filename}_warped.png"), artifacts['warped'])
        print(f"  Saved: {output_dir / f'{input_filename}_warped.png'}")
        
        # Save cells if requested
        if args.save_cells:
            print("Saving individual cells...")
            try:
                # Create cells directory
                cells_dir = Path("data/processed/cells")
                cells_dir.mkdir(parents=True, exist_ok=True)
                
                # Save all 81 cells
                for i, cell in enumerate(cells):
                    row = i // 9
                    col = i % 9
                    filename = f"{input_filename}_r{row}_c{col}.png"
                    cv2.imwrite(str(cells_dir / filename), cell)
                
                print(f"  Saved 81 cells to: {cells_dir}")
                
                # Save debug binary images if debug mode is enabled
                if args.debug:
                    print("Saving debug binary cell images...")
                    from .ocr import preprocess_cell
                    
                    # Create cell debug directory
                    cell_debug_dir = output_dir / "cell_debug"
                    cell_debug_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Process and save binary versions of each cell
                    for i, cell in enumerate(cells):
                        row = i // 9
                        col = i % 9
                        filename = f"{input_filename}_r{row}_c{col}_bin.png"
                        
                        # Preprocess the cell for OCR
                        binary_cell = preprocess_cell(cell)
                        cv2.imwrite(str(cell_debug_dir / filename), binary_cell)
                    
                    print(f"  Saved 81 binary cells to: {cell_debug_dir}")
                
            except Exception as e:
                print(f"Error saving cells: {e}", file=sys.stderr)
                sys.exit(1)
        
        # Stage 3: OCR digit recognition (if requested)
        if args.ocr:
            print("Stage 3: OCR digit recognition...")
            try:
                # Test Tesseract installation
                from .ocr import test_tesseract_installation, get_tesseract_version
                print(f"[INFO] {get_tesseract_version()}")
                
                if not test_tesseract_installation():
                    print("Warning: Tesseract installation test failed. OCR may not work properly.")
                
                print(f"[OK] Tesseract OCR initialized")
                
                # Perform OCR on all cells or just first cell in debug mode
                if args.debug_first_cell:
                    print("Debug mode: Processing only first cell...")
                    from .ocr import ocr_cell, preprocess_cell
                    first_cell = cells[0]
                    print(f"First cell shape: {first_cell.shape}")
                    
                    # Preprocess and show details
                    processed = preprocess_cell(first_cell)
                    print(f"Processed cell shape: {processed.shape}")
                    
                    # Run OCR on first cell
                    digit = ocr_cell(first_cell, conf_thresh=args.tesseract_conf)
                    print(f"First cell result: {digit}")
                    
                    # Save debug images
                    cv2.imwrite(str(output_dir / f"{input_filename}_debug_first_cell_original.png"), first_cell)
                    cv2.imwrite(str(output_dir / f"{input_filename}_debug_first_cell_processed.png"), processed)
                    print(f"  Saved debug images: {output_dir / f'{input_filename}_debug_first_cell_original.png'}")
                    print(f"  Saved debug images: {output_dir / f'{input_filename}_debug_first_cell_processed.png'}")
                    
                    print("[OK] Debug mode completed - processed first cell only")
                    return
                else:
                    print("Recognizing digits...")
                    from .ocr import ocr_cells
                    digits = ocr_cells(cells, conf_thresh=args.tesseract_conf)
                    print(f"[OK] Recognized digits in {len(digits)} cells")
                
                # Convert to 9x9 grid
                grid = to_grid(digits)
                
                # Print the recognized grid
                print("\nRecognized Sudoku Grid:")
                print_grid(grid)
                
                # Save puzzle as JSON (9x9 list of lists)
                import json
                puzzle_json = output_dir / f"{input_filename}_puzzle.json"
                with open(puzzle_json, "w") as f:
                    f.write("[\n")
                    for i, row in enumerate(grid):
                        f.write("  " + json.dumps(row))
                        if i < len(grid) - 1:
                            f.write(",")
                        f.write("\n")
                    f.write("]")
                print(f"  Saved: {puzzle_json}")
                
                # Save puzzle as flat text (81-char string with zeros for blanks)
                puzzle_flat = output_dir / f"{input_filename}_puzzle_flat.txt"
                flat_string = "".join(str(digit) for digit in digits)
                with open(puzzle_flat, "w") as f:
                    f.write(flat_string)
                print(f"  Saved: {puzzle_flat}")
                
                print("[OK] OCR completed successfully!")
                
            except ImportError as e:
                print(f"Error: {e}", file=sys.stderr)
                print("Tesseract dependencies not available. Please install pytesseract.", file=sys.stderr)
                sys.exit(3)
            except Exception as e:
                print(f"Error during OCR: {e}", file=sys.stderr)
                sys.exit(3)
        
        # Save debug images if requested
        if args.debug:
            print("Saving debug images...")
            
            # Save contour mask
            cv2.imwrite(str(output_dir / f"{input_filename}_debug_contour_mask.png"), artifacts['contour_mask'])
            print(f"  Saved: {output_dir / f'{input_filename}_debug_contour_mask.png'}")
            
            # Save additional debug info
            debug_info = {
                'quad_points': artifacts['quad'].tolist() if artifacts['quad'] is not None else None,
                'warped_size': artifacts['warped'].shape,
                'binary_stats': {
                    'white_pixels': np.sum(artifacts['binary'] == 255),
                    'black_pixels': np.sum(artifacts['binary'] == 0),
                    'total_pixels': artifacts['binary'].size
                }
            }
            
            # Save debug info as text
            with open(output_dir / f"{input_filename}_debug_info.txt", "w") as f:
                f.write("Debug Information\n")
                f.write("================\n\n")
                f.write(f"Quad points: {debug_info['quad_points']}\n")
                f.write(f"Warped size: {debug_info['warped_size']}\n")
                f.write(f"Binary stats: {debug_info['binary_stats']}\n")
            
            print(f"  Saved: {output_dir / f'{input_filename}_debug_info.txt'}")
        
        print(f"\n[OK] Processing complete! Check output directory: {output_dir}")
        sys.exit(0)  # Success
        
    except Exception as e:
        print(f"Error during processing: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
