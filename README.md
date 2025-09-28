# Sudoku OCR

A Python-based Sudoku puzzle solver that uses computer vision and OCR to extract digits from Sudoku puzzle images.

## Features

- **Image Preprocessing**: Converts images to binary format with adaptive thresholding and morphological operations
- **Grid Detection**: Automatically detects Sudoku grid boundaries and applies perspective correction
- **Cell Extraction**: Splits the corrected grid into 81 individual cells
- **Multi-Method OCR**: Uses Tesseract OCR with multiple fallback preprocessing methods for robust digit recognition
- **CLI Interface**: Command-line tool for processing Sudoku images with comprehensive options
- **Comprehensive Testing**: Automated test suite with ground truth validation and detailed metrics
- **Clean Output**: JSON and text output files with input filename-based naming

## Installation

### Prerequisites

- Python 3.8 or higher
- OpenCV (`opencv-python`)
- NumPy
- Tesseract OCR engine
- pytesseract (Python wrapper for Tesseract)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/sudoku_ocr.git
cd sudoku_ocr
```

2. Install Tesseract OCR:
   - **Windows**: `choco install tesseract`
   - **macOS**: `brew install tesseract`
   - **Linux**: `sudo apt-get install tesseract-ocr`

3. Install Python dependencies:
```bash
pip install -e .
```

## Usage

### Basic Usage

Process a Sudoku image and get the recognized grid:

```bash
python -m sudoku_ocr.cli --image data/raw/example.png --out output
```

### Advanced Options

```bash
python -m sudoku_ocr.cli --image data/raw/example.png --out output \
    --size 450 \
    --pad 4 \
    --apply-clahe \
    --tesseract-conf 0.45 \
    --save-cells \
    --debug
```

### Command Line Options

- `--image`: Path to input Sudoku image
- `--out`: Output directory for artifacts
- `--size`: Size of warped grid (default: 450)
- `--pad`: Padding around cells (default: 4)
- `--apply-clahe`: Apply contrast enhancement
- `--tesseract-conf`: Tesseract confidence threshold (default: 0.45)
- `--save-cells`: Save individual cell images
- `--debug`: Enable debug mode with extra artifacts
- `--debug-first-cell`: Process only the first cell for debugging
- `--no-ocr`: Skip OCR and only extract grid/cells

## Project Structure

```
sudoku_ocr/
├── src/sudoku_ocr/
│   ├── __init__.py          # Package initialization
│   ├── cli.py               # Command-line interface
│   ├── preprocess.py        # Image preprocessing functions
│   ├── grid.py              # Grid detection and warping
│   ├── cells.py             # Cell extraction
│   └── ocr.py               # Multi-method OCR with Tesseract
├── data/
│   └── raw/                 # Test images and ground truth data
├── tests/                   # Test suite
├── test_all_images.py       # Comprehensive test script
├── test_sudoku_json.py      # JSON output validation test
├── pyproject.toml           # Project configuration
├── requirements.txt         # Dependencies
└── README.md               # This file
```

## Architecture

### Image Processing Pipeline

1. **Preprocessing**: 
   - Convert BGR to grayscale
   - Apply Gaussian blur
   - Adaptive thresholding
   - Morphological operations

2. **Grid Detection**:
   - Find largest contour
   - Approximate to quadrilateral
   - Apply perspective transformation
   - Warp to square grid

3. **Cell Extraction**:
   - Split warped grid into 9×9 cells
   - Apply padding and resizing
   - Ensure minimum cell size (32×32)

4. **Digit Recognition**:
   - Multi-method Tesseract OCR approach
   - Enhanced preprocessing with larger image sizes
   - Alternative preprocessing methods (Otsu thresholding, simple thresholding)
   - Multiple PSM modes for robust recognition
   - Progressive confidence thresholds for fallback methods

## Testing

### Comprehensive Test Suite

Run the main test suite with visual grid comparison:

```bash
python test_all_images.py
```

### JSON Output Validation

Test the JSON output against ground truth data:

```bash
python test_sudoku_json.py data/raw/TestData.txt data/raw
```

Both test suites provide detailed accuracy metrics, error analysis, and performance statistics.

## Performance

Current performance on test dataset (6 test cases):

| Image | Accuracy | Precision | Recall | F1-Score | Status |
|-------|----------|-----------|--------|----------|--------|
| NYT-EASY-2025-09-27 | 100% | 100% | 100% | 100% | ✅ Perfect |
| NYT-MED-2025-09-27 | 0% | 0% | 0% | 0% | ⚠️ Challenging |
| NYT-HARD-2025-09-27 | 100% | 100% | 100% | 100% | ✅ Perfect |
| NYT-EASY-2025-09-28 | 100% | 100% | 100% | 100% | ✅ Perfect |
| NYT-MED-2025-09-28 | 100% | 100% | 100% | 100% | ✅ Perfect |
| NYT-HARD-2025-09-28 | 100% | 100% | 100% | 100% | ✅ Perfect |

**Overall Performance:**
- **Overall Accuracy**: 83.4%
- **Overall Precision**: 91.2%
- **Overall Recall**: 83.4%
- **Overall F1-Score**: 87.2%

**Success Rate**: 5 out of 6 images achieve perfect 100% accuracy

## Development

### Adding New Features

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

### Running Tests

```bash
pytest tests/
```

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with OpenCV for computer vision
- Uses NumPy for numerical computations
- Powered by Tesseract OCR for robust digit recognition
- Inspired by computer vision techniques for document analysis