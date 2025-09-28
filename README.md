# Sudoku OCR

A Python-based Sudoku puzzle solver that uses computer vision and OCR to extract digits from Sudoku puzzle images.

## Features

- **Image Preprocessing**: Converts images to binary format with adaptive thresholding and morphological operations
- **Grid Detection**: Automatically detects Sudoku grid boundaries and applies perspective correction
- **Cell Extraction**: Splits the corrected grid into 81 individual cells
- **Digit Recognition**: Uses custom CNN-based heuristics for digit recognition
- **CLI Interface**: Command-line tool for processing Sudoku images
- **Comprehensive Testing**: Test suite with multiple difficulty levels

## Installation

### Prerequisites

- Python 3.8 or higher
- OpenCV (`opencv-python`)
- NumPy

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/sudoku_ocr.git
cd sudoku_ocr
```

2. Install dependencies:
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
    --ocr-conf 0.45 \
    --save-cells \
    --debug
```

### Command Line Options

- `--image`: Path to input Sudoku image
- `--out`: Output directory for artifacts
- `--size`: Size of warped grid (default: 450)
- `--pad`: Padding around cells (default: 4)
- `--apply-clahe`: Apply contrast enhancement
- `--ocr-conf`: OCR confidence threshold (default: 0.45)
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
│   └── ocr.py               # Digit recognition (CNN/heuristics)
├── data/
│   └── raw/                 # Test images and data
├── tests/                   # Test suite
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
   - Custom CNN-based heuristics
   - Feature extraction (aspect ratio, solidity, peaks, curves)
   - Classification with confidence scoring

## Testing

Run the comprehensive test suite:

```bash
python test_all_images.py
```

This will test the OCR pipeline against multiple difficulty levels and provide detailed accuracy metrics.

## Performance

Current performance on test dataset:

| Difficulty | Accuracy | Precision | Recall | F1-Score |
|------------|----------|-----------|--------|----------|
| Easy       | 63.2%    | 63.2%     | 100%   | 77.4%    |
| Medium     | 8.3%     | 25.0%     | 11.1%  | 15.4%    |
| Hard       | 47.8%    | 55.0%     | 78.6%   | 64.7%    |
| Overall    | 43.5%    | 56.1%     | 66.1%   | 60.7%    |

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
- Inspired by computer vision techniques for document analysis