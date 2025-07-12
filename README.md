# Medical Image Coordinate Visualization System

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/opencv-v4.8+-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

A comprehensive system for visualizing and processing medical imaging coordinates on chest X-ray images. This project provides automated tools for coordinate overlay, image organization, and dataset management for medical image analysis workflows.

## Features

- **🎯 Coordinate Visualization**: Overlay precise medical coordinates on 299x299 resized images
- **📊 Multi-Dataset Support**: Process training, testing, and master datasets independently
- **🗂️ Automatic Organization**: Categorize images by medical condition (COVID-19, Normal, Viral Pneumonia)
- **📈 Statistical Analysis**: Generate detailed processing reports and statistics
- **🔍 Quality Validation**: Integrity checking and validation for all processed images
- **🖼️ Grid Comparison**: Generate comparative visualization grids across categories

## Project Structure

```
medical-image-coordinates/
├── scripts/                          # Core processing scripts
│   ├── visualizar_coordenadas_299x299.py    # Main visualization engine
│   ├── organizar_imagenes_por_categoria.py  # Automatic image organization
│   ├── procesar_coordenadas.py              # Coordinate processing utilities
│   └── escalar_coordenadas.py               # Coordinate scaling functions
├── data/                             # Data directory (see setup instructions)
│   ├── coordenadas_299x299/         # Scaled coordinate CSV files
│   └── COVID-19_Radiography_Dataset/ # Medical image dataset
├── models/                           # Model storage directory
├── results/                          # Analysis results
├── src/                              # Additional source code
├── CLAUDE.md                         # Detailed technical documentation
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

## Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/rhaink/medical-image-coordinates.git
   cd medical-image-coordinates
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Setup data directory**
   ```bash
   # Download the COVID-19 Radiography Dataset
   # Place coordinate CSV files in data/coordenadas_299x299/
   # Place medical images in data/COVID-19_Radiography_Dataset/
   ```

### Basic Usage

#### Process Training Dataset
```bash
python scripts/visualizar_coordenadas_299x299.py --dataset entrenamiento
```

#### Process with Limit
```bash
python scripts/visualizar_coordenadas_299x299.py --dataset prueba --limite 50
```

#### Process Specific Images
```bash
python scripts/visualizar_coordenadas_299x299.py --imagenes COVID-269 Normal-1023
```

#### Organize Processed Images
```bash
# Preview organization (dry-run)
python scripts/organizar_imagenes_por_categoria.py --dry-run

# Execute organization
python scripts/organizar_imagenes_por_categoria.py
```

## Advanced Usage

### Dataset Processing Options

The main visualization script supports multiple datasets and configuration options:

```bash
# Process complete master dataset
python scripts/visualizar_coordenadas_299x299.py --dataset maestro

# Process without generating comparison grid
python scripts/visualizar_coordenadas_299x299.py --dataset entrenamiento --sin-grid

# Verbose organization with custom source
python scripts/organizar_imagenes_por_categoria.py --source /custom/path --verbose
```

### Output Structure

After processing, your visualization directory will be organized as:

```
visualizations/coordenadas_299x299_overlays/
├── entrenamiento/                    # Training dataset (640 images)
│   ├── covid/           (164 images)
│   ├── normal/          (328 images)
│   └── viral-pneumonia/ (152 images)
├── prueba/                           # Test dataset (160 images)
│   ├── covid/           (40 images)
│   ├── normal/          (72 images)
│   └── viral-pneumonia/ (48 images)
├── grid_comparacion.png              # Comparative visualization grid
└── reporte_*.json                    # Processing statistics
```

## Data Requirements

### Coordinate Files
- **Format**: CSV with 15 coordinate pairs per image
- **Structure**: `index,x1,y1,x2,y2,...,x15,y15,image_id`
- **Scale**: Coordinates scaled to 299x299 pixel space

### Medical Images
- **Categories**: COVID-19, Normal, Viral Pneumonia
- **Format**: PNG images
- **Processing**: Automatically resized to 299x299 pixels

### Expected Datasets
- **Master**: 999 images (all categories)
- **Training**: 640 images (for ML training)
- **Test**: 160 images (for validation)

## Visualization Specifications

### Visual Style
- **Coordinate Points**: Green circles (radius 3px solid + 5px border)
- **Point Numbers**: Red text (font size 0.3, position offset +7,−7)
- **Background**: Original medical image resized to 299x299
- **Output Format**: PNG with naming convention `{IMAGE_ID}_anotada.png`

### Statistical Reports
- Processing statistics by category
- Integrity validation results
- File organization summaries
- JSON format for programmatic access

## Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run validation tests
5. Submit a pull request

### Code Style
- Follow PEP 8 conventions
- Use meaningful variable names
- Include docstrings for functions
- Maintain UTF-8 encoding with headers

## Validation and Quality Assurance

### File Integrity Checks
```bash
# Verify total processed images
find visualizations/coordenadas_299x299_overlays -name "*.png" | wc -l

# Check category distributions
find visualizations/coordenadas_299x299_overlays/entrenamiento/covid -name "*.png" | wc -l
```

### Common Troubleshooting

1. **"File not found" errors**: Verify data directory structure
2. **"Image not found" warnings**: Check medical image dataset placement
3. **Import errors**: Ensure script execution from project root
4. **Permission errors**: Verify write permissions for visualization directory

## Technical Details

### Dependencies
- **pandas**: Data manipulation and CSV processing
- **numpy**: Numerical operations and array handling
- **opencv-python**: Image processing and visualization
- **matplotlib**: Grid generation and plotting

### Performance Characteristics
- **Processing Speed**: ~1-2 images per second
- **Memory Usage**: Optimized for batch processing
- **Output Quality**: 299x299 pixel precision
- **Coordinate Accuracy**: Subpixel precision maintained

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- COVID-19 Radiography Dataset contributors
- Medical imaging research community
- Open source computer vision libraries

## Support

For questions, issues, or contributions:
1. Check existing [Issues](https://github.com/rhaink/medical-image-coordinates/issues)
2. Create a new issue with detailed description
3. Include system information and error logs
4. Follow the issue template guidelines

---

**Note**: This system is designed for research and educational purposes. Medical image analysis should always be validated by qualified healthcare professionals.