# Medical Image Landmark Analysis System

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/opencv-v4.8+-green.svg)
![Scientific](https://img.shields.io/badge/quality-publication-gold.svg)
![Analysis](https://img.shields.io/badge/analysis-advanced-purple.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

🔬 **Advanced Scientific Analysis** | 📊 **Multi-Algorithm ROI Detection** | 🏥 **Medical Research Quality**

A comprehensive system for advanced landmark analysis and scientific visualization of medical imaging coordinates on chest X-ray images. This project provides cutting-edge tools for statistical analysis, heatmap generation, ROI detection, and publication-quality scientific exports designed for medical research workflows.

## Features

### Core Visualization
- **🎯 Coordinate Visualization**: Overlay precise medical coordinates on 299x299 resized images
- **📊 Multi-Dataset Support**: Process training, testing, and master datasets independently
- **🗂️ Automatic Organization**: Categorize images by medical condition (COVID-19, Normal, Viral Pneumonia)
- **🖼️ Grid Comparison**: Generate comparative visualization grids across categories

### 🔬 Advanced Scientific Analysis
- **🌡️ Heatmap Generation**: Spatial density analysis with 45+ individual and categorical heatmaps
- **📦 Multi-Algorithm ROI Detection**: 5 algorithms including MinMax (100% coverage guarantee)
- **📊 Advanced Statistical Analysis**: Comprehensive metrics, anomaly detection, and data validation
- **🎯 Bounding Box Analysis**: Scientific-quality ROI detection with detailed spatial metrics
- **🏥 Publication Quality Exports**: 300 DPI exports with scientific nomenclature and professional formatting
- **📈 Comprehensive Metrics**: Area, coverage, efficiency, density, and spatial distribution analysis

## Project Structure

```
medical-image-coordinates/
├── scripts/                              # Processing and analysis scripts
│   ├── visualizar_coordenadas_299x299.py       # Basic coordinate visualization
│   ├── organizar_imagenes_por_categoria.py     # Image organization system
│   ├── analizar_estadisticas_coordenadas.py    # 🆕 Statistical analysis engine
│   ├── analizar_landmarks_heatmaps.py          # 🆕 Heatmap generation system
│   ├── generar_bounding_boxes_landmarks.py     # 🆕 ROI detection and analysis
│   ├── procesar_coordenadas.py                 # Coordinate processing utilities
│   └── escalar_coordenadas.py                  # Coordinate scaling functions
├── data/                             # Data directory (see setup instructions)
│   ├── coordenadas_299x299/         # Scaled coordinate CSV files
│   ├── coordenadas/                 # Original coordinate data
│   ├── indices/                     # Dataset index files
│   └── COVID-19_Radiography_Dataset/ # Medical image dataset
├── visualizations/                   # Basic coordinate overlays
│   └── coordenadas_299x299_overlays/ # Organized visualization outputs
├── heatmaps_landmarks/               # 🆕 Advanced heatmap analysis system
│   ├── individuales/                # Individual landmark heatmaps (45 images)
│   ├── por_categoria/               # Category-based heatmaps (45 images)
│   ├── comparativos/                # Comparative overview grids (3 grids)
│   ├── datos/matrices_densidad/     # NumPy density matrices (60+ files)
│   └── estadisticas/                # Statistical summaries and reports
├── bounding_boxes_landmarks/         # 🆕 Scientific ROI analysis system
│   ├── individuales/                # Scientific visualizations (30+ files)
│   ├── comparativos/                # Scientific overview grids
│   ├── estadisticas/                # ROI metrics and detailed analysis
│   └── [future expansion directories]
├── estadisticas_coordenadas/         # 🆕 Comprehensive statistical analysis
│   ├── estadisticas_*.json          # Detailed statistical reports
│   ├── reporte_anomalias_*.txt      # Anomaly detection reports
│   ├── resumen_ejecutivo_*.txt      # Executive summaries
│   └── resumen_imagenes_*.csv       # Image-level statistical summaries
├── models/                           # Model storage directory
├── results/                          # Analysis results
├── src/                              # Additional source code
├── CLAUDE.md                         # Comprehensive technical documentation
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

#### Basic Coordinate Visualization
```bash
# Process training dataset with coordinate overlay
python scripts/visualizar_coordenadas_299x299.py --dataset entrenamiento

# Process with image limit
python scripts/visualizar_coordenadas_299x299.py --dataset prueba --limite 50

# Process specific images
python scripts/visualizar_coordenadas_299x299.py --imagenes COVID-269 Normal-1023

# Organize processed images by medical category
python scripts/organizar_imagenes_por_categoria.py --dry-run  # Preview
python scripts/organizar_imagenes_por_categoria.py           # Execute
```

#### 🔬 Advanced Scientific Analysis (Recommended Workflow)
```bash
# 1. Generate comprehensive statistical analysis
python scripts/analizar_estadisticas_coordenadas.py --dataset entrenamiento --detalle

# 2. Create spatial density heatmaps for all landmarks
python scripts/analizar_landmarks_heatmaps.py --datasets entrenamiento --landmarks todos

# 3. Perform scientific ROI detection with guaranteed coverage
python scripts/generar_bounding_boxes_landmarks.py --metodos minmax --datasets entrenamiento prueba

# 4. Generate publication-quality comparative analysis
python scripts/analizar_landmarks_heatmaps.py --por-categoria --landmarks todos
```

## 🔬 Advanced Landmark Analysis

### Scientific Heatmap Generation

Generate spatial density analysis with publication-quality heatmaps:

```bash
# Generate all landmark heatmaps for training dataset
python scripts/analizar_landmarks_heatmaps.py --datasets entrenamiento --landmarks todos

# Generate specific landmarks with comparative analysis
python scripts/analizar_landmarks_heatmaps.py --datasets entrenamiento prueba --landmarks 1 5 10

# Generate category-based heatmaps (COVID, Normal, Viral Pneumonia)
python scripts/analizar_landmarks_heatmaps.py --por-categoria --landmarks todos
```

### ROI Detection and Bounding Box Analysis

Advanced region-of-interest detection with multiple algorithms:

```bash
# Scientific ROI analysis with MinMax method (recommended - 100% coverage)
python scripts/generar_bounding_boxes_landmarks.py --metodos minmax --datasets entrenamiento prueba

# Compare multiple detection algorithms
python scripts/generar_bounding_boxes_landmarks.py --metodos minmax percentile statistical --datasets entrenamiento

# Individual landmark analysis with detailed metrics
python scripts/generar_bounding_boxes_landmarks.py --metodos minmax --landmarks 1 4 8 --datasets entrenamiento --verbose
```

### Statistical Analysis and Anomaly Detection

Comprehensive statistical analysis with detailed reporting:

```bash
# Complete statistical analysis with anomaly detection
python scripts/analizar_estadisticas_coordenadas.py --dataset entrenamiento --detalle

# Quick statistical overview for all datasets
python scripts/analizar_estadisticas_coordenadas.py --dataset maestro --resumen

# Generate executive summary reports
python scripts/analizar_estadisticas_coordenadas.py --dataset entrenamiento --ejecutivo
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

After processing, your analysis results will be organized as:

```
📁 Basic Visualizations
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

🔬 Scientific Analysis Results
heatmaps_landmarks/
├── individuales/                     # Individual landmark heatmaps (45 files)
├── por_categoria/                    # Medical category heatmaps (45 files)
├── comparativos/                     # Comparative grids (3 overview files)
├── datos/matrices_densidad/          # NumPy density matrices (60+ files)
└── estadisticas/                     # Statistical summaries and reports

📊 ROI Detection Results
bounding_boxes_landmarks/
├── individuales/                     # Scientific ROI visualizations (30+ files)
│   ├── landmark_XX_dataset_minmax_bbox.png        # Publication-quality images
│   └── landmark_XX_dataset_minmax_bbox_stats.json # Detailed metrics
├── comparativos/                     # Scientific overview grids
├── estadisticas/                     # ROI metrics and analysis reports
└── reportes/                         # Consolidated analysis reports

📈 Statistical Analysis
estadisticas_coordenadas/
├── estadisticas_*.json               # Comprehensive statistical data
├── reporte_anomalias_*.txt          # Anomaly detection reports
├── resumen_ejecutivo_*.txt          # Executive summaries
└── resumen_imagenes_*.csv           # Image-level statistical analysis
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

## 🔬 Scientific Analysis Features

### Advanced Algorithms
- **MinMax ROI Detection**: 100% coordinate coverage guarantee for medical research
- **Multi-Algorithm Support**: Percentile, Statistical, Contours, and Hybrid methods
- **Spatial Density Analysis**: Scientific-quality heatmap generation with perceptually uniform colormaps
- **Anomaly Detection**: Statistical outlier identification with detailed reporting
- **Publication Standards**: 300 DPI exports with scientific nomenclature and metadata

### ROI Detection Methods
1. **MinMax Method** ⭐ (Recommended)
   - **Coverage**: 100% guaranteed coordinate inclusion
   - **Use Case**: Critical medical analysis where no data can be lost
   - **Algorithm**: Absolute bounds with safety margins

2. **Percentile Method**
   - **Coverage**: ~95% (outlier filtering)
   - **Use Case**: Robust statistical analysis
   - **Algorithm**: 5th-95th percentile bounds

3. **Statistical Method**
   - **Coverage**: ~90% of coordinates
   - **Use Case**: Classical statistical analysis
   - **Algorithm**: Mean ± 2 standard deviations

### Scientific Export Quality
- **Resolution**: 300 DPI for publication requirements
- **Color Schemes**: Plasma and Viridis (perceptually uniform, colorblind-friendly)
- **Nomenclature**: Medical-grade landmark identification (L01-L15)
- **Metadata**: Complete provenance and analysis parameters
- **Format Standards**: PNG with comprehensive JSON statistical reports

## Technical Details

### Dependencies
- **pandas**: Data manipulation and CSV processing
- **numpy**: Numerical operations and array handling  
- **opencv-python**: Image processing and visualization
- **matplotlib**: Grid generation and scientific plotting
- **scipy**: Advanced statistical analysis (for new features)
- **json**: Structured data export and metadata

### Performance Characteristics
- **Basic Processing**: ~1-2 images per second
- **Heatmap Generation**: ~10-15 landmarks per minute
- **ROI Detection**: ~5-8 landmarks per minute (with full metrics)
- **Memory Usage**: Optimized for batch processing with NumPy arrays
- **Output Quality**: 299x299 pixel precision with scientific accuracy
- **Statistical Analysis**: Real-time anomaly detection and validation

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

## 🎯 System Evolution and Methodology

This project has evolved from basic coordinate visualization to a comprehensive scientific analysis platform:

1. **Phase 1**: Basic coordinate overlay visualization (299x299 scaling)
2. **Phase 2**: Dataset organization and medical categorization  
3. **Phase 3**: Statistical analysis and anomaly detection
4. **Phase 4**: Spatial density heatmap generation
5. **Phase 5**: Scientific ROI detection with multiple algorithms
6. **Phase 6**: Publication-quality exports and scientific standards

### Research Applications
- **Medical Image Analysis**: Landmark detection and spatial analysis
- **Algorithm Validation**: Comparative analysis of detection methods
- **Statistical Research**: Coordinate distribution and anomaly studies
- **Publication Workflows**: Scientific-quality visualization generation

---

**Note**: This system is designed for research and educational purposes. Medical image analysis should always be validated by qualified healthcare professionals. The advanced analysis features are specifically designed to meet scientific publication standards and support medical research workflows.