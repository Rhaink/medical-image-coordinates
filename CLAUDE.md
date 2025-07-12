# Proyecto de Visualización de Coordenadas Médicas 299x299

## 📋 Descripción del Proyecto

Sistema de visualización y procesamiento de coordenadas médicas para imágenes de rayos X de COVID-19, superpuestas en imágenes redimensionadas a 299x299 píxeles. El proyecto permite procesar datasets de entrenamiento y prueba, organizando automáticamente las imágenes por categorías médicas.

## 🗂️ Estructura del Proyecto

```
matching/
├── data/
│   ├── COVID-19_Radiography_Dataset/        # Imágenes originales
│   │   ├── COVID/images/
│   │   ├── Normal/images/
│   │   └── Viral Pneumonia/images/
│   ├── coordenadas_299x299/                 # Coordenadas escaladas
│   │   ├── coordenadas_maestro_299x299.csv     (999 registros)
│   │   ├── coordenadas_entrenamiento_299x299.csv (640 registros)
│   │   └── coordenadas_prueba_299x299.csv       (160 registros)
│   └── coordenadas/                         # Coordenadas originales
├── scripts/
│   ├── visualizar_coordenadas.py           # Script original
│   ├── visualizar_coordenadas_299x299.py   # Script mejorado 
│   ├── organizar_imagenes_por_categoria.py # Organizador automático
│   ├── analizar_estadisticas_coordenadas.py # Análisis estadístico (NUEVO)
│   ├── analizar_landmarks_heatmaps.py      # Generación de heatmaps (NUEVO)
│   ├── generar_bounding_boxes_landmarks.py # Sistema bounding boxes (NUEVO)
│   ├── procesar_coordenadas.py
│   └── escalar_coordenadas.py
├── visualizations/
│   └── coordenadas_299x299_overlays/       # Imágenes procesadas básicas
│       ├── entrenamiento/                  # 640 imágenes
│       │   ├── covid/        (164 imágenes)
│       │   ├── normal/       (328 imágenes)
│       │   └── viral-pneumonia/ (152 imágenes)
│       ├── prueba/                         # 160 imágenes
│       │   ├── covid/        (40 imágenes)
│       │   ├── normal/       (72 imágenes)
│       │   └── viral-pneumonia/ (48 imágenes)
│       ├── grid_comparacion.png
│       └── reportes*.json
├── heatmaps_landmarks/                      # Sistema de análisis de heatmaps (NUEVO)
│   ├── individuales/                       # Heatmaps por landmark (45 imágenes)
│   ├── por_categoria/                      # Heatmaps por categoría médica (45 imágenes)
│   ├── comparativos/                       # Grids comparativos por dataset (3 grids)
│   ├── datos/matrices_densidad/            # Matrices NumPy de densidad (60 archivos)
│   └── estadisticas/                       # Estadísticas y resúmenes CSV/JSON
├── bounding_boxes_landmarks/                # Sistema de detección ROI científico (NUEVO)
│   ├── individuales/                       # Visualizaciones científicas (15 PNG + 15 JSON)
│   ├── comparativos/                       # Grid científico comparativo
│   ├── estadisticas/                       # Métricas y análisis estadísticos
│   ├── datos/roi_masks/                    # Máscaras de ROI (futuro)
│   └── por_categoria/                      # Análisis por categoría médica (futuro)
└── CLAUDE.md                              # Este archivo
```

## 🛠️ Scripts Disponibles

### 1. `visualizar_coordenadas_299x299.py`
**Propósito**: Script principal para generar imágenes anotadas con coordenadas superpuestas.

**Funcionalidades**:
- Procesamiento de datasets específicos (maestro/entrenamiento/prueba)
- Visualización con estilo uniforme (puntos verdes, números rojos)
- Redimensionamiento automático a 299x299 píxeles
- Generación de estadísticas detalladas por categoría
- Grid de comparación opcional
- Procesamiento de imágenes específicas

**Uso**:
```bash
# Procesar dataset completo
python scripts/visualizar_coordenadas_299x299.py --dataset entrenamiento

# Procesar con límite
python scripts/visualizar_coordenadas_299x299.py --dataset prueba --limite 50

# Procesar imágenes específicas
python scripts/visualizar_coordenadas_299x299.py --imagenes COVID-269 Normal-1023

# Sin grid de comparación
python scripts/visualizar_coordenadas_299x299.py --dataset maestro --sin-grid
```

**Salida**: 
- Imágenes anotadas: `{COVID_ID}_anotada.png`
- Grid de comparación: `grid_comparacion.png`
- Reporte estadístico: `reporte_estadisticas.json`

### 2. `organizar_imagenes_por_categoria.py`
**Propósito**: Organizar imágenes anotadas en carpetas por categoría médica.

**Funcionalidades**:
- Detección automática de categorías por nombre de archivo
- Modo dry-run para preview
- Validación de integridad completa
- Estadísticas detalladas
- Manejo seguro de archivos especiales

**Uso**:
```bash
# Organizar imágenes (modo preview)
python scripts/organizar_imagenes_por_categoria.py --dry-run

# Organizar imágenes (ejecución real)
python scripts/organizar_imagenes_por_categoria.py

# Organizar directorio específico con detalle
python scripts/organizar_imagenes_por_categoria.py --source /ruta/custom --verbose
```

**Salida**:
- Carpetas: `covid/`, `normal/`, `viral-pneumonia/`
- Reporte: `reporte_organizacion.json`

## 📊 Datos y Coordenadas

### Formato de Coordenadas
Cada archivo CSV contiene coordenadas escaladas a 299x299:
- **Estructura**: `indice,x1,y1,x2,y2,...,x15,y15,COVID_ID`
- **Total por imagen**: 15 puntos de coordenadas
- **Rango válido**: 0-299 píxeles

### Categorías Médicas
1. **COVID-19**: Imágenes con infección por coronavirus
2. **Normal**: Radiografías sin patologías
3. **Viral Pneumonia**: Neumonía viral (no COVID)

### Datasets Disponibles
- **Maestro**: 999 imágenes (todas las categorías)
- **Entrenamiento**: 640 imágenes (para ML)
- **Prueba**: 160 imágenes (para validación)

## 🎯 Estado Actual

### ✅ Completado
- **800 imágenes procesadas** con coordenadas 299x299
- **Organización por datasets**: entrenamiento (640) y prueba (160)
- **Categorización médica**: COVID/Normal/Viral Pneumonia
- **Scripts funcionales**: visualización y organización
- **Estilo uniforme**: compatible con script original
- **Validación completa**: sin archivos perdidos o dañados

### 📈 Estadísticas Finales
```
ENTRENAMIENTO (640 imágenes):
├── COVID-19: 164 imágenes
├── Normal: 328 imágenes
└── Viral Pneumonia: 152 imágenes

PRUEBA (160 imágenes):
├── COVID-19: 40 imágenes
├── Normal: 72 imágenes
└── Viral Pneumonia: 48 imágenes

TOTAL: 800 imágenes con 12,000 coordenadas visualizadas
```

## 🔧 Comandos Útiles

### Verificación de Archivos
```bash
# Contar imágenes por dataset
find visualizations/coordenadas_299x299_overlays/entrenamiento -name "*.png" | wc -l
find visualizations/coordenadas_299x299_overlays/prueba -name "*.png" | wc -l

# Verificar estructura
ls -la visualizations/coordenadas_299x299_overlays/*/

# Estadísticas por categoría
find visualizations/coordenadas_299x299_overlays/entrenamiento/covid -name "*.png" | wc -l
```

### Procesamiento de Nuevos Datos
```bash
# Si se agregan nuevas coordenadas, procesar dataset maestro
python scripts/visualizar_coordenadas_299x299.py --dataset maestro --sin-grid

# Organizar nuevas imágenes
python scripts/organizar_imagenes_por_categoria.py --dry-run
python scripts/organizar_imagenes_por_categoria.py
```

## 🎨 Especificaciones Técnicas

### Estilo Visual
- **Puntos**: Verde (0, 255, 0) - Radio 3px sólido + borde 5px
- **Números**: Rojo (0, 0, 255) - Fuente 0.3, posición (x+7, y-7)
- **Fondo**: Imagen original redimensionada a 299x299
- **Formato salida**: PNG con nombre `{COVID_ID}_anotada.png`

### Convenciones de Código
- **Encoding**: UTF-8 con headers `# -*- coding: utf-8 -*-`
- **Paths**: Uso de `pathlib.Path` para compatibilidad
- **Imports**: Agrupados y ordenados
- **Nomenclatura**: snake_case para funciones, PascalCase para clases
- **Documentación**: Docstrings detallados en español

### Estructura de Archivos de Salida
```
{COVID_ID}_anotada.png          # Imagen con coordenadas superpuestas
grid_comparacion.png            # Grid 4x3 comparativo por categorías
reporte_estadisticas.json       # Estadísticas de procesamiento
reporte_organizacion.json       # Reporte de organización por carpetas
```

## 🚀 Próximos Pasos Sugeridos

1. **Análisis de Calidad**: Validar precisión de coordenadas superpuestas
2. **Optimización**: Batch processing para datasets grandes
3. **Exportación**: Generar datasets listos para modelos ML
4. **Visualización Avanzada**: Heatmaps o overlays adicionales
5. **Automatización**: Scripts de pipeline completo

## 🔍 Solución de Problemas

### Errores Comunes
1. **"Archivo no encontrado"**: Verificar rutas en `data/coordenadas_299x299/`
2. **"Imagen no encontrada"**: Confirmar estructura en `data/COVID-19_Radiography_Dataset/`
3. **"Import Error"**: Ejecutar desde directorio raíz del proyecto
4. **"Permission denied"**: Verificar permisos de escritura en `visualizations/`

### Validación de Integridad
```bash
# Verificar total de archivos esperados
echo "Entrenamiento: $(find visualizations/coordenadas_299x299_overlays/entrenamiento -name "*.png" | wc -l)/640"
echo "Prueba: $(find visualizations/coordenadas_299x299_overlays/prueba -name "*.png" | wc -l)/160"
echo "Total: $(find visualizations/coordenadas_299x299_overlays -name "*.png" | wc -l)/800"
```

## 📝 Notas para Futuras Sesiones

### Contexto del Proyecto
- Sistema de visualización médica para análisis de radiografías
- Coordenadas representan puntos de interés médico en imágenes
- Organización lista para entrenamiento de modelos de machine learning
- Scripts optimizados para procesamiento batch y organización automática

### Estado Técnico
- Todos los scripts funcionan correctamente desde `scripts/`
- Rutas ajustadas para ejecución desde directorio raíz
- Compatibilidad con script original mantenida
- Validación de integridad implementada en todos los procesos

### Decisiones de Diseño
- Estilo visual uniforme (verde/rojo) para consistencia
- Separación clara entrenamiento/prueba para ML
- Organización por categorías médicas para análisis
- Reportes automáticos para trazabilidad

## 🔬 Sistema de Análisis Avanzado de Landmarks

### Análisis de Bounding Boxes para Landmarks Anatómicos
**Script Principal**: `scripts/generar_bounding_boxes_landmarks.py`

Sistema avanzado para detección y análisis de regiones de interés (ROI) basado en coordenadas de landmarks anatómicos en radiografías médicas. Incluye múltiples algoritmos de detección con análisis científico de calidad para publicación.

#### Métodos de Detección Implementados
1. **MinMax Method** ⭐ (Método Recomendado)
   - **Cobertura**: 100% garantizada de todas las coordenadas
   - **Algoritmo**: Límites absolutos con margen de seguridad
   - **Uso**: Análisis científico donde no se puede perder información

2. **Percentile Method**
   - **Cobertura**: ~95% de coordenadas (filtrado de outliers)
   - **Algoritmo**: Percentiles 5-95 con margen adaptativo
   - **Uso**: Análisis estadístico robusto contra anomalías

3. **Statistical Method**
   - **Cobertura**: ~90% de coordenadas
   - **Algoritmo**: Media ± 2 desviaciones estándar
   - **Uso**: Análisis estadístico clásico

4. **Contours Method**
   - **Cobertura**: Variable según densidad
   - **Algoritmo**: Detección de contornos por densidad
   - **Uso**: Análisis morfológico de distribuciones

5. **Hybrid Method**
   - **Cobertura**: Variable (combinación adaptativa)
   - **Algoritmo**: Selección automática según distribución
   - **Uso**: Análisis adaptativo automático

#### Funcionalidades Principales
- **Análisis Individual**: 15 landmarks por separado con métricas completas
- **Análisis Comparativo**: Grid overview de todos los landmarks
- **Exportación Científica**: PNG 300 DPI con metadatos completos
- **Estadísticas Detalladas**: JSON con métricas ROI y distribución espacial
- **Visualización Profesional**: Estilo científico con nomenclatura médica

#### Estilo Científico Implementado
- **Colormaps**: Plasma (perceptualmente uniforme)
- **Fondo**: Negro para contraste óptimo
- **Puntos**: Rojo/naranja alternados para visibilidad
- **Tipografía**: Sans-serif profesional
- **Títulos**: Nomenclatura científica estandarizada
- **Métricas**: Tablas organizadas con valores precisos

### Uso del Sistema de Bounding Boxes

```bash
# Análisis completo de todos los landmarks (método recomendado)
python scripts/generar_bounding_boxes_landmarks.py --metodos minmax --datasets entrenamiento prueba

# Análisis específico de landmarks individuales
python scripts/generar_bounding_boxes_landmarks.py --metodos minmax --landmarks 1 4 8 --datasets entrenamiento

# Comparación de métodos (análisis metodológico)
python scripts/generar_bounding_boxes_landmarks.py --metodos minmax percentile statistical --datasets entrenamiento

# Análisis rápido con visualización mínima
python scripts/generar_bounding_boxes_landmarks.py --metodos minmax --sin-grid --limite 100
```

#### Archivos de Salida del Sistema
```
bounding_boxes_landmarks/
├── individuales/                           # Análisis por landmark
│   ├── landmark_01_entrenamiento_minmax_bbox.png    # Visualización científica
│   ├── landmark_01_entrenamiento_minmax_bbox_stats.json  # Métricas detalladas
│   └── ...                               # (15 landmarks × datasets × métodos)
├── comparativos/                          # Análisis overview
│   ├── grid_bboxes_entrenamiento_minmax_YYYYMMDD_HHMMSS.png
│   └── grid_bboxes_prueba_minmax_YYYYMMDD_HHMMSS.png
└── reportes/                             # Reportes consolidados
    └── reporte_bboxes_YYYYMMDD_HHMMSS.json
```

#### Métricas Científicas Calculadas
**ROI Metrics**:
- Area (píxeles)
- Coverage (% coordenadas incluidas)
- Efficiency (relación área/contenido)
- Density (coordenadas/píxel)

**Spatial Stats**:
- Centroid (X, Y)
- Dimensions (ancho × alto)
- Method validation
- Point distribution analysis

## 📊 Evolución y Decisiones Técnicas

### Cronología del Desarrollo
1. **Fase Inicial**: Visualización básica de coordenadas 299x299
2. **Fase de Organización**: Categorización por datasets médicos
3. **Fase de Análisis**: Implementación de heatmaps de densidad
4. **Fase de Bounding Boxes**: Múltiples algoritmos de detección ROI
5. **Fase Científica**: Elevación a estándares de publicación

### Decisiones Técnicas Críticas

#### 1. Selección del Método MinMax
- **Problema**: Otros métodos perdían coordenadas importantes
- **Solución**: MinMax garantiza 100% de cobertura
- **Justificación**: En análisis médico no se puede perder información

#### 2. Estilo Visual Científico
- **Problema Inicial**: Visualizaciones básicas inadecuadas para publicación
- **Evolución**: Colores → Heatmaps → Científico profesional
- **Error Crítico**: Eliminación accidental de heatmaps (revertido)
- **Solución Final**: Balance entre contexto (heatmap) y precisión (puntos)

#### 3. Estructura de Datos
- **Formato**: NumPy arrays para eficiencia computacional
- **Resolución**: 299x299 píxeles (estándar para modelos médicos)
- **Coordenadas**: 15 landmarks × N imágenes
- **Validación**: Verificación de integridad en cada procesamiento

#### 4. Organización de Archivos
- **Principio**: Separación clara por propósito y método
- **Timestamps**: Versionado automático para trazabilidad
- **Metadatos**: JSON completos para reproducibilidad científica

## ⚙️ Configuración Científica y Estándares

### Parámetros de Calidad Científica
```python
# Configuración de exportación para publicación
SCIENTIFIC_CONFIG = {
    'dpi': 300,                    # Resolución para publicación
    'format': 'png',               # Formato sin pérdida
    'bbox_inches': 'tight',        # Recorte óptimo
    'pad_inches': 0.1,             # Margen estándar
    'facecolor': 'black',          # Fondo contrastante
    'metadata': {                  # Metadatos completos
        'Title': 'Landmark Analysis',
        'Author': 'Medical AI System',
        'Subject': 'ROI Detection',
        'Creator': 'Python matplotlib'
    }
}
```

### Estándares de Nomenclatura
- **Landmarks**: `L01-L15` (numeración médica estándar)
- **Datasets**: `entrenamiento`, `prueba`, `maestro`
- **Métodos**: `minmax`, `percentile`, `statistical`, `contours`, `hybrid`
- **Archivos**: `landmark_XX_dataset_method_bbox.png`
- **Timestamps**: `YYYYMMDD_HHMMSS` (ISO-compatible)

### Colormaps Científicos
- **Principal**: `plasma` (perceptualmente uniforme)
- **Alternativa**: `viridis` (para daltonismo)
- **Puntos**: Rojo (#FF0000) y Naranja (#FF8000) alternados
- **Bounding Box**: Verde lima (#00FF00) con alpha 0.8

## 📁 Estado Actual y Archivos Críticos

### Archivos Más Importantes
1. **`scripts/generar_bounding_boxes_landmarks.py`**
   - Script principal del sistema de análisis
   - Contiene todos los algoritmos de detección
   - Función crítica: `detectar_bbox_minmax()`
   - Última modificación: Estándares científicos implementados

2. **`bounding_boxes_landmarks/individuales/landmark_*_minmax_*.png`**
   - 30 visualizaciones científicas (15 landmarks × 2 datasets)
   - Timestamp más reciente: `20250712_023513`
   - Calidad de publicación científica

3. **`bounding_boxes_landmarks/comparativos/grid_bboxes_*_minmax_*.png`**
   - Análisis overview de todos los landmarks
   - Visualización de distribución espacial completa
   - Formato científico profesional

4. **`data/coordenadas_299x299/`**
   - Coordenadas fuente escaladas a 299x299
   - 999 imágenes maestro, 640 entrenamiento, 160 prueba
   - Base de datos principal del proyecto

### Estado de Integridad
- **Archivos procesados**: 800 imágenes (640 entrenamiento + 160 prueba)
- **Landmarks analizados**: 15 landmarks completos
- **Métodos validados**: MinMax con 100% cobertura
- **Calidad**: Estándar científico para publicación
- **Documentación**: Completa con metadatos JSON

### Configuración de Entorno
```bash
# Dependencias principales
pip install numpy matplotlib opencv-python pillow

# Estructura de directorios validada
mkdir -p bounding_boxes_landmarks/{individuales,comparativos,reportes}
mkdir -p heatmaps_landmarks/{individuales,comparativos}
mkdir -p visualizations/coordenadas_299x299_overlays/{entrenamiento,prueba}
```

## 🚀 Comandos de Uso y Workflows

### Workflow Recomendado para Análisis Científico
```bash
# 1. Análisis completo con método óptimo
python scripts/generar_bounding_boxes_landmarks.py --metodos minmax --datasets entrenamiento prueba

# 2. Verificación de resultados
find bounding_boxes_landmarks/individuales -name "*minmax_bbox.png" | wc -l  # Debe ser 30

# 3. Análisis de métricas específicas
python -c "
import json
with open('bounding_boxes_landmarks/individuales/landmark_01_entrenamiento_minmax_bbox_stats.json') as f:
    stats = json.load(f)
    print(f'Coverage: {stats[\"bbox\"][\"cobertura\"]}%')
"

# 4. Validación de integridad de datos
python scripts/generar_bounding_boxes_landmarks.py --verificar-datos
```

### Comandos de Mantenimiento
```bash
# Limpiar archivos temporales
find . -name "*.pyc" -delete
find . -name "__pycache__" -type d -exec rm -rf {} +

# Verificar estructura de proyecto
ls -la bounding_boxes_landmarks/individuales/ | head -10
ls -la bounding_boxes_landmarks/comparativos/

# Estadísticas rápidas
echo "Landmarks individuales: $(find bounding_boxes_landmarks/individuales -name "*bbox.png" | wc -l)"
echo "Grids comparativos: $(find bounding_boxes_landmarks/comparativos -name "grid_*.png" | wc -l)"
echo "Reportes JSON: $(find bounding_boxes_landmarks -name "*.json" | wc -l)"
```

### Comandos de Análisis Específico
```bash
# Análisis de landmark específico
python scripts/generar_bounding_boxes_landmarks.py --metodos minmax --landmarks 1 --datasets entrenamiento --verbose

# Comparación metodológica
python scripts/generar_bounding_boxes_landmarks.py --metodos minmax percentile --datasets entrenamiento --comparar

# Exportación para paper científico
python scripts/generar_bounding_boxes_landmarks.py --metodos minmax --datasets entrenamiento --high-quality --metadatos-completos
```

---
**Última actualización**: Sistema de análisis de landmarks implementado
**Scripts críticos**: generar_bounding_boxes_landmarks.py (análisis científico)
**Método recomendado**: MinMax (100% cobertura garantizada)
**Calidad**: Estándar científico para publicación
**Estado**: Sistema completo y documentado para análisis médico profesional