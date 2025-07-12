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
│   ├── visualizar_coordenadas_299x299.py   # Script mejorado (NUEVO)
│   ├── organizar_imagenes_por_categoria.py # Organizador automático (NUEVO)
│   ├── procesar_coordenadas.py
│   └── escalar_coordenadas.py
├── visualizations/
│   └── coordenadas_299x299_overlays/       # Imágenes procesadas
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

---
**Última actualización**: Sesión de creación de scripts de visualización 299x299
**Scripts creados**: visualizar_coordenadas_299x299.py, organizar_imagenes_por_categoria.py
**Imágenes procesadas**: 800 (640 entrenamiento + 160 prueba)