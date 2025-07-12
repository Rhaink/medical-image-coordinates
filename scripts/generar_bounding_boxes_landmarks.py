# -*- coding: utf-8 -*-
"""
Script para generar bounding boxes de landmarks basados en mapas de calor
Utiliza matrices de densidad para delimitar áreas de interés de cada landmark
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
import os
import sys
from pathlib import Path
from datetime import datetime
import argparse
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Agregar el directorio scripts al path
sys.path.append(str(Path(__file__).parent))
from procesar_coordenadas import leer_archivo_coordenadas_generico

class GeneradorBoundingBoxesLandmarks:
    def __init__(self):
        self.base_path = Path(__file__).parent.parent
        self.heatmaps_path = self.base_path / "heatmaps_landmarks"
        self.output_path = self.base_path / "bounding_boxes_landmarks"
        
        # Crear estructura de directorios
        self.crear_estructura_directorios()
        
        # Configuración
        self.tamaño_cuadricula = 299
        self.num_landmarks = 15
        
        # Métodos de detección disponibles
        self.metodos_deteccion = {
            'percentile': self.detectar_bbox_percentil,
            'stddev': self.detectar_bbox_estadistico,
            'contours': self.detectar_bbox_contornos,
            'hybrid': self.detectar_bbox_hibrido,
            'minmax': self.detectar_bbox_minmax
        }
        
        # Configuración de detección
        self.config_deteccion = {
            'percentil_umbral': 85,  # Percentil para umbralización
            'factor_expansion': 1.2,  # Factor de expansión del bbox
            'area_minima': 50,  # Área mínima del bbox en píxeles
            'factor_std': 2.0,  # Factor de desviación estándar
            'margen_minmax': 5,  # Margen adicional para método minmax
            'validar_cobertura': True,  # Validar cobertura 100% en minmax
            'metodo_default': 'percentile'
        }
        
        # Colores para categorías
        self.colores_categoria = {
            'COVID': '#FF6B6B',       # Rojo
            'Normal': '#4ECDC4',      # Verde azulado  
            'Viral Pneumonia': '#45B7D1',  # Azul
            'Global': '#FFA726',      # Naranja
            'MinMax': '#2ECC71'       # Verde para método minmax
        }
        
        # Configuración de visualización
        self.config_viz = {
            'bbox_linewidth': 2,
            'bbox_alpha': 0.8,
            'figsize': (12, 10),
            'dpi': 150
        }
        
    def crear_estructura_directorios(self):
        """Crea la estructura de directorios para la salida"""
        subdirs = [
            'individuales',
            'por_categoria',
            'comparativos', 
            'estadisticas',
            'datos',
            'datos/roi_masks'
        ]
        
        for subdir in subdirs:
            (self.output_path / subdir).mkdir(parents=True, exist_ok=True)
    
    def cargar_matriz_densidad(self, landmark_id, dataset, timestamp=None):
        """Carga una matriz de densidad específica"""
        # Buscar el archivo más reciente si no se especifica timestamp
        pattern = f"{landmark_id}_{dataset}_*.npy"
        archivos_matriz = list(self.heatmaps_path.glob(f"datos/matrices_densidad/{pattern}"))
        
        if not archivos_matriz:
            raise FileNotFoundError(f"No se encontró matriz de densidad para {landmark_id} del dataset {dataset}")
        
        # Usar el más reciente si hay múltiples
        archivo_matriz = sorted(archivos_matriz)[-1]
        matriz = np.load(archivo_matriz)
        
        print(f"Matriz cargada: {archivo_matriz.name} - Forma: {matriz.shape}")
        return matriz, archivo_matriz.name
    
    def cargar_estadisticas_landmark(self, dataset):
        """Carga las estadísticas de landmarks para el dataset"""
        # Buscar archivo de estadísticas más reciente
        pattern = f"estadisticas_landmarks_{dataset}_*.json"
        archivos_stats = list(self.heatmaps_path.glob(f"estadisticas/{pattern}"))
        
        if not archivos_stats:
            raise FileNotFoundError(f"No se encontraron estadísticas para dataset {dataset}")
        
        archivo_stats = sorted(archivos_stats)[-1]
        
        with open(archivo_stats, 'r', encoding='utf-8') as f:
            stats = json.load(f)
        
        print(f"Estadísticas cargadas: {archivo_stats.name}")
        return stats['estadisticas_landmarks']
    
    def cargar_coordenadas_originales(self, dataset):
        """Carga las coordenadas originales del CSV del dataset"""
        # Ruta al archivo de coordenadas
        coordenadas_path = self.base_path / "data" / "coordenadas_299x299"
        archivo_coordenadas = coordenadas_path / f"coordenadas_{dataset}_299x299.csv"
        
        if not archivo_coordenadas.exists():
            raise FileNotFoundError(f"No se encontró archivo de coordenadas: {archivo_coordenadas}")
        
        # Leer datos usando la función existente
        datos = leer_archivo_coordenadas_generico(archivo_coordenadas)
        
        if datos is None or datos.empty:
            raise ValueError(f"No se pudieron cargar coordenadas del dataset {dataset}")
        
        print(f"Coordenadas originales cargadas: {len(datos)} registros del dataset {dataset}")
        return datos
    
    def extraer_coordenadas_landmark(self, datos, landmark_index):
        """Extrae las coordenadas de un landmark específico de todos los registros"""
        coordenadas_landmark = []
        
        for _, fila in datos.iterrows():
            # Las coordenadas están almacenadas como lista de tuplas (x, y)
            coordenadas_fila = fila['coordenadas']
            
            # Verificar que el índice del landmark esté disponible
            if landmark_index < len(coordenadas_fila):
                x, y = coordenadas_fila[landmark_index]
                coordenadas_landmark.append((float(x), float(y)))
        
        return coordenadas_landmark
    
    def detectar_bbox_percentil(self, matriz_densidad, landmark_stats=None):
        """Detecta bounding box basado en percentil de densidad"""
        # Calcular umbral basado en percentil
        umbral = np.percentile(matriz_densidad, self.config_deteccion['percentil_umbral'])
        
        # Crear máscara binaria
        mask = matriz_densidad >= umbral
        
        # Encontrar límites del área de interés
        y_indices, x_indices = np.where(mask)
        
        if len(y_indices) == 0:  # No hay píxeles por encima del umbral
            # Usar punto de máxima densidad como centro
            max_pos = np.unravel_index(np.argmax(matriz_densidad), matriz_densidad.shape)
            y_center, x_center = max_pos
            
            # Crear bbox mínimo alrededor del punto
            size = 20  # Tamaño mínimo
            x_min = max(0, x_center - size//2)
            y_min = max(0, y_center - size//2)
            x_max = min(self.tamaño_cuadricula, x_center + size//2)
            y_max = min(self.tamaño_cuadricula, y_center + size//2)
        else:
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)
            
            # Aplicar factor de expansión
            factor = self.config_deteccion['factor_expansion']
            centro_x = (x_min + x_max) / 2
            centro_y = (y_min + y_max) / 2
            ancho = (x_max - x_min) * factor
            alto = (y_max - y_min) * factor
            
            x_min = max(0, int(centro_x - ancho/2))
            x_max = min(self.tamaño_cuadricula, int(centro_x + ancho/2))
            y_min = max(0, int(centro_y - alto/2))
            y_max = min(self.tamaño_cuadricula, int(centro_y + alto/2))
        
        bbox = {
            'x_min': x_min, 'y_min': y_min,
            'x_max': x_max, 'y_max': y_max,
            'ancho': x_max - x_min,
            'alto': y_max - y_min,
            'area': (x_max - x_min) * (y_max - y_min),
            'metodo': 'percentile',
            'parametros': {
                'percentil': self.config_deteccion['percentil_umbral'],
                'factor_expansion': self.config_deteccion['factor_expansion']
            }
        }
        
        return bbox, mask
    
    def detectar_bbox_estadistico(self, matriz_densidad, landmark_stats):
        """Detecta bounding box basado en estadísticas del landmark"""
        if landmark_stats is None:
            # Fallback al método de percentil
            return self.detectar_bbox_percentil(matriz_densidad)
        
        # Usar estadísticas globales del landmark
        stats_globales = landmark_stats['estadisticas_globales']
        x_media = stats_globales['x_media']
        y_media = stats_globales['y_media']
        x_std = stats_globales['x_std']
        y_std = stats_globales['y_std']
        
        # Calcular bbox basado en desviaciones estándar
        factor = self.config_deteccion['factor_std']
        ancho = x_std * factor * 2
        alto = y_std * factor * 2
        
        x_min = max(0, int(x_media - ancho/2))
        x_max = min(self.tamaño_cuadricula, int(x_media + ancho/2))
        y_min = max(0, int(y_media - alto/2))
        y_max = min(self.tamaño_cuadricula, int(y_media + alto/2))
        
        # Crear máscara circular para visualización
        y_grid, x_grid = np.mgrid[0:self.tamaño_cuadricula, 0:self.tamaño_cuadricula]
        mask = ((x_grid - x_media)**2 / (x_std * factor)**2 + 
                (y_grid - y_media)**2 / (y_std * factor)**2) <= 1
        
        bbox = {
            'x_min': x_min, 'y_min': y_min,
            'x_max': x_max, 'y_max': y_max,
            'ancho': x_max - x_min,
            'alto': y_max - y_min,
            'area': (x_max - x_min) * (y_max - y_min),
            'metodo': 'stddev',
            'parametros': {
                'factor_std': factor,
                'x_media': x_media, 'y_media': y_media,
                'x_std': x_std, 'y_std': y_std
            }
        }
        
        return bbox, mask
    
    def detectar_bbox_contornos(self, matriz_densidad, landmark_stats=None):
        """Detecta bounding box usando detección de contornos"""
        # Normalizar matriz para OpenCV (simular sin opencv)
        normalized = (matriz_densidad / np.max(matriz_densidad) * 255).astype(np.uint8)
        
        # Crear máscara binaria con umbral automático (método Otsu simplificado)
        umbral = self.calcular_umbral_otsu_simple(normalized)
        mask = normalized >= umbral
        
        # Encontrar el contorno más grande (región conectada)
        labeled_mask = self.etiquetar_regiones_conectadas(mask)
        
        if labeled_mask.max() == 0:  # No hay regiones
            return self.detectar_bbox_percentil(matriz_densidad, landmark_stats)
        
        # Encontrar la región más grande
        region_sizes = [np.sum(labeled_mask == i) for i in range(1, labeled_mask.max() + 1)]
        largest_region = np.argmax(region_sizes) + 1
        
        # Extraer bounding box de la región más grande
        region_mask = labeled_mask == largest_region
        y_indices, x_indices = np.where(region_mask)
        
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        
        bbox = {
            'x_min': x_min, 'y_min': y_min,
            'x_max': x_max, 'y_max': y_max,
            'ancho': x_max - x_min,
            'alto': y_max - y_min,
            'area': (x_max - x_min) * (y_max - y_min),
            'metodo': 'contours',
            'parametros': {
                'umbral_otsu': int(umbral),
                'region_size': region_sizes[largest_region - 1]
            }
        }
        
        return bbox, region_mask
    
    def detectar_bbox_hibrido(self, matriz_densidad, landmark_stats):
        """Combina múltiples métodos para obtener bbox más robusto"""
        # Ejecutar todos los métodos
        bbox_percentil, _ = self.detectar_bbox_percentil(matriz_densidad, landmark_stats)
        bbox_stats, _ = self.detectar_bbox_estadistico(matriz_densidad, landmark_stats)
        bbox_contornos, mask = self.detectar_bbox_contornos(matriz_densidad, landmark_stats)
        
        # Combinar resultados (usar intersección o promedio)
        bboxes = [bbox_percentil, bbox_stats, bbox_contornos]
        
        # Calcular bbox promedio
        x_min = int(np.mean([b['x_min'] for b in bboxes]))
        y_min = int(np.mean([b['y_min'] for b in bboxes]))
        x_max = int(np.mean([b['x_max'] for b in bboxes]))
        y_max = int(np.mean([b['y_max'] for b in bboxes]))
        
        bbox_hibrido = {
            'x_min': x_min, 'y_min': y_min,
            'x_max': x_max, 'y_max': y_max,
            'ancho': x_max - x_min,
            'alto': y_max - y_min,
            'area': (x_max - x_min) * (y_max - y_min),
            'metodo': 'hybrid',
            'parametros': {
                'metodos_combinados': ['percentile', 'stddev', 'contours'],
                'bbox_individuales': bboxes
            }
        }
        
        return bbox_hibrido, mask
    
    def detectar_bbox_minmax(self, landmark_id, dataset, matriz_densidad=None, landmark_stats=None):
        """Detecta bounding box usando min/max absolutos de coordenadas reales"""
        # Extraer índice del landmark (landmark_01 -> índice 0)
        landmark_index = int(landmark_id.split('_')[1]) - 1
        
        # Cargar coordenadas originales del dataset
        datos_originales = self.cargar_coordenadas_originales(dataset)
        
        # Extraer coordenadas específicas de este landmark
        coordenadas_landmark = self.extraer_coordenadas_landmark(datos_originales, landmark_index)
        
        if not coordenadas_landmark:
            raise ValueError(f"No se encontraron coordenadas para {landmark_id}")
        
        # Separar coordenadas X e Y
        x_coords = [coord[0] for coord in coordenadas_landmark]
        y_coords = [coord[1] for coord in coordenadas_landmark]
        
        # Calcular límites absolutos
        x_min_real = min(x_coords)
        x_max_real = max(x_coords)
        y_min_real = min(y_coords)
        y_max_real = max(y_coords)
        
        # Aplicar margen de seguridad
        margen = self.config_deteccion['margen_minmax']
        
        x_min = max(0, int(x_min_real - margen))
        x_max = min(self.tamaño_cuadricula - 1, int(x_max_real + margen))
        y_min = max(0, int(y_min_real - margen))
        y_max = min(self.tamaño_cuadricula - 1, int(y_max_real + margen))
        
        # Crear máscara con todas las coordenadas reales
        mask = np.zeros((self.tamaño_cuadricula, self.tamaño_cuadricula), dtype=bool)
        for x, y in coordenadas_landmark:
            x_idx = int(round(x))
            y_idx = int(round(y))
            if 0 <= x_idx < self.tamaño_cuadricula and 0 <= y_idx < self.tamaño_cuadricula:
                mask[y_idx, x_idx] = True
        
        # Validar cobertura si está habilitada
        if self.config_deteccion['validar_cobertura']:
            puntos_fuera = self.validar_cobertura_completa(x_min, y_min, x_max, y_max, coordenadas_landmark)
            if puntos_fuera > 0:
                print(f"⚠ Advertencia: {puntos_fuera} puntos fuera del bbox para {landmark_id}")
        
        bbox = {
            'x_min': x_min, 'y_min': y_min,
            'x_max': x_max, 'y_max': y_max,
            'ancho': x_max - x_min + 1,
            'alto': y_max - y_min + 1,
            'area': (x_max - x_min + 1) * (y_max - y_min + 1),
            'metodo': 'minmax',
            'parametros': {
                'margen_aplicado': margen,
                'x_min_real': x_min_real,
                'x_max_real': x_max_real,
                'y_min_real': y_min_real,
                'y_max_real': y_max_real,
                'total_puntos_reales': len(coordenadas_landmark),
                'coordenadas_reales': coordenadas_landmark  # Para visualización
            }
        }
        
        return bbox, mask
    
    def validar_cobertura_completa(self, x_min, y_min, x_max, y_max, coordenadas_reales):
        """Valida que todas las coordenadas reales estén dentro del bounding box"""
        puntos_fuera = 0
        
        for x, y in coordenadas_reales:
            if not (x_min <= x <= x_max and y_min <= y <= y_max):
                puntos_fuera += 1
        
        return puntos_fuera
    
    def calcular_umbral_otsu_simple(self, imagen):
        """Implementación simplificada del método de Otsu"""
        # Calcular histograma
        hist, bins = np.histogram(imagen, bins=256, range=(0, 256))
        
        # Normalizar histograma
        hist = hist.astype(float) / np.sum(hist)
        
        # Calcular umbral óptimo
        max_variance = 0
        umbral_optimo = 0
        
        for t in range(1, 255):
            # Probabilidades de las clases
            w0 = np.sum(hist[:t])
            w1 = np.sum(hist[t:])
            
            if w0 == 0 or w1 == 0:
                continue
            
            # Medias de las clases
            mu0 = np.sum(np.arange(t) * hist[:t]) / w0
            mu1 = np.sum(np.arange(t, 256) * hist[t:]) / w1
            
            # Varianza entre clases
            variance = w0 * w1 * (mu0 - mu1) ** 2
            
            if variance > max_variance:
                max_variance = variance
                umbral_optimo = t
        
        return umbral_optimo
    
    def etiquetar_regiones_conectadas(self, mask_binaria):
        """Etiqueta regiones conectadas en una máscara binaria"""
        labeled = np.zeros_like(mask_binaria, dtype=int)
        label = 0
        
        for i in range(mask_binaria.shape[0]):
            for j in range(mask_binaria.shape[1]):
                if mask_binaria[i, j] and labeled[i, j] == 0:
                    label += 1
                    self.flood_fill(mask_binaria, labeled, i, j, label)
        
        return labeled
    
    def flood_fill(self, mask, labeled, start_i, start_j, label):
        """Implementación simple de flood fill"""
        stack = [(start_i, start_j)]
        
        while stack:
            i, j = stack.pop()
            
            if (i < 0 or i >= mask.shape[0] or j < 0 or j >= mask.shape[1] or
                not mask[i, j] or labeled[i, j] != 0):
                continue
            
            labeled[i, j] = label
            
            # Agregar vecinos 4-conectados
            stack.extend([(i+1, j), (i-1, j), (i, j+1), (i, j-1)])
    
    def calcular_metricas_bbox(self, bbox, matriz_densidad, landmark_stats=None):
        """Calcula métricas de calidad del bounding box"""
        x_min, y_min = bbox['x_min'], bbox['y_min']
        x_max, y_max = bbox['x_max'], bbox['y_max']
        
        # Extraer región del bbox
        region = matriz_densidad[y_min:y_max, x_min:x_max]
        
        # Métricas básicas
        metricas = {
            'area_bbox': bbox['area'],
            'area_porcentaje': (bbox['area'] / (self.tamaño_cuadricula ** 2)) * 100,
            'ratio_aspecto': bbox['ancho'] / bbox['alto'] if bbox['alto'] > 0 else 0,
            'densidad_promedio_region': np.mean(region),
            'densidad_maxima_region': np.max(region),
            'densidad_total_capturada': np.sum(region),
            'centro_bbox': {
                'x': (x_min + x_max) / 2,
                'y': (y_min + y_max) / 2
            }
        }
        
        # Métricas adicionales si tenemos estadísticas
        if landmark_stats:
            stats_globales = landmark_stats['estadisticas_globales']
            centro_real = (stats_globales['x_media'], stats_globales['y_media'])
            centro_bbox = (metricas['centro_bbox']['x'], metricas['centro_bbox']['y'])
            
            metricas['distancia_centro_real'] = np.sqrt(
                (centro_real[0] - centro_bbox[0])**2 + 
                (centro_real[1] - centro_bbox[1])**2
            )
            
            # Cobertura: porcentaje de densidad total capturada
            densidad_total = np.sum(matriz_densidad)
            if densidad_total > 0:
                metricas['cobertura_densidad'] = (metricas['densidad_total_capturada'] / densidad_total) * 100
            else:
                metricas['cobertura_densidad'] = 0.0
        
        # Métricas específicas para método minmax
        if bbox['metodo'] == 'minmax' and 'parametros' in bbox:
            params = bbox['parametros']
            metricas['cobertura_real'] = 100.0  # Siempre 100% por diseño
            metricas['puntos_reales_incluidos'] = params['total_puntos_reales']
            metricas['margen_aplicado'] = params['margen_aplicado']
            metricas['eficiencia_espacial'] = (params['total_puntos_reales'] / bbox['area']) * 100
        
        return metricas
    
    def crear_visualizacion_bbox(self, landmark_id, matriz_densidad, bbox, metricas, 
                                titulo_extra="", mostrar_metricas=True):
        """Crea visualización con bounding box superpuesto"""
        # Layout científico con espacio para títulos y métricas
        fig, ax = plt.subplots(figsize=(14, 10), dpi=300, facecolor='black')
        plt.subplots_adjust(top=0.85, bottom=0.15, left=0.1, right=0.85)
        
        # Mostrar mapa de calor con fondo negro real
        # Configurar fondo negro en todas las areas
        ax.set_facecolor('black')
        
        # Enmascarar valores cero para que sean completamente negros
        matriz_masked = np.ma.masked_where(matriz_densidad == 0, matriz_densidad)
        im = ax.imshow(matriz_masked, 
                      cmap='plasma',  # Colormap científico perceptualmente uniforme
                      interpolation='bilinear',
                      origin='upper',
                      extent=[0, self.tamaño_cuadricula, self.tamaño_cuadricula, 0],
                      vmin=0.1,
                      alpha=0.85)
        
        # Elegir color según el método
        color_bbox = self.colores_categoria['MinMax'] if bbox['metodo'] == 'minmax' else self.colores_categoria['Global']
        
        # Superponer bounding box
        rect = patches.Rectangle(
            (bbox['x_min'], bbox['y_min']), 
            bbox['ancho'], bbox['alto'],
            linewidth=self.config_viz['bbox_linewidth'],
            edgecolor=color_bbox,
            facecolor='none',
            alpha=self.config_viz['bbox_alpha']
        )
        ax.add_patch(rect)
        
        # Agregar cruz en el centro del bbox
        centro_x = (bbox['x_min'] + bbox['x_max']) / 2
        centro_y = (bbox['y_min'] + bbox['y_max']) / 2
        ax.plot(centro_x, centro_y, '+', color=color_bbox, 
               markersize=12, markeredgewidth=3)
        
        # Para método minmax, mostrar puntos reales con colores rojo/naranja
        if bbox['metodo'] == 'minmax' and 'coordenadas_reales' in bbox['parametros']:
            coordenadas_reales = bbox['parametros']['coordenadas_reales']
            if coordenadas_reales:
                x_reales = [coord[0] for coord in coordenadas_reales]
                y_reales = [coord[1] for coord in coordenadas_reales]
                
                # Crear colores alternando entre rojo y naranja
                colores = ['red' if i % 2 == 0 else 'orange' for i in range(len(coordenadas_reales))]
                
                ax.scatter(x_reales, y_reales, c=colores, s=12, alpha=0.9, marker='o', 
                          edgecolors='white', linewidths=0.5,
                          label=f'Anatomical landmarks (n={len(coordenadas_reales)})')
                ax.legend(loc='upper right', facecolor='white', edgecolor='black', 
                         fontsize=10, framealpha=0.9)
        
        # Configurar título científico
        landmark_num = landmark_id.split('_')[1]
        titulo_principal = f'Landmark L{landmark_num}: Spatial Distribution and ROI Detection'
        if titulo_extra:
            subtitulo = f'{titulo_extra} Dataset (n=640) - MinMax Method'
        else:
            subtitulo = 'Training Dataset (n=640) - MinMax Method'
        
        # Título principal más grande
        ax.text(0.5, 1.08, titulo_principal, transform=ax.transAxes, 
               fontsize=16, fontweight='bold', ha='center', color='white',
               fontfamily='serif')
        # Subtítulo más pequeño
        ax.text(0.5, 1.04, subtitulo, transform=ax.transAxes,
               fontsize=12, ha='center', color='lightgray',
               fontfamily='serif')
        
        # Labels científicos y grid
        ax.set_xlabel('X-coordinate (pixels)', fontsize=12, color='white', fontfamily='serif')
        ax.set_ylabel('Y-coordinate (pixels)', fontsize=12, color='white', fontfamily='serif')
        ax.tick_params(colors='white', labelsize=10)
        ax.grid(True, alpha=0.2, color='gray', linestyle='--', linewidth=0.5)
        
        # Agregar colorbar científico
        cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label('Point Density Distribution', fontsize=11, color='white', fontfamily='serif')
        cbar.ax.tick_params(colors='white', labelsize=9)
        # Añadir unidades científicas
        cbar.ax.text(1.15, 0.5, '[counts/px²]', transform=cbar.ax.transAxes, 
                    rotation=90, va='center', fontsize=9, color='lightgray')
        
        # Mostrar tabla de métricas científicas
        if mostrar_metricas:
            tabla_metricas = self.formatear_metricas_para_display(metricas, bbox)
            
            # Crear tabla científica en la parte inferior
            y_start = -0.12
            for i, fila in enumerate(tabla_metricas):
                if i == 0:  # Header
                    for j, celda in enumerate(fila):
                        ax.text(0.05 + j*0.22, y_start, celda, transform=ax.transAxes,
                               fontsize=10, fontweight='bold', color='white',
                               fontfamily='serif')
                elif i == 1:  # Separador
                    continue
                else:  # Datos
                    for j, celda in enumerate(fila):
                        color = 'lightgray' if j % 2 == 0 else 'white'
                        ax.text(0.05 + j*0.22, y_start - (i-1)*0.02, celda, transform=ax.transAxes,
                               fontsize=9, color=color, fontfamily='monospace')
        
        plt.tight_layout()
        return fig
    
    def formatear_metricas_para_display(self, metricas, bbox):
        """Formatea métricas científicas para visualización profesional"""
        # Crear tabla científica organizada
        tabla_metricas = [
            ["ROI Metrics", "Value", "Spatial Stats", "Value"],
            ["───────────", "─────", "────────────", "─────"],
            [f"Area (px²)", f"{bbox['area']:,}", f"Centroid X", f"{metricas['centro_bbox']['x']:.1f}"],
            [f"Coverage (%)", f"{metricas.get('cobertura_densidad', 100):.1f}", f"Centroid Y", f"{metricas['centro_bbox']['y']:.1f}"],
            [f"Aspect Ratio", f"{metricas['ratio_aspecto']:.3f}", f"Dimensions", f"{bbox['ancho']}×{bbox['alto']}"],
            [f"Efficiency", f"{metricas.get('eficiencia_espacial', 0):.2f}", f"Method", f"{bbox['metodo'].upper()}"],
            [f"Density (μ)", f"{metricas['densidad_promedio_region']:.3f}", f"Points (n)", f"{metricas.get('puntos_reales_incluidos', 640)}"]
        ]
        
        return tabla_metricas
    
    def procesar_landmark_individual(self, landmark_id, dataset, metodo='percentile'):
        """Procesa un landmark individual y genera su bounding box"""
        print(f"Procesando {landmark_id} del dataset {dataset}...")
        
        # Cargar matriz de densidad
        matriz, archivo_matriz = self.cargar_matriz_densidad(landmark_id, dataset)
        
        # Cargar estadísticas si están disponibles
        try:
            stats_landmarks = self.cargar_estadisticas_landmark(dataset)
            landmark_stats = stats_landmarks.get(landmark_id)
        except:
            landmark_stats = None
            print(f"⚠ No se pudieron cargar estadísticas para {landmark_id}")
        
        # Detectar bounding box usando el método especificado
        detector = self.metodos_deteccion.get(metodo, self.metodos_deteccion['percentile'])
        
        # El método minmax necesita parámetros especiales
        if metodo == 'minmax':
            bbox, mask = detector(landmark_id, dataset, matriz, landmark_stats)
        else:
            bbox, mask = detector(matriz, landmark_stats)
        
        # Validar área mínima
        if bbox['area'] < self.config_deteccion['area_minima']:
            print(f"⚠ Área del bbox muy pequeña ({bbox['area']} px²), expandiendo...")
            bbox = self.expandir_bbox_minimo(bbox)
        
        # Calcular métricas
        metricas = self.calcular_metricas_bbox(bbox, matriz, landmark_stats)
        
        # Crear visualización
        fig = self.crear_visualizacion_bbox(landmark_id, matriz, bbox, metricas, 
                                          f"{dataset.title()} - {metodo.title()}")
        
        # Guardar visualización científica de alta resolución
        archivo_salida = self.output_path / 'individuales' / f'{landmark_id}_{dataset}_{metodo}_bbox.png'
        fig.patch.set_facecolor('black')
        fig.savefig(archivo_salida, bbox_inches='tight', facecolor='black', 
                   dpi=300, format='png',
                   metadata={'Title': f'Landmark {landmark_id} Analysis',
                            'Author': 'Medical Imaging Analysis System',
                            'Subject': 'Spatial Distribution and ROI Detection'})
        plt.close(fig)
        
        # Guardar estadísticas del bbox
        bbox_completo = {
            'landmark_id': landmark_id,
            'dataset': dataset,
            'archivo_matriz_fuente': archivo_matriz,
            'metodo_deteccion': metodo,
            'bbox': bbox,
            'metricas': metricas,
            'timestamp': datetime.now().isoformat()
        }
        
        archivo_stats = self.output_path / 'individuales' / f'{landmark_id}_{dataset}_{metodo}_bbox_stats.json'
        with open(archivo_stats, 'w', encoding='utf-8') as f:
            json.dump(bbox_completo, f, indent=2, ensure_ascii=False)
        
        print(f"✓ {landmark_id} procesado - Área: {bbox['area']} px², Método: {metodo}")
        
        return bbox_completo
    
    def expandir_bbox_minimo(self, bbox):
        """Expande un bbox que es demasiado pequeño"""
        centro_x = (bbox['x_min'] + bbox['x_max']) / 2
        centro_y = (bbox['y_min'] + bbox['y_max']) / 2
        
        # Asegurar tamaño mínimo
        tamaño_min = int(np.sqrt(self.config_deteccion['area_minima']))
        
        nuevo_ancho = max(bbox['ancho'], tamaño_min)
        nuevo_alto = max(bbox['alto'], tamaño_min)
        
        bbox['x_min'] = max(0, int(centro_x - nuevo_ancho/2))
        bbox['x_max'] = min(self.tamaño_cuadricula, int(centro_x + nuevo_ancho/2))
        bbox['y_min'] = max(0, int(centro_y - nuevo_alto/2))
        bbox['y_max'] = min(self.tamaño_cuadricula, int(centro_y + nuevo_alto/2))
        
        bbox['ancho'] = bbox['x_max'] - bbox['x_min']
        bbox['alto'] = bbox['y_max'] - bbox['y_min']
        bbox['area'] = bbox['ancho'] * bbox['alto']
        
        return bbox
    
    def procesar_dataset_completo(self, dataset, metodos=['percentile']):
        """Procesa todos los landmarks de un dataset"""
        print(f"\n=== Procesando dataset completo: {dataset} ===")
        
        resultados = {}
        
        for metodo in metodos:
            print(f"\nUsando método: {metodo}")
            resultados[metodo] = {}
            
            for i in range(1, self.num_landmarks + 1):
                landmark_id = f'landmark_{i:02d}'
                
                try:
                    bbox_resultado = self.procesar_landmark_individual(landmark_id, dataset, metodo)
                    resultados[metodo][landmark_id] = bbox_resultado
                except Exception as e:
                    print(f"❌ Error procesando {landmark_id}: {e}")
                    continue
        
        # Generar resumen y comparativas
        self.generar_resumen_dataset(dataset, resultados)
        self.generar_grid_comparativo_bboxes(dataset, resultados)
        
        return resultados
    
    def generar_resumen_dataset(self, dataset, resultados):
        """Genera resumen estadístico del dataset procesado"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Preparar datos para el resumen
        resumen_datos = []
        
        for metodo, landmarks in resultados.items():
            for landmark_id, data in landmarks.items():
                bbox = data['bbox']
                metricas = data['metricas']
                
                fila = {
                    'dataset': dataset,
                    'landmark': landmark_id,
                    'metodo': metodo,
                    'x_min': bbox['x_min'],
                    'y_min': bbox['y_min'],
                    'x_max': bbox['x_max'],
                    'y_max': bbox['y_max'],
                    'ancho': bbox['ancho'],
                    'alto': bbox['alto'],
                    'area': bbox['area'],
                    'area_porcentaje': metricas['area_porcentaje'],
                    'ratio_aspecto': metricas['ratio_aspecto'],
                    'densidad_promedio': metricas['densidad_promedio_region'],
                    'centro_x': metricas['centro_bbox']['x'],
                    'centro_y': metricas['centro_bbox']['y']
                }
                
                if 'cobertura_densidad' in metricas:
                    fila['cobertura_densidad'] = metricas['cobertura_densidad']
                    fila['distancia_centro_real'] = metricas['distancia_centro_real']
                
                resumen_datos.append(fila)
        
        # Crear DataFrame y exportar CSV
        df_resumen = pd.DataFrame(resumen_datos)
        archivo_csv = self.output_path / 'estadisticas' / f'bbox_coords_{dataset}_{timestamp}.csv'
        df_resumen.to_csv(archivo_csv, index=False, encoding='utf-8')
        
        # Generar estadísticas agregadas
        stats_agregadas = {
            'dataset': dataset,
            'timestamp': timestamp,
            'total_landmarks': len(set(df_resumen['landmark'])),
            'metodos_aplicados': list(df_resumen['metodo'].unique()),
            'estadisticas_por_metodo': {}
        }
        
        for metodo in df_resumen['metodo'].unique():
            df_metodo = df_resumen[df_resumen['metodo'] == metodo]
            
            stats_agregadas['estadisticas_por_metodo'][metodo] = {
                'area_promedio': float(df_metodo['area'].mean()),
                'area_std': float(df_metodo['area'].std()),
                'area_min': int(df_metodo['area'].min()),
                'area_max': int(df_metodo['area'].max()),
                'ratio_aspecto_promedio': float(df_metodo['ratio_aspecto'].mean()),
                'cobertura_total': float(df_metodo['area'].sum() / (self.tamaño_cuadricula**2) * 100)
            }
        
        # Exportar estadísticas agregadas
        archivo_json = self.output_path / 'estadisticas' / f'bbox_summary_{dataset}_{timestamp}.json'
        with open(archivo_json, 'w', encoding='utf-8') as f:
            json.dump(stats_agregadas, f, indent=2, ensure_ascii=False)
        
        print(f"📊 Resumen exportado: {archivo_csv.name}")
        print(f"📊 Estadísticas: {archivo_json.name}")
    
    def generar_grid_comparativo_bboxes(self, dataset, resultados):
        """Genera grid comparativo de bounding boxes"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Para cada método, crear un grid
        for metodo, landmarks in resultados.items():
            if not landmarks:
                continue
                
            fig, axes = plt.subplots(3, 5, figsize=(20, 12), facecolor='black', dpi=300)
            fig.suptitle(f'Landmark Analysis Overview: {dataset.title()} Dataset (MinMax Method)', 
                         fontsize=18, fontweight='bold', color='white', fontfamily='serif')
            fig.text(0.5, 0.94, f'Spatial Distribution and ROI Detection Results (n=640 per landmark)', 
                    ha='center', fontsize=12, color='lightgray', fontfamily='serif')
            
            landmarks_ordenados = sorted(landmarks.keys())
            
            for i, landmark_id in enumerate(landmarks_ordenados):
                if i >= 15:  # Solo mostrar 15 landmarks
                    break
                    
                row = i // 5
                col = i % 5
                ax = axes[row, col]
                
                # Cargar matriz de densidad para este landmark
                try:
                    matriz, _ = self.cargar_matriz_densidad(landmark_id, dataset)
                    bbox = landmarks[landmark_id]['bbox']
                    
                    # Mostrar mapa de calor con fondo negro real
                    # Configurar fondo negro
                    ax.set_facecolor('black')
                    
                    # Enmascarar valores cero para que sean completamente negros
                    matriz_masked = np.ma.masked_where(matriz == 0, matriz)
                    im = ax.imshow(matriz_masked, 
                                  cmap='plasma',  # Colormap científico consistente
                                  interpolation='bilinear',
                                  origin='upper', 
                                  extent=[0, self.tamaño_cuadricula, self.tamaño_cuadricula, 0],
                                  vmin=0.1,
                                  alpha=0.85)
                    
                    # Superponer bounding box
                    rect = patches.Rectangle(
                        (bbox['x_min'], bbox['y_min']), 
                        bbox['ancho'], bbox['alto'],
                        linewidth=2,
                        edgecolor=self.colores_categoria['Global'],
                        facecolor='none',
                        alpha=0.8
                    )
                    ax.add_patch(rect)
                    
                    # Título científico del subplot
                    landmark_num = landmark_id.split('_')[1]
                    ax.set_title(f"L{landmark_num}\n{bbox['area']:,}px² ({bbox['ancho']}×{bbox['alto']})", 
                               fontsize=9, color='white', fontfamily='serif')
                    ax.set_aspect('equal')
                    
                    # Reducir etiquetas
                    ax.set_xticks([0, 150, 299])
                    ax.set_yticks([0, 150, 299])
                    ax.tick_params(colors='white')
                    
                except Exception as e:
                    ax.text(0.5, 0.5, f'Error:\n{landmark_id}', 
                           transform=ax.transAxes, ha='center', va='center')
                    ax.set_title(landmark_id, fontsize=10)
            
            # Ocultar ejes no utilizados
            for i in range(15, 15):
                if i < len(axes.flat):
                    axes.flat[i].set_visible(False)
            
            plt.tight_layout()
            
            # Guardar grid
            archivo_grid = self.output_path / 'comparativos' / f'grid_bboxes_{dataset}_{metodo}_{timestamp}.png'
            fig.patch.set_facecolor('black')
            fig.savefig(archivo_grid, bbox_inches='tight', facecolor='black', dpi=200)
            plt.close(fig)
            
            print(f"🖼️ Grid generado: {archivo_grid.name}")
    
    def crear_comparativa_categorias_bbox(self, landmark_id, dataset):
        """Crea comparativa de bboxes por categoría médica para un landmark"""
        # Esta función se implementaría para comparar bboxes entre COVID/Normal/Viral Pneumonia
        # Requeriría cargar matrices por categoría separadas, que no están implementadas en el sistema actual
        pass

def main():
    parser = argparse.ArgumentParser(description='Generación de bounding boxes para landmarks')
    parser.add_argument('--dataset', 
                       choices=['maestro', 'entrenamiento', 'prueba', 'todos'], 
                       default='entrenamiento',
                       help='Dataset a procesar (default: entrenamiento)')
    parser.add_argument('--methods', 
                       default='percentile',
                       help='Métodos de detección separados por coma (percentile,stddev,contours,hybrid)')
    parser.add_argument('--threshold', 
                       type=int, 
                       default=85,
                       help='Percentil para umbralización (default: 85)')
    parser.add_argument('--expansion-factor', 
                       type=float, 
                       default=1.2,
                       help='Factor de expansión del bbox (default: 1.2)')
    parser.add_argument('--min-area', 
                       type=int, 
                       default=50,
                       help='Área mínima del bbox en píxeles (default: 50)')
    
    args = parser.parse_args()
    
    # Crear generador
    generador = GeneradorBoundingBoxesLandmarks()
    
    # Configurar parámetros
    generador.config_deteccion['percentil_umbral'] = args.threshold
    generador.config_deteccion['factor_expansion'] = args.expansion_factor
    generador.config_deteccion['area_minima'] = args.min_area
    
    # Parsear métodos
    metodos = [m.strip() for m in args.methods.split(',')]
    metodos_validos = [m for m in metodos if m in generador.metodos_deteccion]
    
    if not metodos_validos:
        print(f"❌ Métodos no válidos: {metodos}")
        print(f"Métodos disponibles: {list(generador.metodos_deteccion.keys())}")
        return 1
    
    try:
        if args.dataset == 'todos':
            datasets = ['maestro', 'entrenamiento', 'prueba']
        else:
            datasets = [args.dataset]
        
        for dataset in datasets:
            try:
                resultados = generador.procesar_dataset_completo(dataset, metodos_validos)
                print(f"\n✅ Dataset '{dataset}' procesado exitosamente")
                print(f"   - {len(metodos_validos)} método(s) aplicado(s)")
                print(f"   - {len(resultados[metodos_validos[0]])} landmarks procesados")
                
            except Exception as e:
                print(f"❌ Error procesando dataset '{dataset}': {e}")
                continue
        
        print(f"\n🎯 Procesamiento completado")
        print(f"📁 Resultados guardados en: {generador.output_path}")
        
    except Exception as e:
        print(f"❌ Error durante el procesamiento: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())