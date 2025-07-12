# -*- coding: utf-8 -*-
"""
Script para análisis de landmarks individuales y generación de mapas de calor
Analiza la distribución espacial de cada landmark en cuadrícula 299x299
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

class AnalizadorLandmarksHeatmaps:
    def __init__(self):
        self.base_path = Path(__file__).parent.parent
        self.data_path = self.base_path / "data"
        self.coordenadas_path = self.data_path / "coordenadas_299x299"
        self.output_path = self.base_path / "heatmaps_landmarks"
        
        # Crear estructura de directorios de salida
        self.crear_estructura_directorios()
        
        # Configuración de análisis
        self.tamaño_cuadricula = 299
        self.num_landmarks = 15
        self.sigma_gaussian = 1.0  # Para suavizado de mapas de calor
        
        # Categorías médicas reconocidas
        self.categorias_medicas = {
            'COVID': ['COVID'],
            'Normal': ['Normal'],  
            'Viral Pneumonia': ['Viral Pneumonia', 'Viral-Pneumonia']
        }
        
        # Configuración de mapas de calor
        self.config_heatmap = {
            'colormap': 'viridis',
            'dpi': 150,
            'figsize': (10, 10),
            'interpolation': 'bilinear'
        }
        
        # Estadísticas por landmark
        self.estadisticas_landmarks = {}
        
    def crear_estructura_directorios(self):
        """Crea la estructura de directorios para la salida"""
        subdirs = [
            'individuales',
            'por_categoria', 
            'comparativos',
            'estadisticas',
            'datos',
            'datos/matrices_densidad'
        ]
        
        for subdir in subdirs:
            (self.output_path / subdir).mkdir(parents=True, exist_ok=True)
    
    def determinar_categoria_medica(self, covid_id):
        """Determina la categoría médica basada en el COVID_ID"""
        for categoria, prefijos in self.categorias_medicas.items():
            for prefijo in prefijos:
                if covid_id.startswith(prefijo):
                    return categoria
        return 'Desconocida'
    
    def cargar_datos(self, archivo_coordenadas):
        """Carga y estructura los datos de coordenadas"""
        print(f"Cargando datos desde: {archivo_coordenadas}")
        
        if not archivo_coordenadas.exists():
            raise FileNotFoundError(f"No se encontró el archivo: {archivo_coordenadas}")
        
        datos = leer_archivo_coordenadas_generico(archivo_coordenadas)
        
        if datos is None or datos.empty:
            raise ValueError("El archivo de coordenadas está vacío o no se pudo leer")
        
        print(f"Datos cargados: {len(datos)} registros")
        return datos
    
    def extraer_landmarks_por_posicion(self, datos):
        """Extrae coordenadas organizadas por posición de landmark"""
        landmarks_data = {}
        
        # Inicializar estructura para cada landmark
        for i in range(self.num_landmarks):
            landmarks_data[f'landmark_{i+1:02d}'] = {
                'x_coords': [],
                'y_coords': [],
                'categorias': [],
                'covid_ids': []
            }
        
        # Procesar cada registro
        for _, fila in datos.iterrows():
            categoria = self.determinar_categoria_medica(fila['covid_id'])
            covid_id = fila['covid_id']
            coordenadas = fila['coordenadas']
            
            # Verificar que tenemos exactamente 15 landmarks
            if len(coordenadas) != self.num_landmarks:
                print(f"⚠ Advertencia: {covid_id} tiene {len(coordenadas)} landmarks, esperados {self.num_landmarks}")
                continue
            
            # Asignar coordenadas a cada landmark
            for i, (x, y) in enumerate(coordenadas):
                landmark_key = f'landmark_{i+1:02d}'
                landmarks_data[landmark_key]['x_coords'].append(float(x))
                landmarks_data[landmark_key]['y_coords'].append(float(y))
                landmarks_data[landmark_key]['categorias'].append(categoria)
                landmarks_data[landmark_key]['covid_ids'].append(covid_id)
        
        # Convertir listas a numpy arrays para eficiencia
        for landmark_key in landmarks_data:
            landmarks_data[landmark_key]['x_coords'] = np.array(landmarks_data[landmark_key]['x_coords'])
            landmarks_data[landmark_key]['y_coords'] = np.array(landmarks_data[landmark_key]['y_coords'])
        
        print(f"Landmarks extraídos: {len(landmarks_data)} posiciones")
        return landmarks_data
    
    def crear_matriz_densidad(self, x_coords, y_coords, aplicar_suavizado=True):
        """Crea una matriz de densidad 299x299 para un conjunto de coordenadas"""
        # Crear matriz de densidad vacía
        matriz_densidad = np.zeros((self.tamaño_cuadricula, self.tamaño_cuadricula))
        
        # Validar coordenadas
        x_coords = np.clip(x_coords, 0, self.tamaño_cuadricula - 1)
        y_coords = np.clip(y_coords, 0, self.tamaño_cuadricula - 1)
        
        # Incrementar densidad en posiciones específicas
        for x, y in zip(x_coords, y_coords):
            # Convertir a índices enteros
            x_idx = int(round(x))
            y_idx = int(round(y))
            
            # Verificar límites
            if 0 <= x_idx < self.tamaño_cuadricula and 0 <= y_idx < self.tamaño_cuadricula:
                matriz_densidad[y_idx, x_idx] += 1
        
        # Aplicar suavizado simple si se solicita
        if aplicar_suavizado and self.sigma_gaussian > 0:
            matriz_densidad = self.aplicar_suavizado_simple(matriz_densidad)
        
        return matriz_densidad
    
    def aplicar_suavizado_simple(self, matriz):
        """Aplica un suavizado simple sin scipy"""
        # Crear kernel gaussiano simple 3x3
        kernel = np.array([[1, 2, 1],
                          [2, 4, 2], 
                          [1, 2, 1]], dtype=float) / 16.0
        
        # Aplicar convolución manual
        matriz_suavizada = np.zeros_like(matriz)
        h, w = matriz.shape
        
        for i in range(1, h-1):
            for j in range(1, w-1):
                # Aplicar kernel 3x3
                region = matriz[i-1:i+2, j-1:j+2]
                matriz_suavizada[i, j] = np.sum(region * kernel)
        
        # Copiar bordes sin procesar
        matriz_suavizada[0, :] = matriz[0, :]
        matriz_suavizada[-1, :] = matriz[-1, :]
        matriz_suavizada[:, 0] = matriz[:, 0]
        matriz_suavizada[:, -1] = matriz[:, -1]
        
        return matriz_suavizada
    
    def calcular_estadisticas_landmark(self, x_coords, y_coords, categorias):
        """Calcula estadísticas detalladas para un landmark específico"""
        stats = {
            'total_puntos': len(x_coords),
            'estadisticas_globales': {
                'x_media': float(np.mean(x_coords)),
                'y_media': float(np.mean(y_coords)),
                'x_std': float(np.std(x_coords)),
                'y_std': float(np.std(y_coords)),
                'x_min': float(np.min(x_coords)),
                'x_max': float(np.max(x_coords)),
                'y_min': float(np.min(y_coords)),
                'y_max': float(np.max(y_coords)),
                'distancia_centro': float(np.sqrt((np.mean(x_coords) - 149.5)**2 + (np.mean(y_coords) - 149.5)**2)),
                'dispersion_total': float(np.sqrt(np.var(x_coords) + np.var(y_coords)))
            },
            'estadisticas_por_categoria': {}
        }
        
        # Estadísticas por categoría médica
        categorias_unicas = np.unique(categorias)
        for categoria in categorias_unicas:
            mask = np.array(categorias) == categoria
            if np.sum(mask) > 0:
                x_cat = x_coords[mask]
                y_cat = y_coords[mask]
                
                stats['estadisticas_por_categoria'][categoria] = {
                    'count': int(np.sum(mask)),
                    'x_media': float(np.mean(x_cat)),
                    'y_media': float(np.mean(y_cat)),
                    'x_std': float(np.std(x_cat)),
                    'y_std': float(np.std(y_cat)),
                    'distancia_centro': float(np.sqrt((np.mean(x_cat) - 149.5)**2 + (np.mean(y_cat) - 149.5)**2)),
                    'dispersion': float(np.sqrt(np.var(x_cat) + np.var(y_cat)))
                }
        
        return stats
    
    def crear_heatmap_individual(self, matriz_densidad, landmark_id, titulo_extra=""):
        """Crea un mapa de calor individual para un landmark"""
        plt.figure(figsize=self.config_heatmap['figsize'], dpi=self.config_heatmap['dpi'])
        
        # Crear el heatmap
        im = plt.imshow(matriz_densidad, 
                       cmap=self.config_heatmap['colormap'],
                       interpolation=self.config_heatmap['interpolation'],
                       origin='upper',
                       extent=[0, self.tamaño_cuadricula, self.tamaño_cuadricula, 0])
        
        # Configurar título y labels
        titulo = f'Mapa de Calor - {landmark_id.replace("_", " ").title()}'
        if titulo_extra:
            titulo += f' - {titulo_extra}'
        
        plt.title(titulo, fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Coordenada X (píxeles)', fontsize=12)
        plt.ylabel('Coordenada Y (píxeles)', fontsize=12)
        
        # Agregar colorbar
        cbar = plt.colorbar(im, shrink=0.8)
        cbar.set_label('Densidad de Puntos', fontsize=12)
        
        # Configurar grid y aspecto
        plt.grid(True, alpha=0.3)
        plt.gca().set_aspect('equal')
        
        # Ajustar layout
        plt.tight_layout()
        
        return plt.gcf()
    
    def crear_heatmaps_por_categoria(self, landmarks_data, landmark_id):
        """Crea mapas de calor separados por categoría médica para un landmark"""
        x_coords = landmarks_data[landmark_id]['x_coords']
        y_coords = landmarks_data[landmark_id]['y_coords']
        categorias = landmarks_data[landmark_id]['categorias']
        
        mapas_categoria = {}
        
        # Crear mapa para cada categoría
        for categoria in self.categorias_medicas.keys():
            mask = np.array(categorias) == categoria
            if np.sum(mask) > 0:
                x_cat = x_coords[mask]
                y_cat = y_coords[mask]
                
                matriz_densidad = self.crear_matriz_densidad(x_cat, y_cat)
                mapas_categoria[categoria] = matriz_densidad
                
                # Guardar mapa individual por categoría
                fig = self.crear_heatmap_individual(matriz_densidad, landmark_id, categoria)
                archivo_salida = self.output_path / 'por_categoria' / f'{landmark_id}_{categoria.lower().replace(" ", "_")}.png'
                fig.savefig(archivo_salida, bbox_inches='tight', facecolor='white')
                plt.close(fig)
        
        return mapas_categoria
    
    def generar_grid_comparativo(self, landmarks_data, dataset_nombre):
        """Genera un grid comparativo de todos los landmarks"""
        fig, axes = plt.subplots(3, 5, figsize=(20, 12))
        fig.suptitle(f'Grid Comparativo - Todos los Landmarks ({dataset_nombre.title()})', 
                     fontsize=16, fontweight='bold')
        
        landmarks_ordenados = sorted(landmarks_data.keys())
        
        for i, landmark_id in enumerate(landmarks_ordenados):
            row = i // 5
            col = i % 5
            ax = axes[row, col]
            
            # Crear matriz de densidad
            x_coords = landmarks_data[landmark_id]['x_coords']
            y_coords = landmarks_data[landmark_id]['y_coords']
            matriz_densidad = self.crear_matriz_densidad(x_coords, y_coords)
            
            # Mostrar heatmap
            im = ax.imshow(matriz_densidad, 
                          cmap=self.config_heatmap['colormap'],
                          interpolation='bilinear',
                          origin='upper',
                          extent=[0, self.tamaño_cuadricula, self.tamaño_cuadricula, 0])
            
            ax.set_title(f'{landmark_id.replace("_", " ").title()}', fontsize=10)
            ax.set_aspect('equal')
            
            # Reducir etiquetas para ahorrar espacio
            ax.set_xticks([0, 150, 299])
            ax.set_yticks([0, 150, 299])
        
        # Ajustar layout
        plt.tight_layout()
        
        # Guardar grid
        archivo_salida = self.output_path / 'comparativos' / f'grid_todos_landmarks_{dataset_nombre}.png'
        fig.savefig(archivo_salida, bbox_inches='tight', facecolor='white', dpi=200)
        plt.close(fig)
        
        print(f"Grid comparativo guardado: {archivo_salida}")
    
    def exportar_estadisticas_completas(self, estadisticas, dataset_nombre):
        """Exporta todas las estadísticas en formato JSON"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Preparar estructura de exportación
        export_data = {
            'informacion_general': {
                'dataset': dataset_nombre,
                'total_landmarks': self.num_landmarks,
                'tamaño_cuadricula': self.tamaño_cuadricula,
                'fecha_analisis': datetime.now().isoformat(),
                'configuracion': {
                    'sigma_gaussian': self.sigma_gaussian,
                    'colormap': self.config_heatmap['colormap']
                }
            },
            'estadisticas_landmarks': estadisticas,
            'resumen_comparativo': self.generar_resumen_comparativo(estadisticas)
        }
        
        # Exportar JSON
        archivo_json = self.output_path / 'estadisticas' / f'estadisticas_landmarks_{dataset_nombre}_{timestamp}.json'
        with open(archivo_json, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"Estadísticas exportadas: {archivo_json}")
        
        # Exportar CSV con resumen
        self.exportar_resumen_csv(estadisticas, dataset_nombre, timestamp)
        
        return archivo_json
    
    def generar_resumen_comparativo(self, estadisticas):
        """Genera un resumen comparativo entre landmarks"""
        resumen = {
            'landmark_mas_centralizado': '',
            'landmark_mas_disperso': '',
            'landmark_mas_consistente': '',
            'variabilidad_por_categoria': {},
            'landmarks_ordenados_por_dispersion': []
        }
        
        dispersiones = {}
        distancias_centro = {}
        
        # Calcular métricas para comparación
        for landmark_id, stats in estadisticas.items():
            dispersiones[landmark_id] = stats['estadisticas_globales']['dispersion_total']
            distancias_centro[landmark_id] = stats['estadisticas_globales']['distancia_centro']
        
        # Identificar landmarks extremos
        resumen['landmark_mas_centralizado'] = min(distancias_centro, key=distancias_centro.get)
        resumen['landmark_mas_disperso'] = max(dispersiones, key=dispersiones.get)
        resumen['landmark_mas_consistente'] = min(dispersiones, key=dispersiones.get)
        
        # Ordenar por dispersión
        resumen['landmarks_ordenados_por_dispersion'] = sorted(
            dispersiones.items(), key=lambda x: x[1]
        )
        
        return resumen
    
    def exportar_resumen_csv(self, estadisticas, dataset_nombre, timestamp):
        """Exporta un resumen en formato CSV"""
        filas_csv = []
        
        for landmark_id, stats in estadisticas.items():
            fila = {
                'landmark': landmark_id,
                'total_puntos': stats['total_puntos'],
                'x_media': stats['estadisticas_globales']['x_media'],
                'y_media': stats['estadisticas_globales']['y_media'],
                'x_std': stats['estadisticas_globales']['x_std'],
                'y_std': stats['estadisticas_globales']['y_std'],
                'distancia_centro': stats['estadisticas_globales']['distancia_centro'],
                'dispersion_total': stats['estadisticas_globales']['dispersion_total']
            }
            
            # Agregar estadísticas por categoría
            for categoria in ['COVID', 'Normal', 'Viral Pneumonia']:
                if categoria in stats['estadisticas_por_categoria']:
                    cat_stats = stats['estadisticas_por_categoria'][categoria]
                    fila[f'{categoria}_count'] = cat_stats['count']
                    fila[f'{categoria}_x_media'] = cat_stats['x_media']
                    fila[f'{categoria}_y_media'] = cat_stats['y_media']
                    fila[f'{categoria}_dispersion'] = cat_stats['dispersion']
                else:
                    fila[f'{categoria}_count'] = 0
                    fila[f'{categoria}_x_media'] = 0
                    fila[f'{categoria}_y_media'] = 0
                    fila[f'{categoria}_dispersion'] = 0
            
            filas_csv.append(fila)
        
        # Crear DataFrame y exportar
        df_resumen = pd.DataFrame(filas_csv)
        archivo_csv = self.output_path / 'estadisticas' / f'resumen_landmarks_{dataset_nombre}_{timestamp}.csv'
        df_resumen.to_csv(archivo_csv, index=False, encoding='utf-8')
        
        print(f"Resumen CSV exportado: {archivo_csv}")
    
    def exportar_matrices_densidad(self, landmarks_data, dataset_nombre):
        """Exporta las matrices de densidad como archivos numpy"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for landmark_id in landmarks_data:
            x_coords = landmarks_data[landmark_id]['x_coords']
            y_coords = landmarks_data[landmark_id]['y_coords']
            matriz_densidad = self.crear_matriz_densidad(x_coords, y_coords)
            
            # Guardar matriz como archivo numpy
            archivo_matriz = self.output_path / 'datos' / 'matrices_densidad' / f'{landmark_id}_{dataset_nombre}_{timestamp}.npy'
            np.save(archivo_matriz, matriz_densidad)
        
        print(f"Matrices de densidad guardadas en: datos/matrices_densidad/")
    
    def analizar_dataset_completo(self, nombre_dataset):
        """Análisis completo de landmarks para un dataset"""
        print(f"\n=== Analizando landmarks del dataset: {nombre_dataset} ===")
        
        # Cargar datos
        archivo_coordenadas = self.coordenadas_path / f"coordenadas_{nombre_dataset}_299x299.csv"
        datos = self.cargar_datos(archivo_coordenadas)
        
        # Extraer landmarks por posición
        landmarks_data = self.extraer_landmarks_por_posicion(datos)
        
        # Analizar cada landmark individualmente
        estadisticas_completas = {}
        
        print("Generando mapas de calor individuales...")
        for landmark_id in landmarks_data:
            print(f"  Procesando {landmark_id}...")
            
            # Extraer coordenadas
            x_coords = landmarks_data[landmark_id]['x_coords']
            y_coords = landmarks_data[landmark_id]['y_coords']
            categorias = landmarks_data[landmark_id]['categorias']
            
            # Calcular estadísticas
            stats = self.calcular_estadisticas_landmark(x_coords, y_coords, categorias)
            estadisticas_completas[landmark_id] = stats
            
            # Crear matriz de densidad y heatmap individual
            matriz_densidad = self.crear_matriz_densidad(x_coords, y_coords)
            fig = self.crear_heatmap_individual(matriz_densidad, landmark_id)
            
            # Guardar heatmap individual
            archivo_salida = self.output_path / 'individuales' / f'{landmark_id}_{nombre_dataset}.png'
            fig.savefig(archivo_salida, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            
            # Crear mapas por categoría
            self.crear_heatmaps_por_categoria(landmarks_data, landmark_id)
        
        # Generar visualizaciones comparativas
        print("Generando grid comparativo...")
        self.generar_grid_comparativo(landmarks_data, nombre_dataset)
        
        # Exportar todas las estadísticas
        print("Exportando estadísticas...")
        self.exportar_estadisticas_completas(estadisticas_completas, nombre_dataset)
        
        # Exportar matrices de densidad
        print("Exportando matrices de densidad...")
        self.exportar_matrices_densidad(landmarks_data, nombre_dataset)
        
        return estadisticas_completas, landmarks_data

def main():
    parser = argparse.ArgumentParser(description='Análisis de landmarks y generación de mapas de calor')
    parser.add_argument('--dataset', 
                       choices=['maestro', 'entrenamiento', 'prueba'], 
                       default='entrenamiento',
                       help='Dataset a analizar (default: entrenamiento)')
    parser.add_argument('--todos', 
                       action='store_true',
                       help='Analizar todos los datasets disponibles')
    parser.add_argument('--sigma', 
                       type=float, 
                       default=1.0,
                       help='Sigma para suavizado gaussiano (default: 1.0)')
    parser.add_argument('--sin-suavizado', 
                       action='store_true',
                       help='Deshabilitar suavizado gaussiano')
    parser.add_argument('--colormap', 
                       choices=['viridis', 'plasma', 'inferno', 'magma', 'hot'], 
                       default='viridis',
                       help='Mapa de colores para heatmaps (default: viridis)')
    
    args = parser.parse_args()
    
    # Crear analizador
    analizador = AnalizadorLandmarksHeatmaps()
    
    # Configurar parámetros
    if args.sin_suavizado:
        analizador.sigma_gaussian = 0
    else:
        analizador.sigma_gaussian = args.sigma
    
    analizador.config_heatmap['colormap'] = args.colormap
    
    try:
        if args.todos:
            datasets = ['maestro', 'entrenamiento', 'prueba']
            print("Analizando todos los datasets disponibles...")
        else:
            datasets = [args.dataset]
        
        resultados_todos = {}
        
        for dataset in datasets:
            try:
                estadisticas, landmarks_data = analizador.analizar_dataset_completo(dataset)
                resultados_todos[dataset] = {
                    'estadisticas': estadisticas,
                    'landmarks_data': landmarks_data
                }
                
                print(f"\n✅ Análisis completado para dataset '{dataset}'")
                print(f"   - {len(estadisticas)} landmarks analizados")
                print(f"   - Mapas de calor generados: individuales y por categoría")
                
            except Exception as e:
                print(f"❌ Error analizando dataset '{dataset}': {e}")
                continue
        
        print(f"\n🎯 Análisis de landmarks completado para {len(resultados_todos)} dataset(s)")
        print(f"📁 Resultados guardados en: {analizador.output_path}")
        
        # Mostrar estructura de archivos generados
        print(f"\n📋 Archivos generados:")
        print(f"   📁 individuales/     - Mapas de calor por landmark")
        print(f"   📁 por_categoria/    - Mapas por categoría médica")
        print(f"   📁 comparativos/     - Grids comparativos")
        print(f"   📁 estadisticas/     - JSON y CSV con métricas")
        print(f"   📁 datos/           - Matrices de densidad (numpy)")
        
    except Exception as e:
        print(f"❌ Error durante el análisis: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())