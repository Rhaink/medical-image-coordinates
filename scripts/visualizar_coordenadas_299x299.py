# -*- coding: utf-8 -*-
"""
Script mejorado para visualizar coordenadas 299x299 del dataset
con funcionalidades adicionales y mejor organización
"""

import pandas as pd
import numpy as np
import cv2
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
from datetime import datetime
import argparse

# Agregar el directorio scripts al path
sys.path.append(str(Path(__file__).parent))
from procesar_coordenadas import leer_archivo_coordenadas_generico

class VisualizadorCoordenadas299x299:
    def __init__(self):
        self.base_path = Path(__file__).parent.parent  # Subir un nivel desde scripts/
        self.data_path = self.base_path / "data"
        self.covid_images_path = self.data_path / "COVID-19_Radiography_Dataset" / "COVID" / "images"
        self.viral_images_path = self.data_path / "COVID-19_Radiography_Dataset" / "Viral Pneumonia" / "images"
        self.normal_images_path = self.data_path / "COVID-19_Radiography_Dataset" / "Normal" / "images"
        self.coordenadas_path = self.data_path / "coordenadas_299x299"
        self.output_path = self.base_path / "visualizations" / "coordenadas_299x299_overlays"
        
        # Colores uniformes como el script original
        self.colores = {
            'puntos': (0, 255, 0),    # Verde para todos los puntos
            'numeros': (0, 0, 255),   # Rojo para todos los números
            'borde': (0, 255, 0)      # Verde para todos los bordes
        }
        
        # Estadísticas detalladas
        self.stats = {
            'total_registros': 0,
            'por_categoria': {
                'COVID': {'procesadas': 0, 'faltantes': 0, 'errores': 0},
                'Normal': {'procesadas': 0, 'faltantes': 0, 'errores': 0},
                'Viral Pneumonia': {'procesadas': 0, 'faltantes': 0, 'errores': 0}
            },
            'archivos_faltantes': [],
            'total_coordenadas': 0
        }
    
    def obtener_tipo_imagen(self, covid_id):
        """Determina el tipo de imagen basado en el ID"""
        if covid_id.startswith('COVID-'):
            return 'COVID'
        elif covid_id.startswith('Normal-'):
            return 'Normal'
        elif covid_id.startswith('Viral Pneumonia-'):
            return 'Viral Pneumonia'
        else:
            return 'Desconocido'
    
    def cargar_imagen(self, covid_id):
        """Carga una imagen basada en su COVID ID"""
        try:
            tipo_imagen = self.obtener_tipo_imagen(covid_id)
            
            if tipo_imagen == 'COVID':
                imagen_path = self.covid_images_path / f"{covid_id}.png"
            elif tipo_imagen == 'Normal':
                imagen_path = self.normal_images_path / f"{covid_id}.png"
            elif tipo_imagen == 'Viral Pneumonia':
                imagen_path = self.viral_images_path / f"{covid_id}.png"
            else:
                print(f"⚠ Tipo de imagen no reconocido: {covid_id}")
                return None, tipo_imagen
            
            if imagen_path.exists():
                imagen = cv2.imread(str(imagen_path))
                if imagen is not None:
                    imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
                    return imagen, tipo_imagen
                else:
                    print(f"⚠ Error al cargar la imagen: {imagen_path}")
                    return None, tipo_imagen
            else:
                print(f"⚠ Imagen no encontrada: {imagen_path}")
                self.stats['archivos_faltantes'].append(str(imagen_path))
                return None, tipo_imagen
                
        except Exception as e:
            print(f"✗ Error al cargar {covid_id}: {e}")
            return None, 'Error'
    
    def redimensionar_imagen(self, imagen, target_size=(299, 299)):
        """Redimensiona la imagen al tamaño objetivo"""
        if imagen is None:
            return None
        
        altura_actual, ancho_actual = imagen.shape[:2]
        
        if (ancho_actual, altura_actual) == target_size:
            return imagen
        
        imagen_redimensionada = cv2.resize(imagen, target_size)
        return imagen_redimensionada
    
    def dibujar_coordenadas(self, imagen, coordenadas, tipo_imagen, mostrar_numeros=True):
        """Dibuja las coordenadas sobre la imagen con estilo uniforme como el script original"""
        if imagen is None or not coordenadas:
            return imagen
        
        imagen_anotada = imagen.copy()
        
        # Usar colores uniformes como el script original
        color_puntos = self.colores['puntos']
        color_numeros = self.colores['numeros']
        color_borde = self.colores['borde']
        
        # Dibujar cada coordenada
        for i, (x, y) in enumerate(coordenadas):
            x_int = int(round(x))
            y_int = int(round(y))
            
            altura, ancho = imagen_anotada.shape[:2]
            if 0 <= x_int < ancho and 0 <= y_int < altura:
                # Dibujar círculo con parámetros del script original
                cv2.circle(imagen_anotada, (x_int, y_int), 3, color_puntos, -1)
                cv2.circle(imagen_anotada, (x_int, y_int), 5, color_borde, 1)
                
                # Dibujar número si se solicita
                if mostrar_numeros:
                    cv2.putText(imagen_anotada, str(i+1), (x_int+7, y_int-7), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, color_numeros, 1)
            else:
                print(f"  ⚠ Coordenada fuera de rango: ({x_int}, {y_int}) en imagen {ancho}x{altura}")
        
        return imagen_anotada
    
    def procesar_imagen_individual(self, registro, mostrar_progreso=False):
        """Procesa una imagen individual con sus coordenadas"""
        covid_id = registro['covid_id']
        coordenadas = registro['coordenadas']
        
        if mostrar_progreso:
            print(f"📸 Procesando: {covid_id}")
        
        # Cargar imagen
        imagen, tipo_imagen = self.cargar_imagen(covid_id)
        if imagen is None:
            self.stats['por_categoria'][tipo_imagen]['faltantes'] += 1
            return None, False, tipo_imagen
        
        # Redimensionar a 299x299
        imagen = self.redimensionar_imagen(imagen, (299, 299))
        
        # Dibujar coordenadas
        imagen_anotada = self.dibujar_coordenadas(imagen, coordenadas, tipo_imagen)
        
        # Guardar imagen anotada con nombre simple como el script original
        output_filename = f"{covid_id}_anotada.png"
        output_path = self.output_path / output_filename
        
        # Crear directorio si no existe
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convertir de RGB a BGR para guardar con OpenCV
        imagen_bgr = cv2.cvtColor(imagen_anotada, cv2.COLOR_RGB2BGR)
        
        if cv2.imwrite(str(output_path), imagen_bgr):
            if mostrar_progreso:
                print(f"  ✓ Guardada en: {output_path}")
            self.stats['por_categoria'][tipo_imagen]['procesadas'] += 1
            self.stats['total_coordenadas'] += len(coordenadas)
            return imagen_anotada, True, tipo_imagen
        else:
            print(f"  ✗ Error al guardar: {output_path}")
            self.stats['por_categoria'][tipo_imagen]['errores'] += 1
            return imagen_anotada, False, tipo_imagen
    
    def cargar_coordenadas_dataset(self, dataset='maestro'):
        """Carga las coordenadas del dataset especificado"""
        archivos_disponibles = {
            'maestro': 'coordenadas_maestro_299x299.csv',
            'entrenamiento': 'coordenadas_entrenamiento_299x299.csv',
            'prueba': 'coordenadas_prueba_299x299.csv'
        }
        
        if dataset not in archivos_disponibles:
            print(f"✗ Dataset '{dataset}' no válido. Opciones: {list(archivos_disponibles.keys())}")
            return None
        
        archivo = archivos_disponibles[dataset]
        archivo_path = self.coordenadas_path / archivo
        
        if not archivo_path.exists():
            print(f"✗ Archivo no encontrado: {archivo_path}")
            return None
        
        print(f"📂 Cargando dataset: {dataset} ({archivo})")
        return leer_archivo_coordenadas_generico(archivo_path)
    
    def procesar_dataset(self, dataset='maestro', limite=None, mostrar_muestra=True):
        """Procesa un dataset completo"""
        print(f"=== VISUALIZADOR DE COORDENADAS 299x299 ===")
        print(f"Dataset: {dataset.upper()}\n")
        
        # Cargar datos de coordenadas
        df_coordenadas = self.cargar_coordenadas_dataset(dataset)
        
        if df_coordenadas is None or df_coordenadas.empty:
            print("✗ No se pudieron cargar las coordenadas")
            return []
        
        self.stats['total_registros'] = len(df_coordenadas)
        
        # Aplicar límite si se especifica
        if limite:
            df_coordenadas = df_coordenadas.head(limite)
            print(f"📋 Procesando {len(df_coordenadas)} imágenes (límite: {limite})")
        else:
            print(f"📋 Procesando {len(df_coordenadas)} imágenes")
        
        imagenes_procesadas = []
        
        # Procesar cada registro
        for idx, (_, row) in enumerate(df_coordenadas.iterrows()):
            # Mostrar progreso cada 50 imágenes
            mostrar_progreso = (idx % 50 == 0 and idx > 0) or idx < 5
            
            if mostrar_progreso:
                porcentaje = (idx / len(df_coordenadas)) * 100
                print(f"\n📊 Progreso: {idx}/{len(df_coordenadas)} ({porcentaje:.1f}%)")
            
            imagen_anotada, exito, tipo_imagen = self.procesar_imagen_individual(row, mostrar_progreso)
            if exito and imagen_anotada is not None:
                imagenes_procesadas.append((row['covid_id'], imagen_anotada, tipo_imagen))
        
        # Mostrar estadísticas
        self.mostrar_estadisticas_detalladas()
        
        # Mostrar muestra si se solicita
        if mostrar_muestra and imagenes_procesadas:
            self.crear_grid_comparacion(imagenes_procesadas[:12])
        
        return imagenes_procesadas
    
    def crear_grid_comparacion(self, imagenes_procesadas):
        """Crea un grid de comparación con las imágenes procesadas"""
        print("\n=== GENERANDO GRID DE COMPARACIÓN ===")
        
        # Organizar por tipo de imagen
        por_tipo = {'COVID': [], 'Normal': [], 'Viral Pneumonia': []}
        
        for covid_id, imagen, tipo_imagen in imagenes_procesadas:
            if tipo_imagen in por_tipo:
                por_tipo[tipo_imagen].append((covid_id, imagen))
        
        # Crear grid 4x3
        fig, axes = plt.subplots(4, 3, figsize=(15, 20))
        fig.suptitle('Coordenadas 299x299 - Comparación por Tipo de Imagen', fontsize=16)
        
        # Títulos de columnas
        columnas = ['COVID-19', 'Normal', 'Viral Pneumonia']
        for i, titulo in enumerate(columnas):
            axes[0, i].set_title(f'{titulo}', fontsize=14, fontweight='bold')
        
        # Llenar el grid
        for col, (tipo, imagenes) in enumerate(por_tipo.items()):
            for fila in range(4):
                ax = axes[fila, col]
                
                if fila < len(imagenes):
                    covid_id, imagen = imagenes[fila]
                    ax.imshow(imagen)
                    ax.set_title(f'{covid_id}', fontsize=8)
                else:
                    ax.text(0.5, 0.5, 'Sin imagen', ha='center', va='center', 
                           transform=ax.transAxes, fontsize=12, alpha=0.5)
                
                ax.axis('off')
        
        plt.tight_layout()
        
        # Guardar el grid
        grid_path = self.output_path / f"grid_comparacion.png"
        plt.savefig(grid_path, dpi=300, bbox_inches='tight')
        print(f"🖼️ Grid de comparación guardado en: {grid_path}")
        
        # Mostrar si estamos en un entorno interactivo
        try:
            plt.show()
        except:
            print("📱 Ejecute en un entorno interactivo para ver el grid")
        
        plt.close()
    
    def mostrar_estadisticas_detalladas(self):
        """Muestra estadísticas detalladas del procesamiento"""
        print("\n=== ESTADÍSTICAS DETALLADAS ===")
        print(f"Total de registros procesados: {self.stats['total_registros']}")
        print(f"Total de coordenadas visualizadas: {self.stats['total_coordenadas']}")
        
        # Estadísticas por categoría
        print("\n--- Por Categoría ---")
        for categoria, stats in self.stats['por_categoria'].items():
            total = stats['procesadas'] + stats['faltantes'] + stats['errores']
            if total > 0:
                porcentaje_exito = (stats['procesadas'] / total) * 100
                print(f"{categoria}:")
                print(f"  ✓ Procesadas: {stats['procesadas']}")
                print(f"  ⚠ Faltantes: {stats['faltantes']}")
                print(f"  ✗ Errores: {stats['errores']}")
                print(f"  📊 Éxito: {porcentaje_exito:.1f}%")
        
        # Guardar estadísticas
        self.guardar_reporte_estadisticas()
    
    def guardar_reporte_estadisticas(self):
        """Guarda un reporte detallado de las estadísticas"""
        reporte = {
            'timestamp': datetime.now().isoformat(),
            'estadisticas': self.stats,
            'resumen': {
                'total_registros': self.stats['total_registros'],
                'total_coordenadas': self.stats['total_coordenadas'],
                'por_categoria': {}
            }
        }
        
        # Calcular resumen por categoría
        for categoria, stats in self.stats['por_categoria'].items():
            total = stats['procesadas'] + stats['faltantes'] + stats['errores']
            if total > 0:
                reporte['resumen']['por_categoria'][categoria] = {
                    'total': total,
                    'exito_porcentaje': (stats['procesadas'] / total) * 100,
                    'faltantes_porcentaje': (stats['faltantes'] / total) * 100,
                    'errores_porcentaje': (stats['errores'] / total) * 100
                }
        
        # Crear directorio si no existe
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        reporte_path = self.output_path / "reporte_estadisticas.json"
        
        with open(reporte_path, 'w', encoding='utf-8') as f:
            json.dump(reporte, f, indent=2, ensure_ascii=False)
        
        print(f"📊 Reporte guardado en: {reporte_path}")
    
    def procesar_imagenes_especificas(self, covid_ids):
        """Procesa una lista específica de imágenes"""
        print(f"=== PROCESANDO IMÁGENES ESPECÍFICAS ===")
        print(f"Imágenes a procesar: {covid_ids}")
        
        # Cargar todos los datasets para buscar las imágenes
        datasets = ['maestro', 'entrenamiento', 'prueba']
        registros_encontrados = []
        
        for dataset in datasets:
            df = self.cargar_coordenadas_dataset(dataset)
            if df is not None:
                for covid_id in covid_ids:
                    registro = df[df['covid_id'] == covid_id]
                    if not registro.empty:
                        registros_encontrados.append(registro.iloc[0])
                        print(f"✓ Encontrado {covid_id} en dataset {dataset}")
        
        if not registros_encontrados:
            print("✗ No se encontraron las imágenes especificadas")
            return []
        
        # Procesar las imágenes encontradas
        imagenes_procesadas = []
        for registro in registros_encontrados:
            imagen_anotada, exito, tipo_imagen = self.procesar_imagen_individual(registro, True)
            if exito and imagen_anotada is not None:
                imagenes_procesadas.append((registro['covid_id'], imagen_anotada, tipo_imagen))
        
        # Mostrar estadísticas
        self.mostrar_estadisticas_detalladas()
        
        return imagenes_procesadas


def main():
    """Función principal con argumentos de línea de comandos"""
    parser = argparse.ArgumentParser(description='Visualizador de coordenadas 299x299')
    parser.add_argument('--dataset', choices=['maestro', 'entrenamiento', 'prueba'], 
                       default='maestro', help='Dataset a procesar')
    parser.add_argument('--limite', type=int, help='Límite de imágenes a procesar')
    parser.add_argument('--imagenes', nargs='+', help='Imágenes específicas a procesar (ej: COVID-269 Normal-1023)')
    parser.add_argument('--sin-grid', action='store_true', help='No generar grid de comparación')
    
    args = parser.parse_args()
    
    visualizador = VisualizadorCoordenadas299x299()
    
    if args.imagenes:
        # Procesar imágenes específicas
        imagenes_procesadas = visualizador.procesar_imagenes_especificas(args.imagenes)
        if imagenes_procesadas and not args.sin_grid:
            visualizador.crear_grid_comparacion(imagenes_procesadas)
    else:
        # Procesar dataset completo
        imagenes_procesadas = visualizador.procesar_dataset(
            dataset=args.dataset, 
            limite=args.limite, 
            mostrar_muestra=not args.sin_grid
        )
    
    print(f"\n✅ Procesamiento completado. {len(imagenes_procesadas)} imágenes procesadas.")
    print(f"📁 Revisa los resultados en: {visualizador.output_path}")


if __name__ == "__main__":
    main()