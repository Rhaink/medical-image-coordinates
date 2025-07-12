# -*- coding: utf-8 -*-
"""
Script para visualizar coordenadas escaladas de 299x299 sobre las imágenes
y evaluar la calidad del etiquetado
"""

import pandas as pd
import numpy as np
import cv2
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from procesar_coordenadas import leer_archivo_coordenadas_generico
import json
from datetime import datetime

class VisualizadorCoordenadas:
    def __init__(self):
        self.base_path = Path(__file__).parent.parent
        self.data_path = self.base_path / "data"
        self.covid_images_path = self.data_path / "COVID-19_Radiography_Dataset" / "COVID" / "images"
        self.viral_images_path = self.data_path / "COVID-19_Radiography_Dataset" / "Viral Pneumonia" / "images"
        self.normal_images_path = self.data_path / "COVID-19_Radiography_Dataset" / "Normal" / "images"
        self.coordenadas_path = self.data_path / "coordenadas_299x299"
        self.output_path = self.base_path / "visualizations" / "coordenadas_overlays"
        
        # Estadísticas
        self.stats = {
            'total_registros': 0,
            'imagenes_encontradas': 0,
            'imagenes_faltantes': 0,
            'imagenes_procesadas': 0,
            'errores': 0,
            'archivos_faltantes': []
        }
    
    def cargar_imagen(self, covid_id):
        """
        Carga una imagen basada en su COVID ID
        
        Args:
            covid_id (str): ID de la imagen (ej: 'COVID-269', 'Viral Pneumonia-1331')
        
        Returns:
            numpy.ndarray: Imagen cargada o None si no se encuentra
        """
        try:
            # Determinar la ruta según el tipo de imagen
            if covid_id.startswith('COVID-'):
                imagen_path = self.covid_images_path / f"{covid_id}.png"
            elif covid_id.startswith('Viral Pneumonia-'):
                imagen_path = self.viral_images_path / f"{covid_id}.png"
            elif covid_id.startswith('Normal-'):
                imagen_path = self.normal_images_path / f"{covid_id}.png"
            else:
                print(f"⚠ Tipo de imagen no reconocido: {covid_id}")
                return None
            
            # Cargar imagen
            if imagen_path.exists():
                imagen = cv2.imread(str(imagen_path))
                if imagen is not None:
                    imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
                    return imagen
                else:
                    print(f"⚠ Error al cargar la imagen: {imagen_path}")
                    return None
            else:
                print(f"⚠ Imagen no encontrada: {imagen_path}")
                self.stats['archivos_faltantes'].append(str(imagen_path))
                return None
                
        except Exception as e:
            print(f"✗ Error al cargar {covid_id}: {e}")
            self.stats['errores'] += 1
            return None
    
    def redimensionar_imagen(self, imagen, target_size=(299, 299)):
        """
        Redimensiona la imagen al tamaño objetivo
        
        Args:
            imagen: Imagen a redimensionar
            target_size: Tamaño objetivo (ancho, alto)
        
        Returns:
            numpy.ndarray: Imagen redimensionada
        """
        if imagen is None:
            return None
        
        altura_actual, ancho_actual = imagen.shape[:2]
        
        # Si ya está en el tamaño correcto, no hacer nada
        if (ancho_actual, altura_actual) == target_size:
            return imagen
        
        # Redimensionar
        imagen_redimensionada = cv2.resize(imagen, target_size)
        
        # Informar si hubo redimensionamiento
        if (ancho_actual, altura_actual) != target_size:
            print(f"  📏 Redimensionado de {ancho_actual}x{altura_actual} a {target_size[0]}x{target_size[1]}")
        
        return imagen_redimensionada
    
    def dibujar_coordenadas(self, imagen, coordenadas, mostrar_numeros=True):
        """
        Dibuja las coordenadas sobre la imagen
        
        Args:
            imagen: Imagen sobre la que dibujar
            coordenadas: Lista de tuplas (x, y)
            mostrar_numeros: Si mostrar números en cada punto
        
        Returns:
            numpy.ndarray: Imagen con coordenadas dibujadas
        """
        if imagen is None or not coordenadas:
            return imagen
        
        # Crear una copia de la imagen
        imagen_anotada = imagen.copy()
        
        # Colores para los puntos y numeración
        color_puntos = (0, 255, 0)    # Verde para los puntos
        color_numeros = (0, 0, 255)   # Rojo para la numeración
        
        # Dibujar cada coordenada
        for i, (x, y) in enumerate(coordenadas):
            # Convertir a enteros
            x_int = int(round(x))
            y_int = int(round(y))
            
            # Verificar que las coordenadas estén dentro de la imagen
            altura, ancho = imagen_anotada.shape[:2]
            if 0 <= x_int < ancho and 0 <= y_int < altura:
                # Dibujar círculo verde
                cv2.circle(imagen_anotada, (x_int, y_int), 3, color_puntos, -1)
                cv2.circle(imagen_anotada, (x_int, y_int), 5, color_puntos, 1)
                
                # Dibujar número en rojo si se solicita
                if mostrar_numeros:
                    cv2.putText(imagen_anotada, str(i+1), (x_int+7, y_int-7), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, color_numeros, 1)
            else:
                print(f"  ⚠ Coordenada fuera de rango: ({x_int}, {y_int}) en imagen {ancho}x{altura}")
        
        return imagen_anotada
    
    def procesar_imagen_individual(self, registro):
        """
        Procesa una imagen individual con sus coordenadas
        
        Args:
            registro: Diccionario con datos del registro
        
        Returns:
            tuple: (imagen_anotada, exito)
        """
        covid_id = registro['covid_id']
        coordenadas = registro['coordenadas']
        
        # Solo mostrar mensaje cada 100 imágenes para reducir verbosidad
        if self.stats['imagenes_procesadas'] % 100 == 0:
            print(f"📸 Procesando: {covid_id}")
        
        # Cargar imagen
        imagen = self.cargar_imagen(covid_id)
        if imagen is None:
            self.stats['imagenes_faltantes'] += 1
            return None, False
        
        self.stats['imagenes_encontradas'] += 1
        
        # Redimensionar a 299x299
        imagen = self.redimensionar_imagen(imagen, (299, 299))
        
        # Dibujar coordenadas
        imagen_anotada = self.dibujar_coordenadas(imagen, coordenadas)
        
        # Guardar imagen anotada
        output_filename = f"{covid_id}_anotada.png"
        output_path = self.output_path / output_filename
        
        # Crear directorio si no existe
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convertir de RGB a BGR para guardar con OpenCV
        imagen_bgr = cv2.cvtColor(imagen_anotada, cv2.COLOR_RGB2BGR)
        
        if cv2.imwrite(str(output_path), imagen_bgr):
            # Solo mostrar mensaje de guardado cada 100 imágenes
            if self.stats['imagenes_procesadas'] % 100 == 0:
                print(f"  ✓ Guardada en: {output_path}")
            self.stats['imagenes_procesadas'] += 1
            return imagen_anotada, True
        else:
            print(f"  ✗ Error al guardar: {output_path}")
            self.stats['errores'] += 1
            return imagen_anotada, False
    
    def cargar_coordenadas_escaladas(self, archivo='coordenadas_maestro_299x299.csv'):
        """
        Carga las coordenadas escaladas desde el archivo CSV
        
        Args:
            archivo: Nombre del archivo CSV
        
        Returns:
            pd.DataFrame: DataFrame con las coordenadas
        """
        archivo_path = self.coordenadas_path / archivo
        return leer_archivo_coordenadas_generico(archivo_path)
    
    def procesar_lote_imagenes(self, limite=None, mostrar_muestra=True):
        """
        Procesa un lote de imágenes
        
        Args:
            limite: Número máximo de imágenes a procesar (None para todas)
            mostrar_muestra: Si mostrar una muestra de imágenes procesadas
        
        Returns:
            list: Lista de imágenes procesadas
        """
        print("=== VISUALIZADOR DE COORDENADAS ===\n")
        
        # Cargar datos de coordenadas
        df_coordenadas = self.cargar_coordenadas_escaladas()
        
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
        
        # Procesar cada registro con reporte de progreso
        for idx, (_, row) in enumerate(df_coordenadas.iterrows()):
            # Mostrar progreso cada 50 imágenes
            if idx % 50 == 0 and idx > 0:
                porcentaje = (idx / len(df_coordenadas)) * 100
                print(f"\n📊 Progreso: {idx}/{len(df_coordenadas)} ({porcentaje:.1f}%)")
            
            imagen_anotada, exito = self.procesar_imagen_individual(row)
            if exito and imagen_anotada is not None:
                imagenes_procesadas.append((row['covid_id'], imagen_anotada))
        
        # Mostrar estadísticas
        self.mostrar_estadisticas()
        
        # Mostrar muestra si se solicita
        if mostrar_muestra and imagenes_procesadas:
            self.mostrar_muestra_imagenes(imagenes_procesadas[:6])
        
        return imagenes_procesadas
    
    def mostrar_estadisticas(self):
        """
        Muestra las estadísticas del procesamiento
        """
        print("\n=== ESTADÍSTICAS DE PROCESAMIENTO ===")
        print(f"Total de registros: {self.stats['total_registros']}")
        print(f"Imágenes encontradas: {self.stats['imagenes_encontradas']}")
        print(f"Imágenes faltantes: {self.stats['imagenes_faltantes']}")
        print(f"Imágenes procesadas exitosamente: {self.stats['imagenes_procesadas']}")
        print(f"Errores: {self.stats['errores']}")
        
        if self.stats['total_registros'] > 0:
            porcentaje_exito = (self.stats['imagenes_procesadas'] / self.stats['total_registros']) * 100
            print(f"Porcentaje de éxito: {porcentaje_exito:.1f}%")
        
        # Guardar estadísticas en archivo
        self.guardar_reporte_estadisticas()
    
    def guardar_reporte_estadisticas(self):
        """
        Guarda un reporte detallado de las estadísticas
        """
        reporte = {
            'timestamp': datetime.now().isoformat(),
            'estadisticas': self.stats,
            'resumen': {
                'total_registros': self.stats['total_registros'],
                'exito_porcentaje': (self.stats['imagenes_procesadas'] / self.stats['total_registros'] * 100) if self.stats['total_registros'] > 0 else 0,
                'imagenes_faltantes_porcentaje': (self.stats['imagenes_faltantes'] / self.stats['total_registros'] * 100) if self.stats['total_registros'] > 0 else 0
            }
        }
        
        # Crear directorio si no existe
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        reporte_path = self.output_path / "reporte_estadisticas.json"
        
        with open(reporte_path, 'w', encoding='utf-8') as f:
            json.dump(reporte, f, indent=2, ensure_ascii=False)
        
        print(f"📊 Reporte guardado en: {reporte_path}")
    
    def mostrar_muestra_imagenes(self, imagenes_muestra):
        """
        Muestra una muestra de imágenes procesadas
        
        Args:
            imagenes_muestra: Lista de tuplas (covid_id, imagen)
        """
        print("\n=== MUESTRA DE IMÁGENES PROCESADAS ===")
        
        # Configurar la figura
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Muestra de Imágenes con Coordenadas Superpuestas', fontsize=16)
        
        # Aplanar axes para facilitar iteración
        axes = axes.flatten()
        
        for i, (covid_id, imagen) in enumerate(imagenes_muestra[:6]):
            ax = axes[i]
            ax.imshow(imagen)
            ax.set_title(f"{covid_id}", fontsize=10)
            ax.axis('off')
        
        # Ocultar axes vacíos
        for i in range(len(imagenes_muestra), len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        # Guardar la figura
        muestra_path = self.output_path / "muestra_imagenes_procesadas.png"
        plt.savefig(muestra_path, dpi=150, bbox_inches='tight')
        print(f"🖼️ Muestra guardada en: {muestra_path}")
        
        # Mostrar si estamos en un entorno interactivo
        try:
            plt.show()
        except:
            print("📱 Ejecute en un entorno interactivo para ver la muestra")
        
        plt.close()

def main():
    """
    Función principal
    """
    visualizador = VisualizadorCoordenadas()
    
    # Procesar todas las imágenes del dataset
    imagenes_procesadas = visualizador.procesar_lote_imagenes(limite=None, mostrar_muestra=False)
    
    print(f"\n✅ Procesamiento completado. {len(imagenes_procesadas)} imágenes procesadas.")
    print(f"📁 Revisa los resultados en: {visualizador.output_path}")

if __name__ == "__main__":
    main()