# -*- coding: utf-8 -*-
"""
Script para análisis estadístico completo de coordenadas 299x299
Genera estadísticas detalladas, detecta anomalías y crea reportes
"""

import pandas as pd
import numpy as np
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

class AnalizadorEstadisticasCoordenadas:
    def __init__(self):
        self.base_path = Path(__file__).parent.parent
        self.data_path = self.base_path / "data"
        self.coordenadas_path = self.data_path / "coordenadas_299x299"
        self.output_path = self.base_path / "estadisticas_coordenadas"
        
        # Crear directorio de salida si no existe
        self.output_path.mkdir(exist_ok=True)
        
        # Configuración de análisis
        self.rango_valido = (0, 299)  # Rango válido para coordenadas 299x299
        self.num_puntos = 15  # 15 pares de coordenadas por imagen
        
        # Categorías médicas reconocidas
        self.categorias_medicas = {
            'COVID': ['COVID'],
            'Normal': ['Normal'],  
            'Viral Pneumonia': ['Viral Pneumonia', 'Viral-Pneumonia']
        }
        
        # Estructura de estadísticas
        self.estadisticas = {}
        
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
        
        # Leer datos usando la función del módulo existente
        datos = leer_archivo_coordenadas_generico(archivo_coordenadas)
        
        if datos is None or datos.empty:
            raise ValueError("El archivo de coordenadas está vacío o no se pudo leer")
        
        print(f"Datos cargados: {len(datos)} registros")
        return datos
    
    def extraer_coordenadas_numericas(self, datos):
        """Extrae todas las coordenadas numéricas de los datos"""
        coordenadas = []
        coordenadas_x = []
        coordenadas_y = []
        
        # El formato del DataFrame devuelto por leer_archivo_coordenadas_generico tiene:
        # columnas: 'indice', 'coordenadas', 'num_coordenadas', 'covid_id'
        # donde 'coordenadas' es una lista de tuplas (x, y)
        
        for _, fila in datos.iterrows():
            fila_coords = []
            fila_x = []
            fila_y = []
            
            # Las coordenadas están almacenadas como lista de tuplas (x,y)
            for x, y in fila['coordenadas']:
                fila_coords.extend([float(x), float(y)])
                fila_x.append(float(x))
                fila_y.append(float(y))
            
            coordenadas.append(fila_coords)
            coordenadas_x.append(fila_x)
            coordenadas_y.append(fila_y)
        
        return np.array(coordenadas), np.array(coordenadas_x), np.array(coordenadas_y)
    
    def calcular_estadisticas_basicas(self, coordenadas, coordenadas_x, coordenadas_y):
        """Calcula estadísticas descriptivas básicas"""
        stats = {}
        
        # Todas las coordenadas juntas
        coords_flat = coordenadas.flatten()
        
        stats['total_puntos'] = len(coords_flat)
        stats['min_global'] = float(np.min(coords_flat))
        stats['max_global'] = float(np.max(coords_flat))
        stats['media_global'] = float(np.mean(coords_flat))
        stats['mediana_global'] = float(np.median(coords_flat))
        stats['desviacion_std_global'] = float(np.std(coords_flat))
        
        # Percentiles
        percentiles = [5, 10, 25, 50, 75, 90, 95, 99]
        stats['percentiles'] = {}
        for p in percentiles:
            stats['percentiles'][f'p{p}'] = float(np.percentile(coords_flat, p))
        
        # Estadísticas por eje
        x_flat = coordenadas_x.flatten()
        y_flat = coordenadas_y.flatten()
        
        stats['coordenadas_x'] = {
            'min': float(np.min(x_flat)),
            'max': float(np.max(x_flat)),
            'media': float(np.mean(x_flat)),
            'mediana': float(np.median(x_flat)),
            'desviacion_std': float(np.std(x_flat))
        }
        
        stats['coordenadas_y'] = {
            'min': float(np.min(y_flat)),
            'max': float(np.max(y_flat)),
            'media': float(np.mean(y_flat)),
            'mediana': float(np.median(y_flat)),
            'desviacion_std': float(np.std(y_flat))
        }
        
        return stats
    
    def detectar_anomalias(self, coordenadas, datos):
        """Detecta coordenadas fuera de rango y otros valores anómalos"""
        anomalias = {
            'fuera_de_rango': [],
            'valores_extremos': [],
            'registros_problematicos': [],
            'estadisticas_anomalias': {}
        }
        
        coords_flat = coordenadas.flatten()
        
        # Detectar valores fuera del rango válido
        fuera_rango = (coords_flat < self.rango_valido[0]) | (coords_flat > self.rango_valido[1])
        anomalias['estadisticas_anomalias']['total_fuera_rango'] = int(np.sum(fuera_rango))
        anomalias['estadisticas_anomalias']['porcentaje_fuera_rango'] = float(np.sum(fuera_rango) / len(coords_flat) * 100)
        
        # Detectar outliers usando IQR
        Q1 = np.percentile(coords_flat, 25)
        Q3 = np.percentile(coords_flat, 75)
        IQR = Q3 - Q1
        limite_inferior = Q1 - 1.5 * IQR
        limite_superior = Q3 + 1.5 * IQR
        
        outliers = (coords_flat < limite_inferior) | (coords_flat > limite_superior)
        anomalias['estadisticas_anomalias']['total_outliers'] = int(np.sum(outliers))
        anomalias['estadisticas_anomalias']['porcentaje_outliers'] = float(np.sum(outliers) / len(coords_flat) * 100)
        
        # Analizar registros específicos con problemas
        for idx, (_, fila) in enumerate(datos.iterrows()):
            covid_id = fila['covid_id']
            coords_fila = coordenadas[idx]
            
            # Verificar si hay coordenadas fuera de rango en esta fila
            fuera_rango_fila = (coords_fila < self.rango_valido[0]) | (coords_fila > self.rango_valido[1])
            if np.any(fuera_rango_fila):
                anomalias['fuera_de_rango'].append({
                    'covid_id': covid_id,
                    'indice': int(fila['indice']),
                    'coordenadas_problematicas': [float(c) for c in coords_fila[fuera_rango_fila]]
                })
            
            # Verificar outliers en esta fila
            outliers_fila = (coords_fila < limite_inferior) | (coords_fila > limite_superior)
            if np.any(outliers_fila):
                anomalias['valores_extremos'].append({
                    'covid_id': covid_id,
                    'indice': int(fila['indice']),
                    'valores_extremos': [float(c) for c in coords_fila[outliers_fila]]
                })
        
        return anomalias
    
    def analizar_por_categoria(self, datos, coordenadas, coordenadas_x, coordenadas_y):
        """Analiza estadísticas por categoría médica"""
        stats_categoria = {}
        
        for _, fila in datos.iterrows():
            categoria = self.determinar_categoria_medica(fila['covid_id'])
            if categoria not in stats_categoria:
                stats_categoria[categoria] = {
                    'count': 0,
                    'coordenadas': [],
                    'coordenadas_x': [],
                    'coordenadas_y': []
                }
            
            idx = fila.name
            stats_categoria[categoria]['count'] += 1
            stats_categoria[categoria]['coordenadas'].extend(coordenadas[idx])
            stats_categoria[categoria]['coordenadas_x'].extend(coordenadas_x[idx])
            stats_categoria[categoria]['coordenadas_y'].extend(coordenadas_y[idx])
        
        # Calcular estadísticas para cada categoría
        resultado = {}
        for categoria, data in stats_categoria.items():
            if data['count'] > 0:
                coords = np.array(data['coordenadas'])
                coords_x = np.array(data['coordenadas_x'])
                coords_y = np.array(data['coordenadas_y'])
                
                resultado[categoria] = {
                    'total_imagenes': data['count'],
                    'total_coordenadas': len(coords),
                    'min': float(np.min(coords)),
                    'max': float(np.max(coords)),
                    'media': float(np.mean(coords)),
                    'mediana': float(np.median(coords)),
                    'desviacion_std': float(np.std(coords)),
                    'coordenadas_x': {
                        'min': float(np.min(coords_x)),
                        'max': float(np.max(coords_x)),
                        'media': float(np.mean(coords_x)),
                        'desviacion_std': float(np.std(coords_x))
                    },
                    'coordenadas_y': {
                        'min': float(np.min(coords_y)),
                        'max': float(np.max(coords_y)),
                        'media': float(np.mean(coords_y)),
                        'desviacion_std': float(np.std(coords_y))
                    }
                }
        
        return resultado
    
    def generar_resumen_por_imagen(self, datos, coordenadas):
        """Genera un resumen estadístico por cada imagen"""
        resumen = []
        
        for idx, (_, fila) in enumerate(datos.iterrows()):
            coords = coordenadas[idx]
            categoria = self.determinar_categoria_medica(fila['covid_id'])
            
            resumen.append({
                'indice': int(fila['indice']),
                'COVID_ID': fila['covid_id'],
                'categoria': categoria,
                'min_coord': float(np.min(coords)),
                'max_coord': float(np.max(coords)),
                'media_coord': float(np.mean(coords)),
                'desviacion_std': float(np.std(coords)),
                'coordenadas_fuera_rango': int(np.sum((coords < self.rango_valido[0]) | (coords > self.rango_valido[1])))
            })
        
        return resumen
    
    def analizar_dataset(self, nombre_dataset):
        """Análisis completo de un dataset específico"""
        print(f"\n=== Analizando dataset: {nombre_dataset} ===")
        
        # Determinar archivo de coordenadas
        archivo_coordenadas = self.coordenadas_path / f"coordenadas_{nombre_dataset}_299x299.csv"
        
        # Cargar datos
        datos = self.cargar_datos(archivo_coordenadas)
        
        # Extraer coordenadas numéricas
        coordenadas, coordenadas_x, coordenadas_y = self.extraer_coordenadas_numericas(datos)
        
        # Información general
        info_general = {
            'dataset': nombre_dataset,
            'total_registros': len(datos),
            'total_coordenadas': coordenadas.size,
            'puntos_por_imagen': self.num_puntos,
            'fecha_analisis': datetime.now().isoformat(),
            'archivo_fuente': str(archivo_coordenadas)
        }
        
        # Estadísticas básicas
        stats_basicas = self.calcular_estadisticas_basicas(coordenadas, coordenadas_x, coordenadas_y)
        
        # Análisis por categoría
        stats_categoria = self.analizar_por_categoria(datos, coordenadas, coordenadas_x, coordenadas_y)
        
        # Detección de anomalías
        anomalias = self.detectar_anomalias(coordenadas, datos)
        
        # Resumen por imagen
        resumen_imagenes = self.generar_resumen_por_imagen(datos, coordenadas)
        
        # Consolidar resultados
        resultado = {
            'informacion_general': info_general,
            'estadisticas_basicas': stats_basicas,
            'estadisticas_por_categoria': stats_categoria,
            'anomalias': anomalias,
            'resumen_por_imagen': resumen_imagenes
        }
        
        return resultado
    
    def exportar_resultados(self, resultado, nombre_dataset):
        """Exporta los resultados en múltiples formatos"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Exportar JSON completo
        archivo_json = self.output_path / f"estadisticas_{nombre_dataset}_{timestamp}.json"
        with open(archivo_json, 'w', encoding='utf-8') as f:
            json.dump(resultado, f, indent=2, ensure_ascii=False)
        print(f"Estadísticas exportadas a: {archivo_json}")
        
        # Exportar CSV con resumen por imagen
        archivo_csv = self.output_path / f"resumen_imagenes_{nombre_dataset}_{timestamp}.csv"
        df_resumen = pd.DataFrame(resultado['resumen_por_imagen'])
        df_resumen.to_csv(archivo_csv, index=False, encoding='utf-8')
        print(f"Resumen por imagen exportado a: {archivo_csv}")
        
        # Exportar reporte de anomalías
        archivo_anomalias = self.output_path / f"reporte_anomalias_{nombre_dataset}_{timestamp}.txt"
        self.generar_reporte_anomalias(resultado['anomalias'], archivo_anomalias)
        print(f"Reporte de anomalías exportado a: {archivo_anomalias}")
        
        # Exportar resumen ejecutivo
        archivo_resumen = self.output_path / f"resumen_ejecutivo_{nombre_dataset}_{timestamp}.txt"
        self.generar_resumen_ejecutivo(resultado, archivo_resumen)
        print(f"Resumen ejecutivo exportado a: {archivo_resumen}")
        
        return {
            'json': archivo_json,
            'csv': archivo_csv,
            'anomalias': archivo_anomalias,
            'resumen': archivo_resumen
        }
    
    def generar_reporte_anomalias(self, anomalias, archivo_salida):
        """Genera un reporte detallado de anomalías"""
        with open(archivo_salida, 'w', encoding='utf-8') as f:
            f.write("REPORTE DE ANOMALÍAS EN COORDENADAS\n")
            f.write("=" * 50 + "\n\n")
            
            stats = anomalias['estadisticas_anomalias']
            f.write(f"RESUMEN DE ANOMALÍAS:\n")
            f.write(f"- Coordenadas fuera de rango (0-299): {stats['total_fuera_rango']} ({stats['porcentaje_fuera_rango']:.2f}%)\n")
            f.write(f"- Valores extremos (outliers): {stats['total_outliers']} ({stats['porcentaje_outliers']:.2f}%)\n\n")
            
            if anomalias['fuera_de_rango']:
                f.write("COORDENADAS FUERA DE RANGO:\n")
                f.write("-" * 30 + "\n")
                for item in anomalias['fuera_de_rango']:
                    f.write(f"ID: {item['covid_id']}, Índice: {item['indice']}\n")
                    f.write(f"Coordenadas problemáticas: {item['coordenadas_problematicas']}\n\n")
            
            if anomalias['valores_extremos']:
                f.write("VALORES EXTREMOS (OUTLIERS):\n")
                f.write("-" * 30 + "\n")
                for item in anomalias['valores_extremos'][:20]:  # Limitar a 20 para no saturar
                    f.write(f"ID: {item['covid_id']}, Índice: {item['indice']}\n")
                    f.write(f"Valores extremos: {item['valores_extremos']}\n\n")
                
                if len(anomalias['valores_extremos']) > 20:
                    f.write(f"... y {len(anomalias['valores_extremos']) - 20} registros más\n")
    
    def generar_resumen_ejecutivo(self, resultado, archivo_salida):
        """Genera un resumen ejecutivo del análisis"""
        with open(archivo_salida, 'w', encoding='utf-8') as f:
            info = resultado['informacion_general']
            stats = resultado['estadisticas_basicas']
            categorias = resultado['estadisticas_por_categoria']
            anomalias = resultado['anomalias']['estadisticas_anomalias']
            
            f.write("RESUMEN EJECUTIVO - ANÁLISIS DE COORDENADAS\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"DATASET: {info['dataset']}\n")
            f.write(f"Fecha de análisis: {info['fecha_analisis']}\n")
            f.write(f"Total de registros: {info['total_registros']}\n")
            f.write(f"Total de coordenadas: {info['total_coordenadas']}\n\n")
            
            f.write("ESTADÍSTICAS GLOBALES:\n")
            f.write(f"- Rango: {stats['min_global']:.2f} - {stats['max_global']:.2f}\n")
            f.write(f"- Media: {stats['media_global']:.2f}\n")
            f.write(f"- Mediana: {stats['mediana_global']:.2f}\n")
            f.write(f"- Desviación estándar: {stats['desviacion_std_global']:.2f}\n\n")
            
            f.write("ESTADÍSTICAS POR EJE:\n")
            f.write(f"Eje X - Rango: {stats['coordenadas_x']['min']:.2f} - {stats['coordenadas_x']['max']:.2f}, Media: {stats['coordenadas_x']['media']:.2f}\n")
            f.write(f"Eje Y - Rango: {stats['coordenadas_y']['min']:.2f} - {stats['coordenadas_y']['max']:.2f}, Media: {stats['coordenadas_y']['media']:.2f}\n\n")
            
            f.write("DISTRIBUCIÓN POR CATEGORÍA MÉDICA:\n")
            for categoria, data in categorias.items():
                f.write(f"{categoria}:\n")
                f.write(f"  - Imágenes: {data['total_imagenes']}\n")
                f.write(f"  - Rango: {data['min']:.2f} - {data['max']:.2f}\n")
                f.write(f"  - Media: {data['media']:.2f}\n\n")
            
            f.write("CALIDAD DE DATOS:\n")
            f.write(f"- Coordenadas fuera de rango: {anomalias['total_fuera_rango']} ({anomalias['porcentaje_fuera_rango']:.2f}%)\n")
            f.write(f"- Valores extremos: {anomalias['total_outliers']} ({anomalias['porcentaje_outliers']:.2f}%)\n")

def main():
    parser = argparse.ArgumentParser(description='Análisis estadístico de coordenadas 299x299')
    parser.add_argument('--dataset', 
                       choices=['maestro', 'entrenamiento', 'prueba'], 
                       default='entrenamiento',
                       help='Dataset a analizar (default: entrenamiento)')
    parser.add_argument('--todos', 
                       action='store_true',
                       help='Analizar todos los datasets disponibles')
    parser.add_argument('--verbose', 
                       action='store_true',
                       help='Mostrar información detallada durante el análisis')
    
    args = parser.parse_args()
    
    analizador = AnalizadorEstadisticasCoordenadas()
    
    try:
        if args.todos:
            datasets = ['maestro', 'entrenamiento', 'prueba']
            print("Analizando todos los datasets disponibles...")
        else:
            datasets = [args.dataset]
        
        resultados_todos = {}
        
        for dataset in datasets:
            try:
                resultado = analizador.analizar_dataset(dataset)
                archivos_exportados = analizador.exportar_resultados(resultado, dataset)
                resultados_todos[dataset] = resultado
                
                if args.verbose:
                    print(f"\nRESUMEN PARA {dataset.upper()}:")
                    info = resultado['informacion_general']
                    stats = resultado['estadisticas_basicas']
                    print(f"  Registros: {info['total_registros']}")
                    print(f"  Rango global: {stats['min_global']:.2f} - {stats['max_global']:.2f}")
                    print(f"  Media global: {stats['media_global']:.2f}")
                    
            except Exception as e:
                print(f"Error analizando dataset '{dataset}': {e}")
                continue
        
        print(f"\n✅ Análisis completado para {len(resultados_todos)} dataset(s)")
        print(f"📁 Resultados guardados en: {analizador.output_path}")
        
    except Exception as e:
        print(f"❌ Error durante el análisis: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())