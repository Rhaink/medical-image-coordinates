# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
from pathlib import Path

def leer_coordenadas_maestro():
    """
    Lee el archivo CSV de coordenadas maestro y devuelve un DataFrame estructurado.
    
    Returns:
        pd.DataFrame: DataFrame con las coordenadas y metadatos organizados
    """
    # Ruta al archivo CSV
    archivo_csv = Path(__file__).parent.parent / "data" / "coordenadas" / "coordenadas_maestro_1.csv"
    
    try:
        # Verificar que el archivo existe
        if not archivo_csv.exists():
            raise FileNotFoundError(f"El archivo {archivo_csv} no existe")
        
        # Leer el archivo CSV
        df = pd.read_csv(archivo_csv, header=None)
        
        # Analizar la estructura de los datos
        # Cada fila contiene: �ndice, coordenadas (m�ltiples pares), identificador COVID
        coordenadas_procesadas = []
        
        for index, row in df.iterrows():
            # Convertir la fila a lista y remover valores NaN
            datos = [str(val) for val in row.values if pd.notna(val)]
            
            if len(datos) < 2:
                continue
                
            # El primer valor es el �ndice, el �ltimo es el identificador COVID
            indice = datos[0]
            covid_id = datos[-1]
            
            # Los valores del medio son las coordenadas
            coordenadas_raw = datos[1:-1]
            
            # Convertir coordenadas a pares (x, y)
            coordenadas_pares = []
            for i in range(0, len(coordenadas_raw), 2):
                if i + 1 < len(coordenadas_raw):
                    try:
                        x = float(coordenadas_raw[i])
                        y = float(coordenadas_raw[i + 1])
                        coordenadas_pares.append((x, y))
                    except ValueError:
                        continue
            
            coordenadas_procesadas.append({
                'indice': indice,
                'coordenadas': coordenadas_pares,
                'num_coordenadas': len(coordenadas_pares),
                'covid_id': covid_id
            })
        
        # Crear DataFrame final
        df_resultado = pd.DataFrame(coordenadas_procesadas)
        
        print(f" Archivo le�do exitosamente: {archivo_csv}")
        print(f" Total de registros procesados: {len(df_resultado)}")
        
        return df_resultado
        
    except FileNotFoundError as e:
        print(f" Error: {e}")
        return None
    except Exception as e:
        print(f" Error inesperado al leer el archivo: {e}")
        return None

def mostrar_resumen_datos(df):
    """
    Muestra un resumen de los datos cargados.
    
    Args:
        df (pd.DataFrame): DataFrame con los datos de coordenadas
    """
    if df is None or df.empty:
        print("No hay datos para mostrar")
        return
    
    print("\n=== RESUMEN DE DATOS ===")
    print(f"Total de registros: {len(df)}")
    print(f"Promedio de coordenadas por registro: {df['num_coordenadas'].mean():.2f}")
    print(f"M�nimo de coordenadas: {df['num_coordenadas'].min()}")
    print(f"M�ximo de coordenadas: {df['num_coordenadas'].max()}")
    
    print("\n=== PRIMEROS 5 REGISTROS ===")
    for i, row in df.head().iterrows():
        print(f"�ndice: {row['indice']}")
        print(f"COVID ID: {row['covid_id']}")
        print(f"Coordenadas ({row['num_coordenadas']} pares): {row['coordenadas'][:3]}...")
        print("-" * 50)

if __name__ == "__main__":
    # Ejemplo de uso
    print("Cargando datos de coordenadas maestro...")
    datos = leer_coordenadas_maestro()
    
    if datos is not None:
        mostrar_resumen_datos(datos)