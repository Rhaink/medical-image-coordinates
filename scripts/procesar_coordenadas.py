# -*- coding: utf-8 -*-
"""
Script para procesar y escalar coordenadas de 64x64 a 299x299
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
try:
    from .escalar_coordenadas import leer_coordenadas_maestro
except ImportError:
    from escalar_coordenadas import leer_coordenadas_maestro

def escalar_coordenadas_64_a_299(x, y):
    """
    Escala coordenadas de espacio 64x64 a 299x299
    
    Args:
        x (float): Coordenada x en espacio 64x64
        y (float): Coordenada y en espacio 64x64
    
    Returns:
        tuple: (x_escalada, y_escalada) en espacio 299x299
    """
    # Factor de escala: 299/64 = 4.671875
    factor_escala = 299.0 / 64.0
    
    x_escalada = x * factor_escala
    y_escalada = y * factor_escala
    
    return round(x_escalada, 2), round(y_escalada, 2)

def validar_coordenadas_entrada(x, y):
    """
    Valida que las coordenadas de entrada estén en el rango válido (0-64)
    
    Args:
        x, y: Coordenadas a validar
    
    Returns:
        bool: True si las coordenadas son válidas
    """
    return 0 <= x <= 64 and 0 <= y <= 64

def validar_coordenadas_salida(x, y):
    """
    Valida que las coordenadas de salida estén en el rango válido (0-299)
    
    Args:
        x, y: Coordenadas a validar
    
    Returns:
        bool: True si las coordenadas son válidas
    """
    return 0 <= x <= 299 and 0 <= y <= 299

def leer_archivo_coordenadas_generico(archivo_path):
    """
    Lee cualquier archivo CSV de coordenadas con la estructura estándar
    
    Args:
        archivo_path (Path): Ruta al archivo CSV
    
    Returns:
        pd.DataFrame: DataFrame con las coordenadas procesadas
    """
    try:
        if not archivo_path.exists():
            raise FileNotFoundError(f"El archivo {archivo_path} no existe")
        
        df = pd.read_csv(archivo_path, header=None)
        coordenadas_procesadas = []
        
        for index, row in df.iterrows():
            datos = [str(val) for val in row.values if pd.notna(val)]
            
            if len(datos) < 2:
                continue
                
            indice = datos[0]
            covid_id = datos[-1]
            coordenadas_raw = datos[1:-1]
            
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
        
        return pd.DataFrame(coordenadas_procesadas)
        
    except Exception as e:
        print(f"✗ Error al leer {archivo_path}: {e}")
        return None

def procesar_archivo_coordenadas(archivo_entrada, archivo_salida):
    """
    Procesa un archivo de coordenadas escalando de 64x64 a 299x299
    
    Args:
        archivo_entrada (Path): Ruta del archivo de entrada
        archivo_salida (Path): Ruta del archivo de salida
    
    Returns:
        bool: True si el procesamiento fue exitoso
    """
    print(f"Procesando: {archivo_entrada.name}")
    
    # Leer datos originales
    df_original = leer_archivo_coordenadas_generico(archivo_entrada)
    
    if df_original is None or df_original.empty:
        print(f"✗ No se pudieron leer los datos de {archivo_entrada}")
        return False
    
    # Procesar y escalar coordenadas
    registros_escalados = []
    coordenadas_fuera_rango = 0
    
    for _, row in df_original.iterrows():
        coordenadas_escaladas = []
        
        for x_orig, y_orig in row['coordenadas']:
            # Validar coordenadas de entrada
            if not validar_coordenadas_entrada(x_orig, y_orig):
                print(f"⚠ Advertencia: Coordenadas fuera de rango 64x64: ({x_orig}, {y_orig})")
                coordenadas_fuera_rango += 1
            
            # Escalar coordenadas
            x_escalada, y_escalada = escalar_coordenadas_64_a_299(x_orig, y_orig)
            
            # Validar coordenadas de salida
            if not validar_coordenadas_salida(x_escalada, y_escalada):
                print(f"⚠ Advertencia: Coordenadas escaladas fuera de rango 299x299: ({x_escalada}, {y_escalada})")
                coordenadas_fuera_rango += 1
            
            coordenadas_escaladas.append((x_escalada, y_escalada))
        
        # Crear fila para el archivo de salida
        fila_salida = [row['indice']]
        
        # Añadir coordenadas escaladas (aplanadas)
        for x_esc, y_esc in coordenadas_escaladas:
            fila_salida.extend([x_esc, y_esc])
        
        # Añadir COVID ID
        fila_salida.append(row['covid_id'])
        
        registros_escalados.append(fila_salida)
    
    # Crear DataFrame de salida
    df_salida = pd.DataFrame(registros_escalados)
    
    # Crear directorio de salida si no existe
    archivo_salida.parent.mkdir(parents=True, exist_ok=True)
    
    # Guardar archivo CSV
    df_salida.to_csv(archivo_salida, index=False, header=False)
    
    print(f"✓ Procesado exitosamente: {len(df_salida)} registros")
    print(f"✓ Archivo guardado en: {archivo_salida}")
    
    if coordenadas_fuera_rango > 0:
        print(f"⚠ Se encontraron {coordenadas_fuera_rango} coordenadas fuera de rango")
    
    return True

def procesar_todos_archivos():
    """
    Procesa todos los archivos de coordenadas disponibles
    """
    print("=== PROCESADOR DE COORDENADAS 64x64 → 299x299 ===\n")
    
    # Definir rutas de archivos
    base_path = Path(__file__).parent.parent
    dir_entrada = base_path / "data" / "coordenadas"
    dir_salida = base_path / "data" / "coordenadas_299x299"
    
    archivos_a_procesar = [
        ("coordenadas_maestro_1.csv", "coordenadas_maestro_299x299.csv"),
        ("coordenadas_entrenamiento_1.csv", "coordenadas_entrenamiento_299x299.csv"),
        ("coordenadas_prueba_1.csv", "coordenadas_prueba_299x299.csv")
    ]
    
    resultados = []
    
    for archivo_entrada, archivo_salida in archivos_a_procesar:
        ruta_entrada = dir_entrada / archivo_entrada
        ruta_salida = dir_salida / archivo_salida
        
        if ruta_entrada.exists():
            exito = procesar_archivo_coordenadas(ruta_entrada, ruta_salida)
            resultados.append((archivo_entrada, exito))
        else:
            print(f"✗ Archivo no encontrado: {archivo_entrada}")
            resultados.append((archivo_entrada, False))
        
        print("-" * 60)
    
    # Resumen final
    print("\n=== RESUMEN DEL PROCESAMIENTO ===")
    exitosos = sum(1 for _, exito in resultados if exito)
    total = len(resultados)
    
    print(f"Archivos procesados exitosamente: {exitosos}/{total}")
    
    for archivo, exito in resultados:
        status = "✓" if exito else "✗"
        print(f"{status} {archivo}")
    
    if exitosos > 0:
        print(f"\n✓ Archivos escalados guardados en: {dir_salida}")

def mostrar_ejemplo_escalado():
    """
    Muestra algunos ejemplos de escalado de coordenadas
    """
    print("\n=== EJEMPLOS DE ESCALADO ===")
    ejemplos = [
        (0, 0),
        (32, 32),
        (64, 64),
        (31, 12),
        (50, 59)
    ]
    
    print("Coordenadas 64x64 → 299x299")
    print("Original → Escalada")
    print("-" * 25)
    
    for x, y in ejemplos:
        x_esc, y_esc = escalar_coordenadas_64_a_299(x, y)
        print(f"({x:2}, {y:2}) → ({x_esc:6.2f}, {y_esc:6.2f})")

if __name__ == "__main__":
    # Mostrar ejemplos de escalado
    mostrar_ejemplo_escalado()
    
    # Procesar todos los archivos
    procesar_todos_archivos()