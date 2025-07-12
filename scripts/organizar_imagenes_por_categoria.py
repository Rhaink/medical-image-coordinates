# -*- coding: utf-8 -*-
"""
Script para organizar las imágenes anotadas por categorías
COVID-19, Normal, y Viral Pneumonia en carpetas separadas
"""

import os
import shutil
from pathlib import Path
import argparse
from collections import defaultdict
import json
from datetime import datetime

class OrganizadorImagenes:
    def __init__(self, directorio_base=None):
        self.base_path = Path(__file__).parent.parent
        if directorio_base:
            self.directorio_base = Path(directorio_base)
        else:
            self.directorio_base = self.base_path / "visualizations" / "coordenadas_299x299_overlays"
        
        # Mapeo de categorías
        self.categorias = {
            'covid': {
                'patron': 'COVID-',
                'carpeta': 'covid'
            },
            'normal': {
                'patron': 'Normal-',
                'carpeta': 'normal'
            },
            'viral_pneumonia': {
                'patron': 'Viral Pneumonia-',
                'carpeta': 'viral-pneumonia'
            }
        }
        
        # Estadísticas
        self.stats = {
            'total_archivos': 0,
            'imagenes_procesadas': 0,
            'archivos_ignorados': 0,
            'errores': 0,
            'por_categoria': defaultdict(int),
            'archivos_sin_categoria': []
        }
    
    def detectar_categoria(self, nombre_archivo):
        """
        Detecta la categoría de una imagen basándose en su nombre
        
        Args:
            nombre_archivo (str): Nombre del archivo
            
        Returns:
            str: Categoría detectada o None si no coincide
        """
        for categoria, info in self.categorias.items():
            if nombre_archivo.startswith(info['patron']):
                return categoria
        return None
    
    def crear_estructura_carpetas(self, dry_run=False):
        """
        Crea la estructura de carpetas necesaria
        
        Args:
            dry_run (bool): Si True, solo simula la creación
        """
        print("=== CREANDO ESTRUCTURA DE CARPETAS ===")
        
        for categoria, info in self.categorias.items():
            carpeta_destino = self.directorio_base / info['carpeta']
            
            if dry_run:
                print(f"📁 [DRY-RUN] Crearía carpeta: {carpeta_destino}")
            else:
                carpeta_destino.mkdir(exist_ok=True)
                print(f"📁 Carpeta creada/verificada: {carpeta_destino}")
    
    def escanear_archivos(self):
        """
        Escanea todos los archivos en el directorio base
        
        Returns:
            dict: Archivos organizados por categoría
        """
        print(f"\n=== ESCANEANDO ARCHIVOS EN {self.directorio_base} ===")
        
        archivos_por_categoria = defaultdict(list)
        archivos_especiales = []
        
        # Obtener todos los archivos PNG
        archivos_png = list(self.directorio_base.glob("*.png"))
        self.stats['total_archivos'] = len(archivos_png)
        
        print(f"📊 Total de archivos PNG encontrados: {len(archivos_png)}")
        
        for archivo in archivos_png:
            nombre = archivo.name
            
            # Verificar si es un archivo especial (grid, reporte, etc.)
            if nombre.startswith('grid_') or nombre.startswith('reporte_'):
                archivos_especiales.append(archivo)
                continue
            
            # Detectar categoría
            categoria = self.detectar_categoria(nombre)
            
            if categoria:
                archivos_por_categoria[categoria].append(archivo)
                self.stats['por_categoria'][categoria] += 1
            else:
                self.stats['archivos_sin_categoria'].append(nombre)
                print(f"⚠ Archivo sin categoría reconocida: {nombre}")
        
        # Reportar archivos especiales
        if archivos_especiales:
            print(f"\n📋 Archivos especiales (no se moverán): {len(archivos_especiales)}")
            for archivo in archivos_especiales:
                print(f"  • {archivo.name}")
        
        return archivos_por_categoria
    
    def mover_archivos(self, archivos_por_categoria, dry_run=False, verbose=False):
        """
        Mueve los archivos a sus carpetas correspondientes
        
        Args:
            archivos_por_categoria (dict): Archivos organizados por categoría
            dry_run (bool): Si True, solo simula el movimiento
            verbose (bool): Si True, muestra detalle de cada archivo
        """
        print(f"\n=== {'SIMULANDO' if dry_run else 'MOVIENDO'} ARCHIVOS ===")
        
        for categoria, archivos in archivos_por_categoria.items():
            if not archivos:
                continue
                
            carpeta_destino = self.directorio_base / self.categorias[categoria]['carpeta']
            
            print(f"\n📂 Categoría: {categoria.upper()}")
            print(f"   Destino: {carpeta_destino}")
            print(f"   Archivos: {len(archivos)}")
            
            for archivo in archivos:
                destino = carpeta_destino / archivo.name
                
                try:
                    if dry_run:
                        if verbose:
                            print(f"  [DRY-RUN] {archivo.name} → {self.categorias[categoria]['carpeta']}/")
                    else:
                        shutil.move(str(archivo), str(destino))
                        if verbose:
                            print(f"  ✓ {archivo.name} → {self.categorias[categoria]['carpeta']}/")
                        
                    self.stats['imagenes_procesadas'] += 1
                    
                except Exception as e:
                    print(f"  ✗ Error moviendo {archivo.name}: {e}")
                    self.stats['errores'] += 1
            
            if not verbose and not dry_run:
                print(f"  ✓ {len(archivos)} archivos movidos exitosamente")
    
    def verificar_integridad(self):
        """
        Verifica que todos los archivos se hayan movido correctamente
        """
        print(f"\n=== VERIFICACIÓN DE INTEGRIDAD ===")
        
        # Contar archivos en cada carpeta
        total_movidos = 0
        
        for categoria, info in self.categorias.items():
            carpeta = self.directorio_base / info['carpeta']
            if carpeta.exists():
                archivos_en_carpeta = len(list(carpeta.glob("*.png")))
                esperados = self.stats['por_categoria'][categoria]
                
                print(f"📁 {categoria.upper()}:")
                print(f"   Esperados: {esperados}")
                print(f"   Encontrados: {archivos_en_carpeta}")
                print(f"   Estado: {'✓' if archivos_en_carpeta == esperados else '✗'}")
                
                total_movidos += archivos_en_carpeta
        
        # Verificar archivos restantes en directorio base
        archivos_restantes = len(list(self.directorio_base.glob("*.png")))
        archivos_especiales = len([f for f in self.directorio_base.glob("*.png") 
                                 if f.name.startswith(('grid_', 'reporte_'))])
        
        print(f"\n📊 RESUMEN:")
        print(f"   Total original: {self.stats['total_archivos']}")
        print(f"   Archivos movidos: {total_movidos}")
        print(f"   Archivos especiales (no movidos): {archivos_especiales}")
        print(f"   Archivos restantes: {archivos_restantes}")
        print(f"   Errores: {self.stats['errores']}")
        
        # Validación
        if total_movidos + archivos_especiales == self.stats['total_archivos']:
            print(f"✅ INTEGRIDAD VERIFICADA: Todos los archivos están en su lugar correcto")
        else:
            print(f"❌ PROBLEMA DE INTEGRIDAD: Faltan archivos")
    
    def mostrar_estadisticas(self):
        """
        Muestra las estadísticas del procesamiento
        """
        print(f"\n=== ESTADÍSTICAS FINALES ===")
        print(f"Total de archivos procesados: {self.stats['total_archivos']}")
        print(f"Imágenes organizadas: {self.stats['imagenes_procesadas']}")
        print(f"Archivos ignorados: {self.stats['archivos_ignorados']}")
        print(f"Errores: {self.stats['errores']}")
        
        print(f"\n--- Por Categoría ---")
        for categoria, cantidad in self.stats['por_categoria'].items():
            print(f"{categoria.upper()}: {cantidad} imágenes")
        
        if self.stats['archivos_sin_categoria']:
            print(f"\n⚠ Archivos sin categoría: {len(self.stats['archivos_sin_categoria'])}")
            for archivo in self.stats['archivos_sin_categoria']:
                print(f"  • {archivo}")
    
    def guardar_reporte(self):
        """
        Guarda un reporte detallado en JSON
        """
        reporte = {
            'timestamp': datetime.now().isoformat(),
            'directorio_base': str(self.directorio_base),
            'estadisticas': dict(self.stats),
            'estructura_creada': {
                categoria: info['carpeta'] 
                for categoria, info in self.categorias.items()
            }
        }
        
        reporte_path = self.directorio_base / "reporte_organizacion.json"
        
        with open(reporte_path, 'w', encoding='utf-8') as f:
            json.dump(reporte, f, indent=2, ensure_ascii=False)
        
        print(f"📊 Reporte guardado en: {reporte_path}")
    
    def organizar(self, dry_run=False, verbose=False):
        """
        Ejecuta el proceso completo de organización
        
        Args:
            dry_run (bool): Si True, solo simula las acciones
            verbose (bool): Si True, muestra detalle de cada archivo
        """
        if dry_run:
            print("🔍 MODO DRY-RUN ACTIVADO - Solo se simulan las acciones")
        
        print(f"=== ORGANIZADOR DE IMÁGENES POR CATEGORÍA ===")
        print(f"Directorio base: {self.directorio_base}")
        
        # Verificar que el directorio existe
        if not self.directorio_base.exists():
            print(f"❌ Error: El directorio {self.directorio_base} no existe")
            return False
        
        # 1. Crear estructura de carpetas
        self.crear_estructura_carpetas(dry_run)
        
        # 2. Escanear archivos
        archivos_por_categoria = self.escanear_archivos()
        
        # 3. Mover archivos
        self.mover_archivos(archivos_por_categoria, dry_run, verbose)
        
        # 4. Mostrar estadísticas
        self.mostrar_estadisticas()
        
        # 5. Verificar integridad (solo si no es dry-run)
        if not dry_run:
            self.verificar_integridad()
            self.guardar_reporte()
        
        print(f"\n✅ Organización {'simulada' if dry_run else 'completada'} exitosamente")
        return True


def main():
    """
    Función principal con argumentos de línea de comandos
    """
    parser = argparse.ArgumentParser(description='Organizador de imágenes por categoría')
    parser.add_argument('--source', type=str, help='Directorio fuente con las imágenes')
    parser.add_argument('--dry-run', action='store_true', 
                       help='Solo simular las acciones sin ejecutarlas')
    parser.add_argument('--verbose', action='store_true', 
                       help='Mostrar detalle de cada archivo procesado')
    
    args = parser.parse_args()
    
    # Crear organizador
    organizador = OrganizadorImagenes(args.source)
    
    # Ejecutar organización
    organizador.organizar(dry_run=args.dry_run, verbose=args.verbose)


if __name__ == "__main__":
    main()