import os
import pandas as pd

# Definimos las rutas de las carpetas
CARPETA_ORIGEN = "./ArchivosBenignos"
CARPETA_DESTINO = "./ArchivosRenombrados"

def procesar_archivos():
    # Nos aseguramos de que la carpeta de destino exista
    if not os.path.exists(CARPETA_DESTINO):
        os.makedirs(CARPETA_DESTINO)
        print(f"📁 Creada la carpeta de destino: {CARPETA_DESTINO}")

    if not os.path.exists(CARPETA_ORIGEN):
        print(f"❌ Error: No se encontró la carpeta de origen '{CARPETA_ORIGEN}'")
        return

    # Listar todos los archivos CSV en la carpeta de origen
    archivos = [f for f in os.listdir(CARPETA_ORIGEN) if f.lower().endswith('.csv')]

    if not archivos:
        print(f"⚠️ No se encontraron archivos .csv en '{CARPETA_ORIGEN}'")
        return

    print(f"🚀 Iniciando estructuración de {len(archivos)} archivo(s).\n")

    for archivo in archivos:
        ruta_origen = os.path.join(CARPETA_ORIGEN, archivo)
        ruta_destino = os.path.join(CARPETA_DESTINO, archivo)

        try:
            # 1. Leemos el archivo original separado por comas
            df = pd.read_csv(ruta_origen, sep=',', skipinitialspace=True)

            # 2. Guardamos en la carpeta de destino usando PUNTO Y COMA (;) 
            # Esto corrige el problema con Excel en español y separa las celdas automáticamente.
            df.to_csv(ruta_destino, sep=';', index=False)
            print(f"  [OK] Estructurado en celdas: {archivo}")

        except Exception as e:
            print(f"  [ERROR] No se pudo procesar {archivo}. Motivo: {e}")

    print("\n¡Primer paso completado! Corre el script y abre el nuevo archivo en Excel para comprobarlo. 😎")

if __name__ == "__main__":
    procesar_archivos()