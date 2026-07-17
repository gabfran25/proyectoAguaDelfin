import os
import glob
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import RobustScaler

# --- CONFIGURACIÓN DE RUTAS ---
CARPETA_ENTRADA = "ArchivosRenombrados" 
CARPETA_SALIDA = "CincoNormalizados"
ARCHIVO_ESCALADOR = "scaler_robust_5_cols.pkl"

# Las 5 columnas que SÍ se van a normalizar
COLUMNAS_OBJETIVO = ["Rate", "Header_Length", "ack_flag_number", "IAT", "Min"]

# Todas las cabeceras originales (39 en total)
CABECERAS = [
    "Header_Length", "Protocol Type", "Time_To_Live", "Rate", "fin_flag_number", 
    "syn_flag_number", "rst_flag_number", "psh_flag_number", "ack_flag_number", 
    "ece_flag_number", "cwr_flag_number", "ack_count", "syn_count", "fin_count", 
    "rst_count", "HTTP", "HTTPS", "DNS", "Telnet", "SMTP", 
    "SSH", "IRC", "TCP", "UDP", "DHCP", 
    "ARP", "ICMP", "IGMP", "IPv", "LLC", 
    "Tot sum", "Min", "Max", "AVG", "Std", 
    "Tot size", "IAT", "Number", "Variance"
]

# Asegurar que la carpeta CincoNormalizados exista
os.makedirs(CARPETA_SALIDA, exist_ok=True)

# 1. OBTENER LAS RUTAS DE LOS ARCHIVOS PCAP.CSV
archivos = glob.glob(os.path.join(CARPETA_ENTRADA, "BenignTraffic*.csv"))
if not archivos:
    archivos = glob.glob(os.path.join(CARPETA_ENTRADA, "*.csv"))

print(f"📂 Archivos encontrados: {[os.path.basename(f) for f in archivos]}")

# 2. CARGAR Y UNIR EL TRÁFICO PARA EL AJUSTE GLOBAL
print("\n⏳ Uniendo datos temporalmente para ajustar el RobustScaler globalmente...")
lista_dfs = []
for archivo in archivos:
    # Leemos el archivo completo con las 39 columnas
    df_temp = pd.read_csv(archivo, sep=';', names=CABECERAS, header=0)
    lista_dfs.append(df_temp)

df_completo = pd.concat(lista_dfs, ignore_index=True)

# --- LIMPIEZA GENERAL (Necesario para evitar errores con Rate e IAT) ---
print("🧹 Limpiando valores infinitos (inf) o nulos...")
df_completo = df_completo.apply(pd.to_numeric, errors='coerce')
df_completo = df_completo.replace([np.inf, -np.inf], np.nan)
df_completo = df_completo.dropna()

# 3. CONFIGURAR Y AJUSTAR EL ESCALADOR SOLO CON LAS 5 COLUMNAS
print(f"⚙️ Ajustando RobustScaler estrictamente para: {COLUMNAS_OBJETIVO}...")
scaler = RobustScaler()
# OJO AQUÍ: Solo le pasamos las 5 columnas al método fit
scaler.fit(df_completo[COLUMNAS_OBJETIVO])

joblib.dump(scaler, ARCHIVO_ESCALADOR)
print(f"💾 ¡Escalador guardado con éxito como '{ARCHIVO_ESCALADOR}'!")

# 4. TRANSFORMAR Y GUARDAR EN 'CincoNormalizados'
print("\n🔄 Normalizando y guardando archivos individuales (manteniendo 39 columnas)...")
for archivo in archivos:
    nombre_archivo = os.path.basename(archivo)
    
    # Leemos el archivo completo (39 columnas)
    df_original = pd.read_csv(archivo, sep=';', names=CABECERAS, header=0)
    
    # Aplicamos la misma limpieza para no romper la transformación
    df_original = df_original.apply(pd.to_numeric, errors='coerce')
    df_original = df_original.replace([np.inf, -np.inf], np.nan)
    df_original = df_original.dropna()
    
    # LA MAGIA: Transformamos solo las 5 columnas y las sobreescribimos en el DataFrame original
    df_original[COLUMNAS_OBJETIVO] = scaler.transform(df_original[COLUMNAS_OBJETIVO])
    
    # Guardamos el archivo final (que sigue teniendo las 39 columnas)
    ruta_salida = os.path.join(CARPETA_SALIDA, nombre_archivo)
    df_original.to_csv(ruta_salida, index=False)
    print(f"   [OK] Guardado: {ruta_salida}")

print("\n🚀 ¡Proceso completado! Tus archivos tienen 39 columnas, pero solo 5 fueron normalizadas.")