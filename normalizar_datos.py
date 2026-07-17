import os
import glob
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import RobustScaler

# --- CONFIGURACIÓN DE RUTAS ---
CARPETA_ENTRADA = "ArchivosRenombrados"
CARPETA_SALIDA = "ArchivosNormalizados"
ARCHIVO_ESCALADOR = "scaler_robust.pkl"

# Tus 39 cabeceras exactas
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

# Asegurar que la carpeta de salida exista
os.makedirs(CARPETA_SALIDA, exist_ok=True)

# 1. OBTENER LAS RUTAS DE LOS ARCHIVOS PCAP.CSV
archivos = glob.glob(os.path.join(CARPETA_ENTRADA, "BenignTraffic*.csv"))
if not archivos:
    archivos = glob.glob(os.path.join(CARPETA_ENTRADA, "*.csv"))

print(f"📂 Archivos encontrados para procesar: {[os.path.basename(f) for f in archivos]}")

# 2. CARGAR Y CONCATENAR TODO EL TRÁFICO BENIGNO (Usando sep=';')
print("\n⏳ Uniendo datos temporalmente para ajustar el RobustScaler globalmente...")
lista_dfs = []
for archivo in archivos:
    df_temp = pd.read_csv(archivo, sep=';', names=CABECERAS, header=0)
    lista_dfs.append(df_temp)

df_completo = pd.concat(lista_dfs, ignore_index=True)

# --- LIMPIEZA DE INFINITOS Y VALORES EXTRAÑOS ---
print("🧹 Limpiando valores infinitos (inf) o nulos...")
# Convertir todo a numérico por si quedó algún texto, forzando errores a NaN
df_completo = df_completo.apply(pd.to_numeric, errors='coerce')

# Reemplazar infinitos por NaN y eliminarlos
filas_antes = len(df_completo)
df_completo = df_completo.replace([np.inf, -np.inf], np.nan)
df_completo = df_completo.dropna()
filas_despues = len(df_completo)

print(f"   -> Se eliminaron {filas_antes - filas_despues} filas que contenían valores infinitos o corruptos.")

# 3. FILTRAR COLUMNAS CONSTANTES (Puro 0)
columnas_constantes = [col for col in df_completo.columns if df_completo[col].min() == df_completo[col].max()]

if columnas_constantes:
    print(f"⚠️ Columnas constantes detectadas y eliminadas: {columnas_constantes}\n")
    df_completo = df_completo.drop(columns=columnas_constantes)
else:
    print("✅ No se detectaron columnas constantes. ¡Excelente!")

columnas_finales = df_completo.columns.tolist()

# 4. CONFIGURAR Y AJUSTAR EL ESCALADOR
print("⚙️ Ajustando RobustScaler con los datos globales...")
scaler = RobustScaler()
scaler.fit(df_completo)

# Guardar el escalador
joblib.dump(scaler, ARCHIVO_ESCALADOR)
print(f"💾 ¡Escalador guardado con éxito como '{ARCHIVO_ESCALADOR}'!")

# 5. TRANSFORMAR CADA ARCHIVO INDIVIDUALMENTE Y GUARDARLO
print("\n🔄 Normalizando y guardando archivos individuales...")
for archivo in archivos:
    nombre_archivo = os.path.basename(archivo)
    df_original = pd.read_csv(archivo, sep=';', names=CABECERAS, header=0)
    
    # Aplicar la misma limpieza numérica al archivo individual
    df_original = df_original.apply(pd.to_numeric, errors='coerce')
    df_original = df_original.replace([np.inf, -np.inf], np.nan)
    df_original = df_original.dropna()
    
    if columnas_constantes:
        df_original = df_original.drop(columns=columnas_constantes, errors='ignore')
    
    # Transformar con el escalador ajustado
    datos_normalizados = scaler.transform(df_original)
    df_normalizado = pd.DataFrame(datos_normalizados, columns=columnas_finales)
    
    # Guardamos en la nueva carpeta
    ruta_salida = os.path.join(CARPETA_SALIDA, nombre_archivo)
    df_normalizado.to_csv(ruta_salida, index=False)
    print(f"   [OK] Guardado: {ruta_salida}")

print("\n🚀 ¡Proceso de normalización con RobustScaler completado con éxito!")