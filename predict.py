"""
predict.py — Inferencia sobre ventanas de ataque con el LSTM-AE entrenado.

Uso:
    python predict.py
    python predict.py --input ruta/a/mi_ventana.npy
    python predict.py --threshold 0.0042
"""
import os
os.environ["KERAS_BACKEND"] = "torch"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import argparse
import numpy as np
import keras
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

# ════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ════════════════════════════════════════════════════════════════
WIN_DIR  = Path(r"C:\Users\Daniel\ventanas")
MOD_DIR  = Path(r"C:\Users\Daniel\Desktop\Proyectos\verano delfin\LSTM-AE\proyectoAguaDelfin\modelos")
SCALER_PATH = Path(r"C:\Users\Daniel\Desktop\Proyectos\verano delfin\LSTM-AE\proyectoAguaDelfin\Notebooks_IaTesisz\scaler.pkl")
T       = 20
N_FEAT  = 37
BATCH   = 512

plt.rcParams.update({'font.size': 12, 'axes.titlesize': 14, 'figure.dpi': 150})

# ════════════════════════════════════════════════════════════════
# ARGUMENTOS CLI
# ════════════════════════════════════════════════════════════════
parser = argparse.ArgumentParser(description="Inferencia LSTM-AE para IDS IoT")
parser.add_argument("--input",     type=str, default=None,
                    help="Ruta a archivo .npy con ventanas (N, T, 37). "
                         "Si no se especifica usa atk_w20.npy de Win_DIR.")
parser.add_argument("--labels",    type=str, default=None,
                    help="Ruta a atk_labels_w20.npy (opcional para colorear por clase).")
parser.add_argument("--threshold", type=float, default=None,
                    help="Threshold manual. Si no se pasa, carga modelos/threshold.npy")
parser.add_argument("--output",    type=str, default=None,
                    help="Directorio de salida para gráficas y CSV. Default: modelos/predicciones/")
args = parser.parse_args()

OUT_DIR = Path(args.output) if args.output else MOD_DIR / "predicciones"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ════════════════════════════════════════════════════════════════
# CARGAR MODELO Y THRESHOLD
# ════════════════════════════════════════════════════════════════
print("=" * 65)
print("  PREDICT — Inferencia LSTM-AE IDS IoT")
print("=" * 65)

model_path = MOD_DIR / "lstm_ae_best.keras"
assert model_path.exists(), f"Modelo no encontrado: {model_path}\nEjecuta train.py primero."
print(f"  Cargando modelo: {model_path.name}")
model = keras.models.load_model(str(model_path))

if args.threshold is not None:
    threshold = args.threshold
    print(f"  Threshold (manual):    {threshold:.8f}")
else:
    thr_path = MOD_DIR / "threshold.npy"
    assert thr_path.exists(), f"No se encontró {thr_path}\nEjecuta evaluate.py o usa --threshold."
    threshold = float(np.load(thr_path))
    print(f"  Threshold (desde archivo): {threshold:.8f}")

# ════════════════════════════════════════════════════════════════
# CARGAR DATOS DE ENTRADA
# ════════════════════════════════════════════════════════════════
if args.input is not None:
    X = np.load(args.input)
    nombre_datos = Path(args.input).stem
else:
    X = np.load(WIN_DIR / f"atk_w{T}.npy")
    nombre_datos = f"atk_w{T}"

assert X.ndim == 3, f"Se esperan 3 dimensiones (N, T, F), recibido {X.shape}"
assert X.shape[1] == T,      f"T incorrecto: esperado {T}, recibido {X.shape[1]}"
assert X.shape[2] == N_FEAT, f"Features incorrecto: esperado {N_FEAT}, recibido {X.shape[2]}"
print(f"  Ventanas de entrada: {X.shape}")

y_true = None
if args.labels is not None:
    y_true = np.load(args.labels, allow_pickle=True)
elif (WIN_DIR / f"atk_labels_w{T}.npy").exists() and args.input is None:
    y_true = np.load(WIN_DIR / f"atk_labels_w{T}.npy", allow_pickle=True)
    if y_true is not None:
        print(f"  Etiquetas: {y_true.shape}  ({len(np.unique(y_true))} clases)")
print()

# ════════════════════════════════════════════════════════════════
# INFERENCIA
# ════════════════════════════════════════════════════════════════
print("=" * 65)
print("  INFERENCIA")
print("=" * 65)

X_hat = model.predict(X, batch_size=BATCH, verbose=1)
mse   = np.mean((X - X_hat) ** 2, axis=(1, 2))
flags = (mse > threshold).astype(int)

n_anomalias = flags.sum()
n_total     = len(flags)
pct         = n_anomalias / n_total * 100

print()
print(f"  Ventanas analizadas: {n_total:,}")
print(f"  Anomalías detectadas: {n_anomalias:,}  ({pct:.1f}%)")
print(f"  Tráfico normal:       {n_total - n_anomalias:,}  ({100-pct:.1f}%)")
print()

# ════════════════════════════════════════════════════════════════
# GUARDAR RESULTADOS CSV
# ════════════════════════════════════════════════════════════════
import pandas as pd

df_out = pd.DataFrame({
    "ventana_idx": np.arange(n_total),
    "mse":         mse,
    "anomalia":    flags,
    "threshold":   threshold,
})
if y_true is not None:
    df_out["clase_real"] = y_true
    # Métricas de detección por clase
    print("  Detección por clase:")
    print(f"  {'Clase':42}  {'Det.%':>6}  {'N':>7}")
    print(f"  {'-'*60}")
    for clase in sorted(np.unique(y_true)):
        mask = y_true == clase
        tasa = flags[mask].mean() * 100
        marca = "✓" if tasa >= 80 else ("~" if tasa >= 50 else "✗")
        print(f"  {marca} {str(clase):40}  {tasa:>5.1f}%  {mask.sum():>7,}")
    print()

csv_path = OUT_DIR / f"predicciones_{nombre_datos}.csv"
df_out.to_csv(csv_path, index=False)
print(f"  CSV guardado: {csv_path}")

# ════════════════════════════════════════════════════════════════
# GRÁFICAS
# ════════════════════════════════════════════════════════════════
print()
print("  Generando gráficas ...")
PALETA = {'benigno': '#2ecc71', 'ataque': '#e74c3c'}

# 1 — Señal de MSE a lo largo del tiempo
fig, axes = plt.subplots(2, 1, figsize=(16, 8), gridspec_kw={'height_ratios': [3, 1]})
fig.patch.set_facecolor('#f8f9fa')

ax1 = axes[0]
ax1.set_facecolor('#f8f9fa')
colores = np.where(flags, PALETA['ataque'], PALETA['benigno'])
ax1.scatter(np.arange(n_total), mse, c=colores, s=5, alpha=0.6, linewidths=0)
ax1.axhline(threshold, color='black', linewidth=1.5, linestyle='--',
            label=f'Threshold = {threshold:.5f}')
ax1.set_ylabel('MSE de reconstrucción')
ax1.set_title(f'Señal de Anomalía — {nombre_datos}\n'
              f'{n_anomalias:,} anomalías detectadas ({pct:.1f}%)',
              fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(alpha=0.3)
patch_ben = plt.scatter([], [], c=PALETA['benigno'], s=30, label='Normal')
patch_atk = plt.scatter([], [], c=PALETA['ataque'],  s=30, label='Anomalía')
ax1.legend(handles=[patch_ben, patch_atk,
                    plt.Line2D([0],[0], color='black', linestyle='--', label=f'Threshold={threshold:.5f}')],
           fontsize=10)

ax2 = axes[1]
ax2.set_facecolor('#f8f9fa')
ax2.fill_between(np.arange(n_total), flags, alpha=0.7,
                 color=PALETA['ataque'], step='mid')
ax2.set_ylabel('Alerta')
ax2.set_xlabel('Índice de ventana')
ax2.set_ylim(-0.1, 1.3)
ax2.set_yticks([0, 1])
ax2.set_yticklabels(['Normal', 'Alerta'])
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_DIR / f'anomaly_signal_{nombre_datos}.png', dpi=150, bbox_inches='tight')
plt.close()

# 2 — Histograma de MSE con threshold
fig, ax = plt.subplots(figsize=(10, 5))
ax.set_facecolor('#f8f9fa')
clip = np.percentile(mse, 99)
ax.hist(np.clip(mse[flags == 0], 0, clip), bins=60, density=True,
        color=PALETA['benigno'], alpha=0.7, label='Normal')
ax.hist(np.clip(mse[flags == 1], 0, clip), bins=60, density=True,
        color=PALETA['ataque'],  alpha=0.7, label='Anomalía')
ax.axvline(threshold, color='black', linewidth=2, linestyle='--',
           label=f'Threshold={threshold:.5f}')
ax.set_xlabel('MSE')
ax.set_ylabel('Densidad')
ax.set_title('Distribución de MSE — Datos de Predicción', fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / f'mse_histogram_{nombre_datos}.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"  Gráficas guardadas en {OUT_DIR}")
print()
print("=" * 65)
print("  RESUMEN")
print("=" * 65)
print(f"  Modelo:      {model_path.name}")
print(f"  Threshold:   {threshold:.8f}")
print(f"  Ventanas:    {n_total:,}")
print(f"  Anomalías:   {n_anomalias:,}  ({pct:.1f}%)")
print(f"  Resultados:  {csv_path}")
