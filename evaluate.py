import os
os.environ["KERAS_BACKEND"] = "torch"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pickle
import torch
import keras
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, f1_score, precision_score, recall_score
)
import seaborn as sns

# ════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ════════════════════════════════════════════════════════════════
WIN_DIR  = Path(r"C:\Users\Daniel\ventanas")
MOD_DIR  = Path(r"C:\Users\Daniel\Desktop\Proyectos\verano delfin\LSTM-AE\proyectoAguaDelfin\modelos")
GRAF_DIR = MOD_DIR / "graficas_eval"
GRAF_DIR.mkdir(parents=True, exist_ok=True)

T      = 20
BATCH  = 512
PERCENTIL_THRESHOLD = 95   # percentil sobre MSE en validación benigna

plt.rcParams.update({'font.size': 12, 'axes.titlesize': 14, 'figure.dpi': 150})
PALETA = {'benigno': '#2ecc71', 'ataque': '#e74c3c', 'fondo': '#f8f9fa'}

# ════════════════════════════════════════════════════════════════
# CARGAR MODELO Y DATOS
# ════════════════════════════════════════════════════════════════
print("=" * 65)
print("  EVALUATE — Threshold y Métricas LSTM-AE")
print("=" * 65)

model_path = MOD_DIR / "lstm_ae_best.keras"
assert model_path.exists(), f"No se encontró {model_path}\nEjecuta train.py primero."

print(f"  Cargando modelo: {model_path.name}")
model = keras.models.load_model(str(model_path))

print(f"  Cargando val_w{T}.npy  ...", end=" ")
X_val = np.load(WIN_DIR / f"val_w{T}.npy")
print(X_val.shape)

print(f"  Cargando atk_w{T}.npy  ...", end=" ")
X_atk = np.load(WIN_DIR / f"atk_w{T}.npy")
print(X_atk.shape)

print(f"  Cargando atk_labels_w{T}.npy ...", end=" ")
y_atk = np.load(WIN_DIR / f"atk_labels_w{T}.npy", allow_pickle=True)
print(y_atk.shape)
print()

# ════════════════════════════════════════════════════════════════
# CALCULAR ERROR DE RECONSTRUCCIÓN (MSE por ventana)
# ════════════════════════════════════════════════════════════════
print("=" * 65)
print("  PASO 1 — Errores de reconstrucción")
print("=" * 65)

def mse_por_ventana(model, X, batch_size=512):
    X_hat = model.predict(X, batch_size=batch_size, verbose=0)
    mse   = np.mean((X - X_hat) ** 2, axis=(1, 2))
    return mse, X_hat

print("  Calculando MSE val benigno ...")
mse_val, _ = mse_por_ventana(model, X_val, BATCH)

print("  Calculando MSE ataque ...")
mse_atk, _ = mse_por_ventana(model, X_atk, BATCH)

print(f"  Val  MSE → media={mse_val.mean():.6f}  std={mse_val.std():.6f}  "
      f"p95={np.percentile(mse_val, 95):.6f}  p99={np.percentile(mse_val, 99):.6f}")
print(f"  Atk  MSE → media={mse_atk.mean():.6f}  std={mse_atk.std():.6f}")
print()

# ════════════════════════════════════════════════════════════════
# CALCULAR THRESHOLD
# ════════════════════════════════════════════════════════════════
print("=" * 65)
print("  PASO 2 — Threshold")
print("=" * 65)

threshold = float(np.percentile(mse_val, PERCENTIL_THRESHOLD))
np.save(MOD_DIR / "threshold.npy", np.array(threshold))
print(f"  Threshold (p{PERCENTIL_THRESHOLD} sobre val benigno): {threshold:.8f}")
print(f"  Guardado en modelos/threshold.npy")
print()

# ════════════════════════════════════════════════════════════════
# CONSTRUIR DATASET BINARIO PARA MÉTRICAS
# ════════════════════════════════════════════════════════════════
mse_all = np.concatenate([mse_val, mse_atk])
y_true  = np.concatenate([
    np.zeros(len(mse_val), dtype=int),   # 0 = benigno
    np.ones( len(mse_atk), dtype=int),   # 1 = ataque
])
y_pred  = (mse_all > threshold).astype(int)

# ════════════════════════════════════════════════════════════════
# MÉTRICAS GLOBALES
# ════════════════════════════════════════════════════════════════
print("=" * 65)
print("  PASO 3 — Métricas globales (benigno vs ataque)")
print("=" * 65)

prec  = precision_score(y_true, y_pred)
rec   = recall_score(y_true, y_pred)
f1    = f1_score(y_true, y_pred)
auc   = roc_auc_score(y_true, mse_all)
fpr   = 1 - recall_score(y_true, y_pred, pos_label=0)  # FPR = FP/(FP+TN)

print(f"  Precision:   {prec:.4f}")
print(f"  Recall:      {rec:.4f}")
print(f"  F1-score:    {f1:.4f}")
print(f"  AUC-ROC:     {auc:.4f}")
print(f"  FPR:         {fpr:.4f}")
print()
print(classification_report(y_true, y_pred, target_names=["Benigno", "Ataque"]))

# ════════════════════════════════════════════════════════════════
# MÉTRICAS POR CLASE DE ATAQUE
# ════════════════════════════════════════════════════════════════
print("=" * 65)
print("  Detección por categoría de ataque")
print("=" * 65)
clases = np.unique(y_atk)
y_pred_atk = (mse_atk > threshold).astype(int)

print(f"  {'Clase de ataque':42}  {'Ventanas':>8}  {'Det.%':>6}")
print(f"  {'-'*65}")
for clase in sorted(clases):
    mask = y_atk == clase
    det  = y_pred_atk[mask].mean() * 100
    cnt  = mask.sum()
    marca = "✓" if det >= 80 else ("~" if det >= 50 else "✗")
    print(f"  {marca} {str(clase):40}  {cnt:>8,}  {det:>5.1f}%")
print()

# ════════════════════════════════════════════════════════════════
# GRÁFICAS
# ════════════════════════════════════════════════════════════════
print("  Generando gráficas ...")

# 1 — Distribución de MSE benigno vs ataque
fig, ax = plt.subplots(figsize=(12, 5))
fig.patch.set_facecolor(PALETA['fondo']); ax.set_facecolor(PALETA['fondo'])

clip = np.percentile(mse_atk, 99)
ax.hist(np.clip(mse_val, 0, clip), bins=80, density=True,
        color=PALETA['benigno'], alpha=0.65, label='Benigno (val)')
ax.hist(np.clip(mse_atk, 0, clip), bins=80, density=True,
        color=PALETA['ataque'],  alpha=0.65, label='Ataque')
ax.axvline(threshold, color='black', linewidth=2, linestyle='--',
           label=f'Threshold={threshold:.5f}')
ax.set_xlabel('MSE de reconstrucción')
ax.set_ylabel('Densidad')
ax.set_title(f'Distribución MSE — LSTM-AE  (threshold p{PERCENTIL_THRESHOLD}={threshold:.5f})',
             fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(GRAF_DIR / 'mse_distribution.png', dpi=150, bbox_inches='tight')
plt.close()

# 2 — Confusion matrix
cm = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Pred Benigno', 'Pred Ataque'],
            yticklabels=['Real Benigno', 'Real Ataque'])
ax.set_title(f'Matriz de Confusión\nF1={f1:.3f}  AUC={auc:.3f}', fontweight='bold')
plt.tight_layout()
plt.savefig(GRAF_DIR / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()

# 3 — Detection rate por clase
tasas = []
for clase in sorted(clases):
    mask = y_atk == clase
    tasas.append((str(clase), y_pred_atk[mask].mean() * 100, mask.sum()))
tasas.sort(key=lambda x: x[1])

fig, ax = plt.subplots(figsize=(14, max(6, len(tasas) * 0.4)))
colors = [PALETA['ataque'] if t[1] < 50 else
          ('#f39c12' if t[1] < 80 else PALETA['benigno']) for t in tasas]
ax.barh([t[0] for t in tasas], [t[1] for t in tasas],
        color=colors, edgecolor='white')
ax.axvline(80, color='black', linestyle='--', linewidth=1, alpha=0.6, label='80% umbral')
ax.set_xlabel('Tasa de detección (%)')
ax.set_title('Detección por Clase de Ataque — LSTM-AE', fontweight='bold')
ax.set_xlim(0, 105)
ax.legend()
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(GRAF_DIR / 'detection_per_class.png', dpi=150, bbox_inches='tight')
plt.close()

# 4 — Curva de entrenamiento
if (MOD_DIR / "history_loss.npy").exists():
    train_loss = np.load(MOD_DIR / "history_loss.npy")
    val_loss   = np.load(MOD_DIR / "history_val_loss.npy")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(train_loss, label='Train loss', color='#2980b9')
    ax.plot(val_loss,   label='Val loss',   color='#e74c3c')
    ax.axvline(val_loss.argmin(), color='gray', linestyle='--', alpha=0.7,
               label=f'Best epoch={val_loss.argmin()+1}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title('Curva de Entrenamiento — LSTM-AE', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(GRAF_DIR / 'training_curve.png', dpi=150, bbox_inches='tight')
    plt.close()

print("  Gráficas guardadas en modelos/graficas_eval/")
print()
print("=" * 65)
print("  RESUMEN FINAL")
print("=" * 65)
print(f"  Threshold:   {threshold:.8f}")
print(f"  F1-score:    {f1:.4f}")
print(f"  AUC-ROC:     {auc:.4f}")
print(f"  Precision:   {prec:.4f}")
print(f"  Recall:      {rec:.4f}")
print()
print("  Siguiente paso: ejecutar predict.py")
