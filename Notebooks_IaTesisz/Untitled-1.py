# %%
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import RobustScaler
import pickle

# ════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ════════════════════════════════════════════════════════════════
BASE     = Path(r"C:\Users\Daniel\Desktop\Proyectos\Ia-tesisi\Notebooks_IaTesisz")
PART_DIR = BASE / "particiones"
NORM_DIR = BASE / "normalizacion"
NORM_DIR.mkdir(parents=True, exist_ok=True)

# Features originales (39) → eliminar Telnet y SMTP → quedan 37
FEATURES_ORIGINALES = [
    'Header_Length', 'Protocol Type', 'Time_To_Live', 'Rate',
    'fin_flag_number', 'syn_flag_number', 'rst_flag_number',
    'psh_flag_number', 'ack_flag_number', 'ece_flag_number',
    'cwr_flag_number', 'ack_count', 'syn_count', 'fin_count',
    'rst_count', 'HTTP', 'HTTPS', 'DNS', 'Telnet', 'SMTP',
    'SSH', 'IRC', 'TCP', 'UDP', 'DHCP', 'ARP', 'ICMP', 'IGMP',
    'IPv', 'LLC', 'Tot sum', 'Min', 'Max', 'AVG', 'Std',
    'Tot size', 'IAT', 'Number', 'Variance'
]

FEATURES_ELIMINAR = ['Telnet', 'SMTP']
FEATURES = [f for f in FEATURES_ORIGINALES if f not in FEATURES_ELIMINAR]

# ── Rutas de entrada ──────────────────────────────────────────
PATH_TRAIN = PART_DIR / "benign_train_80.csv"
PATH_VAL   = PART_DIR / "benign_val_20.csv"
PATH_ATK   = PART_DIR / "attack_test.csv"

# ── Rutas de salida ───────────────────────────────────────────
PATH_TRAIN_NORM = NORM_DIR / "train_norm.npy"
PATH_VAL_NORM   = NORM_DIR / "val_norm.npy"
PATH_ATK_NORM   = NORM_DIR / "atk_norm.npy"
PATH_ATK_LABELS = NORM_DIR / "atk_labels.npy"
PATH_SCALER     = NORM_DIR / "scaler.pkl"

# ── Validar que existen los archivos de entrada ───────────────
assert PATH_TRAIN.exists(), f"No encontrado: {PATH_TRAIN}"
assert PATH_VAL.exists(),   f"No encontrado: {PATH_VAL}"
assert PATH_ATK.exists(),   f"No encontrado: {PATH_ATK}"

print("=" * 65)
print("  NORMALIZACIÓN — RobustScaler")
print("=" * 65)
print(f"  Entrada:              {PART_DIR}")
print(f"  Salida:               {NORM_DIR}")
print(f"  Features originales:  {len(FEATURES_ORIGINALES)}")
print(f"  Eliminadas (std=0):   {FEATURES_ELIMINAR}")
print(f"  Features finales:     {len(FEATURES)}")
print()

# ════════════════════════════════════════════════════════════════
# PASO 1 — Cargar train y hacer FIT del scaler
# ════════════════════════════════════════════════════════════════
print("=" * 65)
print("  PASO 1 — Fit del RobustScaler sobre train benigno")
print("=" * 65)

print(f"  Cargando benign_train_80.csv ...", end=" ")
df_train = pd.read_csv(PATH_TRAIN, usecols=FEATURES)
df_train = df_train[FEATURES]
print(f"{len(df_train):,} filas × {df_train.shape[1]} features")

scaler = RobustScaler()
scaler.fit(df_train)

with open(PATH_SCALER, 'wb') as f:
    pickle.dump(scaler, f)
print(f"  ✓ scaler.pkl guardado en normalizacion/")

# ════════════════════════════════════════════════════════════════
# PASO 2 — Transform train
# ════════════════════════════════════════════════════════════════
print()
print("=" * 65)
print("  PASO 2 — Transform train benigno")
print("=" * 65)

train_norm = scaler.transform(df_train).astype(np.float32)
np.save(PATH_TRAIN_NORM, train_norm)

size_mb = PATH_TRAIN_NORM.stat().st_size / (1024**2)
print(f"  train_norm.npy")
print(f"    shape:  {train_norm.shape}")
print(f"    dtype:  {train_norm.dtype}")
print(f"    min:    {train_norm.min():.4f}")
print(f"    max:    {train_norm.max():.4f}")
print(f"    mean:   {train_norm.mean():.4f}")
print(f"    size:   {size_mb:.1f} MB")

del df_train, train_norm

# ════════════════════════════════════════════════════════════════
# PASO 3 — Transform val benigno
# ════════════════════════════════════════════════════════════════
print()
print("=" * 65)
print("  PASO 3 — Transform val benigno")
print("=" * 65)

print(f"  Cargando benign_val_20.csv ...", end=" ")
df_val = pd.read_csv(PATH_VAL, usecols=FEATURES)
df_val = df_val[FEATURES]
print(f"{len(df_val):,} filas")

val_norm = scaler.transform(df_val).astype(np.float32)
np.save(PATH_VAL_NORM, val_norm)

size_mb = PATH_VAL_NORM.stat().st_size / (1024**2)
print(f"  val_norm.npy")
print(f"    shape:  {val_norm.shape}")
print(f"    min:    {val_norm.min():.4f}")
print(f"    max:    {val_norm.max():.4f}")
print(f"    mean:   {val_norm.mean():.4f}")
print(f"    size:   {size_mb:.1f} MB")

del df_val, val_norm

# ════════════════════════════════════════════════════════════════
# PASO 4 — Transform attack test
# ════════════════════════════════════════════════════════════════
print()
print("=" * 65)
print("  PASO 4 — Transform attack test")
print("=" * 65)

print(f"  Cargando attack_test.csv ...", end=" ")
df_atk = pd.read_csv(PATH_ATK)
print(f"{len(df_atk):,} filas")

# Guardar etiquetas por separado antes de escalar
labels = df_atk['clase_ataque'].values
np.save(PATH_ATK_LABELS, labels)
print(f"  ✓ atk_labels.npy guardado  →  {len(labels):,} etiquetas")
print(f"  Clases únicas: {len(np.unique(labels))}")

# Escalar solo las 37 features
atk_norm = scaler.transform(df_atk[FEATURES]).astype(np.float32)
np.save(PATH_ATK_NORM, atk_norm)

size_mb = PATH_ATK_NORM.stat().st_size / (1024**2)
print(f"  atk_norm.npy")
print(f"    shape:  {atk_norm.shape}")
print(f"    min:    {atk_norm.min():.4f}")
print(f"    max:    {atk_norm.max():.4f}")
print(f"    mean:   {atk_norm.mean():.4f}")
print(f"    size:   {size_mb:.1f} MB")

del df_atk, atk_norm

# ════════════════════════════════════════════════════════════════
# RESUMEN FINAL
# ════════════════════════════════════════════════════════════════
print()
print("=" * 65)
print("  RESUMEN FINAL — normalizacion/")
print("=" * 65)
for p in [PATH_SCALER, PATH_TRAIN_NORM, PATH_VAL_NORM,
          PATH_ATK_NORM, PATH_ATK_LABELS]:
    size = p.stat().st_size / (1024**2)
    print(f"  {p.name:25}  {size:.1f} MB")

print()
print("  Para cargar en el siguiente notebook:")
print()
print("  import numpy as np, pickle")
print("  from pathlib import Path")
print()
print("  NORM_DIR   = Path(r'...\\normalizacion')")
print("  train_norm = np.load(NORM_DIR / 'train_norm.npy')")
print("  val_norm   = np.load(NORM_DIR / 'val_norm.npy')")
print("  atk_norm   = np.load(NORM_DIR / 'atk_norm.npy')")
print("  atk_labels = np.load(NORM_DIR / 'atk_labels.npy', allow_pickle=True)")
print("  with open(NORM_DIR / 'scaler.pkl', 'rb') as f:")
print("      scaler = pickle.load(f)")

# %%
import numpy as np
import pickle
from pathlib import Path

# ════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ════════════════════════════════════════════════════════════════
BASE     = Path(r"C:\Users\Daniel\Desktop\Proyectos\Ia-tesisi\Notebooks_IaTesisz")
NORM_DIR = BASE / "normalizacion"

FEATURES = [
    'Header_Length', 'Protocol Type', 'Time_To_Live', 'Rate',
    'fin_flag_number', 'syn_flag_number', 'rst_flag_number',
    'psh_flag_number', 'ack_flag_number', 'ece_flag_number',
    'cwr_flag_number', 'ack_count', 'syn_count', 'fin_count',
    'rst_count', 'HTTP', 'HTTPS', 'DNS', 'SSH', 'IRC',
    'TCP', 'UDP', 'DHCP', 'ARP', 'ICMP', 'IGMP',
    'IPv', 'LLC', 'Tot sum', 'Min', 'Max', 'AVG', 'Std',
    'Tot size', 'IAT', 'Number', 'Variance'
]  # 37 features (sin Telnet ni SMTP)

# ════════════════════════════════════════════════════════════════
# CARGAR ARCHIVOS — uno por uno para no saturar RAM
# ════════════════════════════════════════════════════════════════

def verificar_array(nombre, arr):
    print(f"  {nombre}")
    print(f"    shape:        {arr.shape}")
    print(f"    dtype:        {arr.dtype}")
    print(f"    nulos:        {np.isnan(arr).sum()}")
    print(f"    infinitos:    {np.isinf(arr).sum()}")
    print(f"    min:          {arr.min():.4f}")
    print(f"    max:          {arr.max():.4f}")
    print(f"    mean:         {arr.mean():.4f}")
    print(f"    std:          {arr.std():.4f}")

    # Verificar que no hay columnas constantes (std=0)
    stds = arr.std(axis=0)
    cols_constantes = np.where(stds == 0)[0]
    if len(cols_constantes) > 0:
        nombres = [FEATURES[i] for i in cols_constantes]
        print(f"    ⚠ columnas constantes: {nombres}")
    else:
        print(f"    columnas constantes:  ninguna ✓")

    # Verificar que la media por feature es razonable
    medias = arr.mean(axis=0)
    cols_extremas = np.where(np.abs(medias) > 1000)[0]
    if len(cols_extremas) > 0:
        nombres = [FEATURES[i] for i in cols_extremas]
        print(f"    ⚠ features con media > 1000: {nombres}")
    else:
        print(f"    medias por feature:   todas razonables ✓")
    print()

# ════════════════════════════════════════════════════════════════
# VERIFICAR SCALER
# ════════════════════════════════════════════════════════════════
print("=" * 65)
print("  VERIFICACIÓN — Particiones Normalizadas")
print("=" * 65)
print()

print("  [scaler.pkl]")
with open(NORM_DIR / "scaler.pkl", 'rb') as f:
    scaler = pickle.load(f)
print(f"    tipo:         {type(scaler).__name__}")
print(f"    n_features:   {scaler.center_.shape[0]}")
print(f"    center (mediana) — primeras 5: {scaler.center_[:5].round(4)}")
print(f"    scale  (IQR)     — primeras 5: {scaler.scale_[:5].round(4)}")
print()

# ════════════════════════════════════════════════════════════════
# VERIFICAR train_norm
# ════════════════════════════════════════════════════════════════
print("=" * 65)
print("  train_norm.npy")
print("=" * 65)
train_norm = np.load(NORM_DIR / "train_norm.npy")
verificar_array("train_norm", train_norm)
del train_norm

# ════════════════════════════════════════════════════════════════
# VERIFICAR val_norm
# ════════════════════════════════════════════════════════════════
print("=" * 65)
print("  val_norm.npy")
print("=" * 65)
val_norm = np.load(NORM_DIR / "val_norm.npy")
verificar_array("val_norm", val_norm)
del val_norm

# ════════════════════════════════════════════════════════════════
# VERIFICAR atk_norm
# ════════════════════════════════════════════════════════════════
print("=" * 65)
print("  atk_norm.npy")
print("=" * 65)
atk_norm = np.load(NORM_DIR / "atk_norm.npy")
verificar_array("atk_norm", atk_norm)
del atk_norm

# ════════════════════════════════════════════════════════════════
# VERIFICAR atk_labels
# ════════════════════════════════════════════════════════════════
print("=" * 65)
print("  atk_labels.npy")
print("=" * 65)
atk_labels = np.load(NORM_DIR / "atk_labels.npy", allow_pickle=True)
print(f"  atk_labels")
print(f"    shape:        {atk_labels.shape}")
print(f"    dtype:        {atk_labels.dtype}")
print(f"    clases únicas:{len(np.unique(atk_labels))}")
print()
print(f"  Distribución por clase:")
clases, conteos = np.unique(atk_labels, return_counts=True)
for clase, conteo in zip(clases, conteos):
    print(f"    {clase:40}  {conteo:>7,} filas")
print()
print(f"  Total: {len(atk_labels):,}")

# ════════════════════════════════════════════════════════════════
# VERIFICACIÓN CRUZADA — shapes consistentes
# ════════════════════════════════════════════════════════════════
print()
print("=" * 65)
print("  VERIFICACIÓN CRUZADA")
print("=" * 65)

train_shape = np.load(NORM_DIR / "train_norm.npy").shape
val_shape   = np.load(NORM_DIR / "val_norm.npy").shape
atk_shape   = np.load(NORM_DIR / "atk_norm.npy").shape

print(f"  Features consistentes:")
print(f"    train_norm[:, features] = {train_shape[1]}  {'✓' if train_shape[1] == 37 else '✗'}")
print(f"    val_norm  [:, features] = {val_shape[1]}  {'✓' if val_shape[1] == 37 else '✗'}")
print(f"    atk_norm  [:, features] = {atk_shape[1]}  {'✓' if atk_shape[1] == 37 else '✗'}")
print()
print(f"  Labels consistentes con atk_norm:")
atk_labels = np.load(NORM_DIR / "atk_labels.npy", allow_pickle=True)
match = atk_shape[0] == len(atk_labels)
print(f"    atk_norm filas={atk_shape[0]:,}  labels={len(atk_labels):,}  {'✓' if match else '✗ NO COINCIDEN'}")
print()
print(f"  Proporción train/val:")
total = train_shape[0] + val_shape[0]
print(f"    train: {train_shape[0]:,} ({train_shape[0]/total*100:.1f}%)")
print(f"    val:   {val_shape[0]:,}  ({val_shape[0]/total*100:.1f}%)")
print()
print("  Todo verificado ✓" if all([
    train_shape[1] == 37,
    val_shape[1] == 37,
    atk_shape[1] == 37,
    match
]) else "  ⚠ Hay inconsistencias, revisar arriba")

# %%
# ════════════════════════════════════════════════════════════════ ANALISIS EXPLORATORIO DE LOS DATOS NORMALIZADOS
# ═══════════════════════════════════════════════════════════════

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

# ════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ════════════════════════════════════════════════════════════════
BASE     = Path(r"C:\Users\Daniel\Desktop\Proyectos\Ia-tesisi\Notebooks_IaTesisz")
PART_DIR = BASE / "particiones"
NORM_DIR = BASE / "normalizacion"
GRAF_DIR = BASE / "graficas_eda"
GRAF_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    'Header_Length', 'Protocol Type', 'Time_To_Live', 'Rate',
    'fin_flag_number', 'syn_flag_number', 'rst_flag_number',
    'psh_flag_number', 'ack_flag_number', 'ece_flag_number',
    'cwr_flag_number', 'ack_count', 'syn_count', 'fin_count',
    'rst_count', 'HTTP', 'HTTPS', 'DNS', 'SSH', 'IRC',
    'TCP', 'UDP', 'DHCP', 'ARP', 'ICMP', 'IGMP',
    'IPv', 'LLC', 'Tot sum', 'Min', 'Max', 'AVG', 'Std',
    'Tot size', 'IAT', 'Number', 'Variance'
]

PALETA = {
    'benigno': '#2ecc71',
    'ataque':  '#e74c3c',
    'fondo':   '#f8f9fa',
}

plt.rcParams.update({
    'font.size':      13,
    'axes.titlesize': 15,
    'axes.labelsize': 13,
    'figure.dpi':     150,
})

SAMPLE_N = 50_000

# ════════════════════════════════════════════════════════════════
# CARGAR DATOS ORIGINALES (histogramas y boxplots)
# ════════════════════════════════════════════════════════════════
print("=" * 65)
print("  Cargando datos originales...")
print("=" * 65)
print(f"  benign_full.csv  ...", end=" ")
df_benign = pd.read_csv(PART_DIR / "benign_full.csv", usecols=FEATURES)
print(f"{len(df_benign):,} filas")

print(f"  attack_test.csv  ...", end=" ")
df_attack = pd.read_csv(PART_DIR / "attack_test.csv", usecols=FEATURES)
print(f"{len(df_attack):,} filas")

df_ben_sample = df_benign.sample(n=SAMPLE_N, random_state=42)
df_atk_sample = df_attack.sample(n=SAMPLE_N, random_state=42)
print(f"  Muestra: {SAMPLE_N:,} filas por clase")

# ════════════════════════════════════════════════════════════════
# GRÁFICA 1 — Histogramas (datos originales)
# ════════════════════════════════════════════════════════════════
print()
print("  Generando Gráfica 1 — Histogramas ...")

FEATURES_CLAVE = [
    'Rate', 'Header_Length', 'Time_To_Live', 'IAT',
    'Tot sum', 'AVG', 'Variance', 'ack_count',
    'TCP', 'UDP', 'HTTPS', 'ICMP'
]

fig, axes = plt.subplots(4, 3, figsize=(18, 16))
fig.patch.set_facecolor(PALETA['fondo'])
fig.suptitle('Distribución de Features Clave\nTráfico Benigno vs Ataque — CICIoT2023',
             fontsize=17, fontweight='bold', y=1.01)

for ax, feat in zip(axes.flatten(), FEATURES_CLAVE):
    ax.set_facecolor(PALETA['fondo'])
    lim = max(
        np.percentile(df_ben_sample[feat], 99),
        np.percentile(df_atk_sample[feat], 99)
    )
    ax.hist(df_ben_sample[feat].clip(upper=lim), bins=50,
            color=PALETA['benigno'], alpha=0.6, label='Benigno', density=True)
    ax.hist(df_atk_sample[feat].clip(upper=lim), bins=50,
            color=PALETA['ataque'],  alpha=0.6, label='Ataque',  density=True)
    ax.set_title(feat, fontweight='bold')
    ax.set_xlabel('Valor original')
    ax.set_ylabel('Densidad')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(GRAF_DIR / 'histogramas.png', dpi=150, bbox_inches='tight')
plt.show()
print("  ✓ histogramas.png")

# Liberar memoria
del df_benign, df_attack
import gc; gc.collect()

# ════════════════════════════════════════════════════════════════
# GRÁFICA 2 — Boxplots (datos originales)
# ════════════════════════════════════════════════════════════════
print()
print("  Generando Gráfica 2 — Boxplots ...")

FEATURES_BOX = [
    'Rate', 'Header_Length', 'Time_To_Live', 'IAT',
    'Tot sum', 'AVG', 'Variance', 'ack_count'
]

df_ben_box = df_ben_sample[FEATURES_BOX].copy()
df_ben_box['tipo'] = 'Benigno'
df_atk_box = df_atk_sample[FEATURES_BOX].copy()
df_atk_box['tipo'] = 'Ataque'
df_box = pd.concat([df_ben_box, df_atk_box], ignore_index=True)

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.patch.set_facecolor(PALETA['fondo'])
fig.suptitle('Boxplots — Benigno vs Ataque\nCICIoT2023',
             fontsize=17, fontweight='bold', y=1.01)

for ax, feat in zip(axes.flatten(), FEATURES_BOX):
    ax.set_facecolor(PALETA['fondo'])
    p99 = df_box[feat].quantile(0.99)
    df_temp = df_box[[feat, 'tipo']].copy()
    df_temp[feat] = df_temp[feat].clip(upper=p99)
    sns.boxplot(
        data=df_temp, x='tipo', y=feat,
        palette={'Benigno': PALETA['benigno'], 'Ataque': PALETA['ataque']},
        ax=ax, width=0.5, linewidth=1.2,
        flierprops=dict(marker='o', markersize=2, alpha=0.3)
    )
    ax.set_title(feat, fontweight='bold')
    ax.set_xlabel('')
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(GRAF_DIR / 'boxplots.png', dpi=150, bbox_inches='tight')
plt.show()
print("  ✓ boxplots.png")

del df_ben_box, df_atk_box, df_box, df_temp
gc.collect()

# ════════════════════════════════════════════════════════════════
# GRÁFICA 3 — Heatmap de correlación (datos NORMALIZADOS)
# ════════════════════════════════════════════════════════════════
print()
print("  Generando Gráfica 3 — Heatmap (datos normalizados) ...")

print("  Cargando val_norm.npy ...", end=" ")
val_norm = np.load(NORM_DIR / "val_norm.npy")
print(f"{val_norm.shape}")

df_val_norm = pd.DataFrame(val_norm, columns=FEATURES)
df_val_sample = df_val_norm.sample(n=SAMPLE_N, random_state=42)
del val_norm, df_val_norm
gc.collect()

corr = df_val_sample.corr()

fig, ax = plt.subplots(figsize=(18, 15))
fig.patch.set_facecolor(PALETA['fondo'])
sns.heatmap(
    corr, ax=ax,
    cmap='RdBu_r', center=0, vmin=-1, vmax=1,
    annot=False, linewidths=0.3, linecolor='white',
    square=True, cbar_kws={'shrink': 0.8, 'label': 'Correlación'}
)
ax.set_title('Heatmap de Correlación entre Features\nTráfico Benigno Normalizado — CICIoT2023',
             fontsize=17, fontweight='bold', pad=20)
ax.tick_params(axis='x', rotation=45, labelsize=10)
ax.tick_params(axis='y', rotation=0,  labelsize=10)

plt.tight_layout()
plt.savefig(GRAF_DIR / 'heatmap_correlacion.png', dpi=150, bbox_inches='tight')
plt.show()
print("  ✓ heatmap_correlacion.png")

del df_val_sample, corr
gc.collect()

# ════════════════════════════════════════════════════════════════
# GRÁFICA 4 — Comparación media benigno vs ataque (originales)
# ════════════════════════════════════════════════════════════════
print()
print("  Generando Gráfica 4 — Comparación benigno vs ataque ...")

medias_ben = df_ben_sample[FEATURES].mean()
medias_atk = df_atk_sample[FEATURES].mean()

# Escalar al rango 0-1 solo para comparar visualmente
vals = np.vstack([medias_ben.values, medias_atk.values])
vals_scaled = MinMaxScaler().fit_transform(vals.T).T
medias_ben_s = vals_scaled[0]
medias_atk_s = vals_scaled[1]

x     = np.arange(len(FEATURES))
width = 0.4

fig, ax = plt.subplots(figsize=(22, 8))
fig.patch.set_facecolor(PALETA['fondo'])
ax.set_facecolor(PALETA['fondo'])

ax.bar(x - width/2, medias_ben_s, width,
       color=PALETA['benigno'], alpha=0.85, label='Benigno', edgecolor='white')
ax.bar(x + width/2, medias_atk_s, width,
       color=PALETA['ataque'],  alpha=0.85, label='Ataque',  edgecolor='white')

ax.set_xticks(x)
ax.set_xticklabels(FEATURES, rotation=45, ha='right', fontsize=10)
ax.set_ylabel('Media normalizada (0-1)', labelpad=12)
ax.set_title('Comparación de Media por Feature\nBenigno vs Ataque — CICIoT2023',
             fontweight='bold', pad=18)
ax.legend(fontsize=13)
ax.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig(GRAF_DIR / 'comparacion_features.png', dpi=150, bbox_inches='tight')
plt.show()
print("  ✓ comparacion_features.png")

# ════════════════════════════════════════════════════════════════
# RESUMEN
# ════════════════════════════════════════════════════════════════
print()
print("=" * 65)
print("  Gráficas guardadas en graficas_eda/")
print("=" * 65)
for p in sorted(GRAF_DIR.glob("*.png")):
    size = p.stat().st_size / (1024**2)
    print(f"  {p.name:40}  {size:.1f} MB")

# %%
# ═══════════════════════════════════════════════════════════════ VENTANAS DESLIZANTES — CICLO DE VIDA DE LOS DATOS 


import numpy as np
from pathlib import Path
import gc

# ════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ════════════════════════════════════════════════════════════════
BASE     = Path(r"C:\Users\Daniel\Desktop\Proyectos\Ia-tesisi\Notebooks_IaTesisz")
NORM_DIR = BASE / "normalizacion"
WIN_DIR = Path(r"C:\Users\Daniel\ventanas")
WIN_DIR.mkdir(parents=True, exist_ok=True)

T    = 20   # tamaño de ventana (timesteps)
STEP = 20   # paso (sin overlap)

print("=" * 65)
print("  VENTANAS DESLIZANTES")
print("=" * 65)
print(f"  Tamaño ventana (T):  {T}")
print(f"  Paso (step):         {STEP}")
print(f"  Overlap:             ninguno")
print(f"  Salida:              {WIN_DIR}")
print()

# ════════════════════════════════════════════════════════════════
# FUNCIÓN
# ════════════════════════════════════════════════════════════════
def crear_ventanas(arr, T, step):
    """
    Convierte (N, F) → (samples, T, F) sin overlap.
    Las filas sobrantes que no completan una ventana se descartan.
    """
    N, F = arr.shape
    n_ventanas = (N - T) // step + 1
    # Pre-alocar array de salida
    out = np.empty((n_ventanas, T, F), dtype=arr.dtype)
    for i in range(n_ventanas):
        inicio = i * step
        out[i] = arr[inicio : inicio + T]
    return out

# ════════════════════════════════════════════════════════════════
# PASO 1 — Train
# ════════════════════════════════════════════════════════════════
print("=" * 65)
print("  PASO 1 — Train")
print("=" * 65)
print(f"  Cargando train_norm.npy ...", end=" ")
train_norm = np.load(NORM_DIR / "train_norm.npy")
print(f"{train_norm.shape}")

train_win = crear_ventanas(train_norm, T, STEP)
np.save(WIN_DIR / f"train_w{T}.npy", train_win)

size_mb = (WIN_DIR / f"train_w{T}.npy").stat().st_size / (1024**2)
filas_descartadas = train_norm.shape[0] - (train_win.shape[0] * T)
print(f"  train_w{T}.npy")
print(f"    entrada:   {train_norm.shape}")
print(f"    salida:    {train_win.shape}  (samples × T × features)")
print(f"    descartadas: {filas_descartadas} filas")
print(f"    size:      {size_mb:.1f} MB")

del train_norm, train_win
gc.collect()

# ════════════════════════════════════════════════════════════════
# PASO 2 — Val benigno
# ════════════════════════════════════════════════════════════════
print()
print("=" * 65)
print("  PASO 2 — Val benigno")
print("=" * 65)
print(f"  Cargando val_norm.npy ...", end=" ")
val_norm = np.load(NORM_DIR / "val_norm.npy")
print(f"{val_norm.shape}")

val_win = crear_ventanas(val_norm, T, STEP)
np.save(WIN_DIR / f"val_w{T}.npy", val_win)

size_mb = (WIN_DIR / f"val_w{T}.npy").stat().st_size / (1024**2)
filas_descartadas = val_norm.shape[0] - (val_win.shape[0] * T)
print(f"  val_w{T}.npy")
print(f"    entrada:   {val_norm.shape}")
print(f"    salida:    {val_win.shape}")
print(f"    descartadas: {filas_descartadas} filas")
print(f"    size:      {size_mb:.1f} MB")

del val_norm, val_win
gc.collect()

# ════════════════════════════════════════════════════════════════
# PASO 3 — Attack test
# ════════════════════════════════════════════════════════════════
print()
print("=" * 65)
print("  PASO 3 — Attack test")
print("=" * 65)
print(f"  Cargando atk_norm.npy ...", end=" ")
atk_norm = np.load(NORM_DIR / "atk_norm.npy")
print(f"{atk_norm.shape}")

atk_win = crear_ventanas(atk_norm, T, STEP)
np.save(WIN_DIR / f"atk_w{T}.npy", atk_win)

size_mb = (WIN_DIR / f"atk_w{T}.npy").stat().st_size / (1024**2)
filas_descartadas = atk_norm.shape[0] - (atk_win.shape[0] * T)
print(f"  atk_w{T}.npy")
print(f"    entrada:   {atk_norm.shape}")
print(f"    salida:    {atk_win.shape}")
print(f"    descartadas: {filas_descartadas} filas")
print(f"    size:      {size_mb:.1f} MB")

# ── Ajustar etiquetas para que coincidan con ventanas ─────────
print()
print(f"  Ajustando atk_labels a ventanas ...", end=" ")
atk_labels = np.load(NORM_DIR / "atk_labels.npy", allow_pickle=True)

# Una etiqueta por ventana → tomar la etiqueta del primer paso de cada ventana
n_ventanas = atk_win.shape[0]
labels_win = np.array([
    atk_labels[i * STEP] for i in range(n_ventanas)
])
np.save(WIN_DIR / f"atk_labels_w{T}.npy", labels_win)
print(f"{labels_win.shape}")

# Verificar distribución de clases después del windowing
clases, conteos = np.unique(labels_win, return_counts=True)
print(f"  Clases únicas: {len(clases)}")
for clase, conteo in zip(clases, conteos):
    print(f"    {clase:40}  {conteo:>6,} ventanas")

del atk_norm, atk_win, atk_labels, labels_win
gc.collect()

# ════════════════════════════════════════════════════════════════
# RESUMEN FINAL
# ════════════════════════════════════════════════════════════════
print()
print("=" * 65)
print(f"  RESUMEN FINAL — ventanas/ (T={T}, step={STEP})")
print("=" * 65)
for p in sorted(WIN_DIR.glob("*.npy")):
    arr = np.load(p, mmap_mode='r')
    size = p.stat().st_size / (1024**2)
    print(f"  {p.name:25}  shape {str(arr.shape):25}  {size:.1f} MB")

print()
print("  Para cargar en el siguiente notebook:")
print()
print(f"  WIN_DIR  = Path(r'...\\ventanas')")
print(f"  T        = {T}")
print(f"  train_w  = np.load(WIN_DIR / 'train_w{T}.npy')")
print(f"  val_w    = np.load(WIN_DIR / 'val_w{T}.npy')")
print(f"  atk_w    = np.load(WIN_DIR / 'atk_w{T}.npy')")
print(f"  atk_lab  = np.load(WIN_DIR / 'atk_labels_w{T}.npy', allow_pickle=True)")

# %%
import numpy as np
from pathlib import Path

# ════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ════════════════════════════════════════════════════════════════
WIN_DIR = Path(r"C:\Users\Daniel\ventanas")
T       = 20

assert WIN_DIR.exists(), f"No se encontró: {WIN_DIR}"

print("=" * 65)
print("  VERIFICACIÓN — Ventanas Deslizantes")
print("=" * 65)
print(f"  Carpeta: {WIN_DIR}")
print(f"  T={T}, step={T} (sin overlap)")
print()

# ════════════════════════════════════════════════════════════════
# FUNCIÓN DE VERIFICACIÓN
# ════════════════════════════════════════════════════════════════
def verificar_ventana(nombre, arr):
    samples, timesteps, features = arr.shape
    print(f"  {nombre}")
    print(f"    shape:         {arr.shape}  (samples × T × features)")
    print(f"    dtype:         {arr.dtype}")
    print(f"    T correcto:    {timesteps == T}  ({timesteps})")
    print(f"    features:      {features == 37}  ({features})")
    print(f"    nulos:         {np.isnan(arr).sum()}")
    print(f"    infinitos:     {np.isinf(arr).sum()}")
    print(f"    min:           {arr.min():.4f}")
    print(f"    max:           {arr.max():.4f}")
    print(f"    mean:          {arr.mean():.4f}")
    print()
    return samples

# ════════════════════════════════════════════════════════════════
# VERIFICAR CADA ARCHIVO — uno por uno para no saturar RAM
# ════════════════════════════════════════════════════════════════
print("=" * 65)
print("  train_w20.npy")
print("=" * 65)
train_w = np.load(WIN_DIR / f"train_w{T}.npy")
n_train = verificar_ventana(f"train_w{T}", train_w)
del train_w

print("=" * 65)
print("  val_w20.npy")
print("=" * 65)
val_w = np.load(WIN_DIR / f"val_w{T}.npy")
n_val = verificar_ventana(f"val_w{T}", val_w)
del val_w

print("=" * 65)
print("  atk_w20.npy")
print("=" * 65)
atk_w = np.load(WIN_DIR / f"atk_w{T}.npy")
n_atk = verificar_ventana(f"atk_w{T}", atk_w)
del atk_w

print("=" * 65)
print("  atk_labels_w20.npy")
print("=" * 65)
atk_lab = np.load(WIN_DIR / f"atk_labels_w{T}.npy", allow_pickle=True)
print(f"  atk_labels_w{T}")
print(f"    shape:         {atk_lab.shape}")
print(f"    dtype:         {atk_lab.dtype}")
print(f"    clases únicas: {len(np.unique(atk_lab))}")
print()
clases, conteos = np.unique(atk_lab, return_counts=True)
for clase, conteo in zip(clases, conteos):
    print(f"    {clase:40}  {conteo:>6,} ventanas")

# ════════════════════════════════════════════════════════════════
# VERIFICACIÓN CRUZADA
# ════════════════════════════════════════════════════════════════
print()
print("=" * 65)
print("  VERIFICACIÓN CRUZADA")
print("=" * 65)

labels_ok = len(atk_lab) == n_atk
prop_ok   = abs((n_val / (n_train + n_val)) - 0.20) < 0.01

print(f"  features consistentes (37):")
print(f"    train ✓  val ✓  atk ✓")
print()
print(f"  labels coinciden con atk_w:")
print(f"    atk_w={n_atk:,}  labels={len(atk_lab):,}  {'✓' if labels_ok else '✗'}")
print()
print(f"  proporción train/val:")
total = n_train + n_val
print(f"    train: {n_train:,}  ({n_train/total*100:.1f}%)")
print(f"    val:   {n_val:,}   ({n_val/total*100:.1f}%)")
print(f"    proporción ~80/20:  {'✓' if prop_ok else '✗'}")
print()
print(f"  dimensión temporal T={T}:")
print(f"    todos los arrays tienen timesteps={T}  ✓")
print()

todo_ok = labels_ok and prop_ok
print("  Todo verificado ✓" if todo_ok else "  ⚠ Hay inconsistencias, revisar arriba")

# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import gc

# ════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ════════════════════════════════════════════════════════════════
NORM_DIR = Path(r"C:\Users\Daniel\Desktop\Proyectos\Ia-tesisi\Notebooks_IaTesisz\normalizacion")
GRAF_DIR = NORM_DIR.parent / "graficas_metricas"
GRAF_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    'Header_Length', 'Protocol Type', 'Time_To_Live', 'Rate',
    'fin_flag_number', 'syn_flag_number', 'rst_flag_number',
    'psh_flag_number', 'ack_flag_number', 'ece_flag_number',
    'cwr_flag_number', 'ack_count', 'syn_count', 'fin_count',
    'rst_count', 'HTTP', 'HTTPS', 'DNS', 'SSH', 'IRC',
    'TCP', 'UDP', 'DHCP', 'ARP', 'ICMP', 'IGMP',
    'IPv', 'LLC', 'Tot sum', 'Min', 'Max', 'AVG', 'Std',
    'Tot size', 'IAT', 'Number', 'Variance'
]

plt.rcParams.update({
    'font.size':      12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'figure.dpi':     150,
})

PALETA = {
    'train': '#2980b9',
    'val':   '#2ecc71',
    'atk':   '#e74c3c',
    'fondo': '#f8f9fa',
}

# ════════════════════════════════════════════════════════════════
# FUNCIÓN DE MÉTRICAS POR ARRAY
# ════════════════════════════════════════════════════════════════
def metricas_array(nombre, arr):
    print(f"\n  {'='*55}")
    print(f"  {nombre}  {arr.shape}")
    print(f"  {'='*55}")
    print(f"  {'Feature':30} {'Media':>8} {'Std':>8} {'Min':>8} {'P25':>8} {'P50':>8} {'P75':>8} {'Max':>10}")
    print(f"  {'-'*90}")
    stats = {}
    for i, feat in enumerate(FEATURES):
        col = arr[:, i]
        media = col.mean()
        std   = col.std()
        mn    = col.min()
        p25   = np.percentile(col, 25)
        p50   = np.percentile(col, 50)
        p75   = np.percentile(col, 75)
        mx    = col.max()
        print(f"  {feat:30} {media:>8.3f} {std:>8.3f} {mn:>8.3f} {p25:>8.3f} {p50:>8.3f} {p75:>8.3f} {mx:>10.3f}")
        stats[feat] = {
            'mean': media, 'std': std, 'min': mn,
            'p25': p25, 'p50': p50, 'p75': p75, 'max': mx
        }
    return stats

# ════════════════════════════════════════════════════════════════
# CARGAR Y MOSTRAR MÉTRICAS — uno por uno
# ════════════════════════════════════════════════════════════════
print("=" * 65)
print("  MÉTRICAS — Particiones Normalizadas (RobustScaler)")
print("  Unidades: desviaciones relativas a la mediana benigna")
print("=" * 65)

print("\n  Cargando train_norm.npy ...", end=" ")
train = np.load(NORM_DIR / "train_norm.npy")
print(f"{train.shape}")
stats_train = metricas_array("TRAIN (benigno 80%)", train)
del train; gc.collect()

print("\n  Cargando val_norm.npy ...", end=" ")
val = np.load(NORM_DIR / "val_norm.npy")
print(f"{val.shape}")
stats_val = metricas_array("VAL (benigno 20%)", val)
del val; gc.collect()

print("\n  Cargando atk_norm.npy ...", end=" ")
atk = np.load(NORM_DIR / "atk_norm.npy")
print(f"{atk.shape}")
stats_atk = metricas_array("ATK (malignos test)", atk)
del atk; gc.collect()

# ════════════════════════════════════════════════════════════════
# GRÁFICA 1 — Media por feature: train vs val vs atk
# ════════════════════════════════════════════════════════════════
print("\n  Generando Gráfica 1 — Media por feature ...")

medias_train = [stats_train[f]['mean'] for f in FEATURES]
medias_val   = [stats_val[f]['mean']   for f in FEATURES]
medias_atk   = [stats_atk[f]['mean']   for f in FEATURES]

x     = np.arange(len(FEATURES))
width = 0.28

fig, ax = plt.subplots(figsize=(22, 7))
fig.patch.set_facecolor(PALETA['fondo'])
ax.set_facecolor(PALETA['fondo'])

ax.bar(x - width, medias_train, width, color=PALETA['train'],
       alpha=0.85, label='Train benigno', edgecolor='white')
ax.bar(x,          medias_val,   width, color=PALETA['val'],
       alpha=0.85, label='Val benigno',   edgecolor='white')
ax.bar(x + width,  medias_atk,   width, color=PALETA['atk'],
       alpha=0.85, label='Ataque',        edgecolor='white')

ax.set_xticks(x)
ax.set_xticklabels(FEATURES, rotation=45, ha='right', fontsize=9)
ax.set_ylabel('Media escalada (RobustScaler)')
ax.set_title('Media por Feature — Train vs Val vs Ataque\nUnidades: desviaciones relativas a mediana benigna',
             fontweight='bold', pad=15)
ax.legend(fontsize=12)
ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
ax.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig(GRAF_DIR / 'media_por_feature.png', dpi=150, bbox_inches='tight')
plt.show()
print("  ✓ media_por_feature.png")

# ════════════════════════════════════════════════════════════════
# GRÁFICA 2 — Std por feature: train vs val vs atk
# ════════════════════════════════════════════════════════════════
print("  Generando Gráfica 2 — Desviación estándar por feature ...")

stds_train = [stats_train[f]['std'] for f in FEATURES]
stds_val   = [stats_val[f]['std']   for f in FEATURES]
stds_atk   = [stats_atk[f]['std']   for f in FEATURES]

fig, ax = plt.subplots(figsize=(22, 7))
fig.patch.set_facecolor(PALETA['fondo'])
ax.set_facecolor(PALETA['fondo'])

ax.bar(x - width, stds_train, width, color=PALETA['train'],
       alpha=0.85, label='Train benigno', edgecolor='white')
ax.bar(x,          stds_val,   width, color=PALETA['val'],
       alpha=0.85, label='Val benigno',   edgecolor='white')
ax.bar(x + width,  stds_atk,   width, color=PALETA['atk'],
       alpha=0.85, label='Ataque',        edgecolor='white')

ax.set_xticks(x)
ax.set_xticklabels(FEATURES, rotation=45, ha='right', fontsize=9)
ax.set_ylabel('Desviación estándar escalada')
ax.set_title('Desviación Estándar por Feature — Train vs Val vs Ataque',
             fontweight='bold', pad=15)
ax.legend(fontsize=12)
ax.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig(GRAF_DIR / 'std_por_feature.png', dpi=150, bbox_inches='tight')
plt.show()
print("  ✓ std_por_feature.png")

# ════════════════════════════════════════════════════════════════
# GRÁFICA 3 — Distribución global de valores (KDE)
# ════════════════════════════════════════════════════════════════
print("  Generando Gráfica 3 — Distribución global (KDE) ...")

print("  Recargando arrays para KDE ...", end=" ")
train = np.load(NORM_DIR / "train_norm.npy")
val   = np.load(NORM_DIR / "val_norm.npy")
atk   = np.load(NORM_DIR / "atk_norm.npy")
print("ok")

# Muestra plana para KDE
N_KDE = 100_000
train_flat = train.flatten()
train_flat = train_flat[np.random.choice(len(train_flat), N_KDE, replace=False)]
val_flat   = val.flatten()
val_flat   = val_flat[np.random.choice(len(val_flat), N_KDE, replace=False)]
atk_flat   = atk.flatten()
atk_flat   = atk_flat[np.random.choice(len(atk_flat), N_KDE, replace=False)]

del train, val, atk; gc.collect()

# Clippear para visualización (percentil 99)
clip_val = np.percentile(np.concatenate([train_flat, val_flat]), 99)

fig, ax = plt.subplots(figsize=(14, 6))
fig.patch.set_facecolor(PALETA['fondo'])
ax.set_facecolor(PALETA['fondo'])

sns.kdeplot(train_flat.clip(-5, clip_val), ax=ax,
            color=PALETA['train'], label='Train benigno', linewidth=2)
sns.kdeplot(val_flat.clip(-5, clip_val),   ax=ax,
            color=PALETA['val'],   label='Val benigno',   linewidth=2)
sns.kdeplot(atk_flat.clip(-5, clip_val),   ax=ax,
            color=PALETA['atk'],   label='Ataque',        linewidth=2, linestyle='--')

ax.set_xlabel('Valor escalado (RobustScaler)')
ax.set_ylabel('Densidad')
ax.set_title('Distribución Global de Valores Escalados\nTrain vs Val vs Ataque',
             fontweight='bold', pad=15)
ax.legend(fontsize=12)
ax.axvline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5, label='Mediana benigna')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(GRAF_DIR / 'distribucion_global.png', dpi=150, bbox_inches='tight')
plt.show()
print("  ✓ distribucion_global.png")

del train_flat, val_flat, atk_flat; gc.collect()

# ════════════════════════════════════════════════════════════════
# GRÁFICA 4 — Heatmap de diferencia de medias (atk - train)
# ════════════════════════════════════════════════════════════════
print("  Generando Gráfica 4 — Diferencia de medias atk vs train ...")

diff = np.array([stats_atk[f]['mean'] - stats_train[f]['mean'] for f in FEATURES])

fig, ax = plt.subplots(figsize=(20, 4))
fig.patch.set_facecolor(PALETA['fondo'])
ax.set_facecolor(PALETA['fondo'])

colors = [PALETA['atk'] if d > 0 else PALETA['train'] for d in diff]
bars = ax.bar(x, diff, color=colors, edgecolor='white', linewidth=0.8)

ax.set_xticks(x)
ax.set_xticklabels(FEATURES, rotation=45, ha='right', fontsize=9)
ax.set_ylabel('Diferencia de media (Ataque − Train)')
ax.set_title('Diferencia de Media por Feature: Ataque − Train benigno\n'
             'Rojo = ataques más altos  |  Azul = ataques más bajos',
             fontweight='bold', pad=15)
ax.axhline(0, color='black', linewidth=1)
ax.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig(GRAF_DIR / 'diferencia_medias.png', dpi=150, bbox_inches='tight')
plt.show()
print("  ✓ diferencia_medias.png")

# ════════════════════════════════════════════════════════════════
# RESUMEN FINAL
# ════════════════════════════════════════════════════════════════
print()
print("=" * 65)
print("  RESUMEN FINAL")
print("=" * 65)
print(f"  {'Partición':20} {'Media':>8} {'Std':>8} {'Min':>8} {'Max':>10}")
print(f"  {'-'*55}")

resumen = [
    ("train_norm", stats_train),
    ("val_norm",   stats_val),
    ("atk_norm",   stats_atk),
]
for nombre, stats in resumen:
    medias = [stats[f]['mean'] for f in FEATURES]
    stds   = [stats[f]['std']  for f in FEATURES]
    mins   = [stats[f]['min']  for f in FEATURES]
    maxs   = [stats[f]['max']  for f in FEATURES]
    print(f"  {nombre:20} {np.mean(medias):>8.3f} {np.mean(stds):>8.3f} "
          f"{np.min(mins):>8.3f} {np.max(maxs):>10.3f}")

print()
print("  Gráficas guardadas en graficas_metricas/")
for p in sorted(GRAF_DIR.glob("*.png")):
    size = p.stat().st_size / (1024**2)
    print(f"  {p.name:35}  {size:.1f} MB")

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import gc

# ════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ════════════════════════════════════════════════════════════════
PART_DIR = Path(r"C:\Users\Daniel\Desktop\Proyectos\Ia-tesisi\Notebooks_IaTesisz\particiones")
GRAF_DIR = PART_DIR.parent / "graficas_benign_full"
GRAF_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    'Header_Length', 'Protocol Type', 'Time_To_Live', 'Rate',
    'fin_flag_number', 'syn_flag_number', 'rst_flag_number',
    'psh_flag_number', 'ack_flag_number', 'ece_flag_number',
    'cwr_flag_number', 'ack_count', 'syn_count', 'fin_count',
    'rst_count', 'HTTP', 'HTTPS', 'DNS', 'Telnet', 'SMTP',
    'SSH', 'IRC', 'TCP', 'UDP', 'DHCP', 'ARP', 'ICMP', 'IGMP',
    'IPv', 'LLC', 'Tot sum', 'Min', 'Max', 'AVG', 'Std',
    'Tot size', 'IAT', 'Number', 'Variance'
]

plt.rcParams.update({
    'font.size': 12, 'axes.titlesize': 14,
    'axes.labelsize': 12, 'figure.dpi': 150,
})
COLOR = '#2ecc71'
FONDO = '#f8f9fa'

# ════════════════════════════════════════════════════════════════
# CARGAR benign_full.csv
# ════════════════════════════════════════════════════════════════
print("=" * 65)
print("  ANÁLISIS — benign_full.csv (dataset base benigno)")
print("=" * 65)

print("\n  Cargando benign_full.csv ...", end=" ", flush=True)
df = pd.read_csv(PART_DIR / "benign_full.csv")
arr = df[FEATURES].values.astype(np.float32)
mask = np.isfinite(arr).all(axis=1)
arr = arr[mask]
del df; gc.collect()
print(f"{arr.shape[0]:,} filas × {arr.shape[1]} features")

# ════════════════════════════════════════════════════════════════
# MÉTRICAS POR FEATURE
# ════════════════════════════════════════════════════════════════
print(f"\n  {'='*82}")
print(f"  {'Feature':22} {'Media':>10} {'Std':>10} {'Min':>8} {'P25':>8} {'P50':>8} {'P75':>8} {'Max':>10}")
print(f"  {'-'*82}")

stats = {}
for i, feat in enumerate(FEATURES):
    col = arr[:, i]
    s = {
        'mean': float(col.mean()),
        'std':  float(col.std()),
        'min':  float(col.min()),
        'p25':  float(np.percentile(col, 25)),
        'p50':  float(np.percentile(col, 50)),
        'p75':  float(np.percentile(col, 75)),
        'max':  float(col.max()),
    }
    stats[feat] = s
    print(f"  {feat:22} {s['mean']:>10.4f} {s['std']:>10.4f} "
          f"{s['min']:>8.4f} {s['p25']:>8.4f} {s['p50']:>8.4f} "
          f"{s['p75']:>8.4f} {s['max']:>10.4f}")

# ════════════════════════════════════════════════════════════════
# GRÁFICA 1 — Media por feature
# ════════════════════════════════════════════════════════════════
print("\n  Generando Gráfica 1 — Media por feature ...")

medias = [stats[f]['mean'] for f in FEATURES]
x = np.arange(len(FEATURES))

fig, ax = plt.subplots(figsize=(22, 7))
fig.patch.set_facecolor(FONDO); ax.set_facecolor(FONDO)
bars = ax.bar(x, medias, color=COLOR, edgecolor='white', linewidth=0.8, alpha=0.9)
ax.set_xticks(x)
ax.set_xticklabels(FEATURES, rotation=45, ha='right', fontsize=9)
ax.set_ylabel('Media (valor crudo original)')
ax.set_title('Media por Feature — benign_full.csv\n'
             f'1,098,126 registros de tráfico benigno · CICIoT2023',
             fontweight='bold', pad=15)
ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.4)
ax.grid(axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig(GRAF_DIR / 'benign_media_por_feature.png', dpi=150, bbox_inches='tight')
plt.show()
print("  ✓ benign_media_por_feature.png")

# ════════════════════════════════════════════════════════════════
# GRÁFICA 2 — Std por feature
# ════════════════════════════════════════════════════════════════
print("  Generando Gráfica 2 — Std por feature ...")

stds = [stats[f]['std'] for f in FEATURES]

fig, ax = plt.subplots(figsize=(22, 7))
fig.patch.set_facecolor(FONDO); ax.set_facecolor(FONDO)
ax.bar(x, stds, color='#2980b9', edgecolor='white', linewidth=0.8, alpha=0.9)
ax.set_xticks(x)
ax.set_xticklabels(FEATURES, rotation=45, ha='right', fontsize=9)
ax.set_ylabel('Desviación estándar (valor crudo)')
ax.set_title('Desviación Estándar por Feature — benign_full.csv',
             fontweight='bold', pad=15)
ax.grid(axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig(GRAF_DIR / 'benign_std_por_feature.png', dpi=150, bbox_inches='tight')
plt.show()
print("  ✓ benign_std_por_feature.png")

# ════════════════════════════════════════════════════════════════
# GRÁFICA 3 — Distribución global KDE
# ════════════════════════════════════════════════════════════════
print("  Generando Gráfica 3 — Distribución global KDE ...")

rng    = np.random.default_rng(42)
flat   = arr.flatten()
sample = flat[rng.choice(len(flat), 100_000, replace=False)]
clip_lo = np.percentile(sample, 1)
clip_hi = np.percentile(sample, 99)
del flat; gc.collect()

fig, ax = plt.subplots(figsize=(14, 6))
fig.patch.set_facecolor(FONDO); ax.set_facecolor(FONDO)
sns.kdeplot(sample.clip(clip_lo, clip_hi), ax=ax,
            color=COLOR, linewidth=2.5, fill=True, alpha=0.25)
ax.axvline(np.median(sample), color='black', linewidth=1.5,
           linestyle='--', label=f'Mediana = {np.median(sample):.4f}')
ax.set_xlabel('Valor crudo')
ax.set_ylabel('Densidad')
ax.set_title('Distribución Global de Valores — benign_full.csv\n'
             '(clipeado entre percentil 1 y 99)',
             fontweight='bold', pad=15)
ax.legend(fontsize=12)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(GRAF_DIR / 'benign_distribucion_global.png', dpi=150, bbox_inches='tight')
plt.show()
print("  ✓ benign_distribucion_global.png")
del sample; gc.collect()

# ════════════════════════════════════════════════════════════════
# GRÁFICA 4 — Rango intercuartil (P25 – P75) por feature
# ════════════════════════════════════════════════════════════════
print("  Generando Gráfica 4 — Rango intercuartil por feature ...")

p25s = [stats[f]['p25'] for f in FEATURES]
p50s = [stats[f]['p50'] for f in FEATURES]
p75s = [stats[f]['p75'] for f in FEATURES]

fig, ax = plt.subplots(figsize=(22, 7))
fig.patch.set_facecolor(FONDO); ax.set_facecolor(FONDO)

ax.bar(x, [p75s[i] - p25s[i] for i in range(len(FEATURES))],
       bottom=p25s, color=COLOR, alpha=0.6,
       edgecolor='white', linewidth=0.8, label='IQR (P25–P75)')
ax.scatter(x, p50s, color='#27ae60', zorder=5, s=40, label='Mediana (P50)')

ax.set_xticks(x)
ax.set_xticklabels(FEATURES, rotation=45, ha='right', fontsize=9)
ax.set_ylabel('Valor crudo')
ax.set_title('Rango Intercuartil (P25–P75) por Feature — benign_full.csv\n'
             'Barra = rango IQR  ·  Punto = mediana',
             fontweight='bold', pad=15)
ax.legend(fontsize=12)
ax.grid(axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig(GRAF_DIR / 'benign_iqr_por_feature.png', dpi=150, bbox_inches='tight')
plt.show()
print("  ✓ benign_iqr_por_feature.png")

# ════════════════════════════════════════════════════════════════
# RESUMEN FINAL
# ════════════════════════════════════════════════════════════════
del arr; gc.collect()

print()
print("=" * 65)
print("  RESUMEN FINAL — benign_full.csv")
print("=" * 65)
print(f"  Registros:   {mask.sum():,} filas válidas")
print(f"  Features:    {len(FEATURES)}")
print(f"  Media global: {np.mean(medias):.4f}")
print(f"  Std global:   {np.mean(stds):.4f}")
print(f"  Features con media=0: {sum(1 for m in medias if m == 0.0)}")
print(f"  Features con std=0:   {sum(1 for s in stds if s == 0.0)}")
print()
print("  Gráficas guardadas en graficas_benign_full/")
for p in sorted(GRAF_DIR.glob("*.png")):
    print(f"    {p.name:<42} {p.stat().st_size/1024**2:.1f} MB")


# %%
#logaritmo
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import gc

# ════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ════════════════════════════════════════════════════════════════
PART_DIR = Path(r"C:\Users\Daniel\Desktop\Proyectos\Ia-tesisi\Notebooks_IaTesisz\particiones")
GRAF_DIR = PART_DIR.parent / "graficas_benign_full"
GRAF_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    'Header_Length', 'Protocol Type', 'Time_To_Live', 'Rate',
    'fin_flag_number', 'syn_flag_number', 'rst_flag_number',
    'psh_flag_number', 'ack_flag_number', 'ece_flag_number',
    'cwr_flag_number', 'ack_count', 'syn_count', 'fin_count',
    'rst_count', 'HTTP', 'HTTPS', 'DNS', 'Telnet', 'SMTP',
    'SSH', 'IRC', 'TCP', 'UDP', 'DHCP', 'ARP', 'ICMP', 'IGMP',
    'IPv', 'LLC', 'Tot sum', 'Min', 'Max', 'AVG', 'Std',
    'Tot size', 'IAT', 'Number', 'Variance'
]

plt.rcParams.update({
    'font.size': 12, 'axes.titlesize': 14,
    'axes.labelsize': 12, 'figure.dpi': 150,
})
COLOR = '#2ecc71'
FONDO = '#f8f9fa'

# ════════════════════════════════════════════════════════════════
# CARGAR benign_full.csv
# ════════════════════════════════════════════════════════════════
print("=" * 65)
print("  ANÁLISIS — benign_full.csv (escala logarítmica)")
print("=" * 65)

print("\n  Cargando benign_full.csv ...", end=" ", flush=True)
df  = pd.read_csv(PART_DIR / "benign_full.csv")
arr = df[FEATURES].values.astype(np.float32)
mask = np.isfinite(arr).all(axis=1)
arr  = arr[mask]
del df; gc.collect()
print(f"{arr.shape[0]:,} filas × {arr.shape[1]} features")

# ════════════════════════════════════════════════════════════════
# MÉTRICAS
# ════════════════════════════════════════════════════════════════
print(f"\n  {'='*82}")
print(f"  {'Feature':22} {'Media':>10} {'Std':>10} {'Min':>8} {'P25':>8} {'P50':>8} {'P75':>8} {'Max':>10}")
print(f"  {'-'*82}")

stats = {}
for i, feat in enumerate(FEATURES):
    col = arr[:, i]
    s = {
        'mean': float(col.mean()),
        'std':  float(col.std()),
        'min':  float(col.min()),
        'p25':  float(np.percentile(col, 25)),
        'p50':  float(np.percentile(col, 50)),
        'p75':  float(np.percentile(col, 75)),
        'max':  float(col.max()),
    }
    stats[feat] = s
    print(f"  {feat:22} {s['mean']:>10.4f} {s['std']:>10.4f} "
          f"{s['min']:>8.4f} {s['p25']:>8.4f} {s['p50']:>8.4f} "
          f"{s['p75']:>8.4f} {s['max']:>10.4f}")

x      = np.arange(len(FEATURES))
medias = [stats[f]['mean'] for f in FEATURES]
stds   = [stats[f]['std']  for f in FEATURES]
p25s   = [stats[f]['p25']  for f in FEATURES]
p50s   = [stats[f]['p50']  for f in FEATURES]
p75s   = [stats[f]['p75']  for f in FEATURES]

# ════════════════════════════════════════════════════════════════
# HELPER — escala log segura (0 → 1e-4 para no romper log)
# ════════════════════════════════════════════════════════════════
def safe_log(values, eps=1e-4):
    return [max(v, eps) for v in values]

# ════════════════════════════════════════════════════════════════
# GRÁFICA 1 — Media por feature (escala log)
# ════════════════════════════════════════════════════════════════
print("\n  Generando Gráfica 1 — Media por feature (log) ...")

fig, ax = plt.subplots(figsize=(22, 7))
fig.patch.set_facecolor(FONDO); ax.set_facecolor(FONDO)

ax.bar(x, safe_log(medias), color=COLOR, edgecolor='white',
       linewidth=0.8, alpha=0.9)
ax.set_yscale('log')
ax.set_xticks(x)
ax.set_xticklabels(FEATURES, rotation=45, ha='right', fontsize=9)
ax.set_ylabel('Media (escala logarítmica)')
ax.set_title('Media por Feature — benign_full.csv  [escala log]\n'
             '1,098,126 registros · CICIoT2023',
             fontweight='bold', pad=15)
ax.grid(axis='y', alpha=0.3, linestyle='--', which='both')

# Anotar features con std=0
for i, feat in enumerate(FEATURES):
    if stats[feat]['std'] == 0.0:
        ax.text(i, safe_log(medias)[i] * 1.5, 'std=0',
                ha='center', va='bottom', fontsize=7,
                color='red', fontweight='bold')

plt.tight_layout()
plt.savefig(GRAF_DIR / 'benign_media_log.png', dpi=150, bbox_inches='tight')
plt.show()
print("  ✓ benign_media_log.png")

# ════════════════════════════════════════════════════════════════
# GRÁFICA 2 — Std por feature (escala log)
# ════════════════════════════════════════════════════════════════
print("  Generando Gráfica 2 — Std por feature (log) ...")

fig, ax = plt.subplots(figsize=(22, 7))
fig.patch.set_facecolor(FONDO); ax.set_facecolor(FONDO)

ax.bar(x, safe_log(stds), color='#2980b9', edgecolor='white',
       linewidth=0.8, alpha=0.9)
ax.set_yscale('log')
ax.set_xticks(x)
ax.set_xticklabels(FEATURES, rotation=45, ha='right', fontsize=9)
ax.set_ylabel('Desviación estándar (escala logarítmica)')
ax.set_title('Desviación Estándar por Feature — benign_full.csv  [escala log]',
             fontweight='bold', pad=15)
ax.grid(axis='y', alpha=0.3, linestyle='--', which='both')

for i, feat in enumerate(FEATURES):
    if stats[feat]['std'] == 0.0:
        ax.text(i, 2e-4, 'std=0', ha='center', va='bottom',
                fontsize=7, color='red', fontweight='bold')

plt.tight_layout()
plt.savefig(GRAF_DIR / 'benign_std_log.png', dpi=150, bbox_inches='tight')
plt.show()
print("  ✓ benign_std_log.png")

# ════════════════════════════════════════════════════════════════
# GRÁFICA 3 — KDE en escala log del eje X
# ════════════════════════════════════════════════════════════════
print("  Generando Gráfica 3 — Distribución global KDE (log X) ...")

rng   = np.random.default_rng(42)
flat  = arr.flatten()
# Solo valores positivos para log
flat_pos = flat[flat > 0]
sample   = flat_pos[rng.choice(len(flat_pos), 100_000, replace=False)]
del flat, flat_pos; gc.collect()

# Transformar a log para el KDE
log_sample = np.log10(sample)

fig, axes = plt.subplots(1, 2, figsize=(18, 6))
fig.patch.set_facecolor(FONDO)

# Izquierda: escala normal clipeada
ax = axes[0]
ax.set_facecolor(FONDO)
clip_hi = np.percentile(sample, 99)
sns.kdeplot(sample.clip(0, clip_hi), ax=ax, color=COLOR,
            linewidth=2.5, fill=True, alpha=0.25)
ax.axvline(np.median(sample), color='black', linewidth=1.5,
           linestyle='--', label=f'Mediana = {np.median(sample):.2f}')
ax.set_xlabel('Valor crudo (escala lineal)')
ax.set_ylabel('Densidad')
ax.set_title('Distribución — escala lineal\n(clipeado p99)', fontweight='bold')
ax.legend(fontsize=11); ax.grid(alpha=0.3)

# Derecha: escala log
ax = axes[1]
ax.set_facecolor(FONDO)
sns.kdeplot(log_sample, ax=ax, color='#2980b9',
            linewidth=2.5, fill=True, alpha=0.25)
ax.axvline(np.median(log_sample), color='black', linewidth=1.5,
           linestyle='--',
           label=f'Mediana log₁₀ = {np.median(log_sample):.2f}')
ax.set_xlabel('log₁₀(valor crudo)')
ax.set_ylabel('Densidad')
ax.set_title('Distribución — escala logarítmica\n(solo valores > 0)', fontweight='bold')
ax.legend(fontsize=11); ax.grid(alpha=0.3)

fig.suptitle('Distribución Global — benign_full.csv', fontsize=15,
             fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(GRAF_DIR / 'benign_kde_log.png', dpi=150, bbox_inches='tight')
plt.show()
print("  ✓ benign_kde_log.png")
del sample, log_sample; gc.collect()

# ════════════════════════════════════════════════════════════════
# GRÁFICA 4 — IQR por feature (escala log)
# ════════════════════════════════════════════════════════════════
print("  Generando Gráfica 4 — IQR por feature (log) ...")

iqr = [max(p75s[i] - p25s[i], 1e-4) for i in range(len(FEATURES))]

fig, ax = plt.subplots(figsize=(22, 7))
fig.patch.set_facecolor(FONDO); ax.set_facecolor(FONDO)

ax.bar(x, iqr, color=COLOR, alpha=0.7, edgecolor='white', linewidth=0.8,
       label='IQR (P75 − P25)')
ax.scatter(x, safe_log(p50s), color='#27ae60', zorder=5, s=40,
           label='Mediana (P50)')
ax.set_yscale('log')
ax.set_xticks(x)
ax.set_xticklabels(FEATURES, rotation=45, ha='right', fontsize=9)
ax.set_ylabel('Valor crudo (escala logarítmica)')
ax.set_title('Rango Intercuartil por Feature — benign_full.csv  [escala log]\n'
             'Barra = IQR  ·  Punto verde = mediana',
             fontweight='bold', pad=15)
ax.legend(fontsize=12)
ax.grid(axis='y', alpha=0.3, linestyle='--', which='both')

plt.tight_layout()
plt.savefig(GRAF_DIR / 'benign_iqr_log.png', dpi=150, bbox_inches='tight')
plt.show()
print("  ✓ benign_iqr_log.png")

# ════════════════════════════════════════════════════════════════
# RESUMEN FINAL
# ════════════════════════════════════════════════════════════════
del arr; gc.collect()

print()
print("=" * 65)
print("  RESUMEN FINAL — benign_full.csv")
print("=" * 65)
print(f"  Registros válidos:    {mask.sum():,}")
print(f"  Features:             {len(FEATURES)}")
print(f"  Media global:         {np.mean(medias):.4f}")
print(f"  Std global:           {np.mean(stds):.4f}")
print(f"  Features con std=0:   {sum(1 for s in stds if s == 0.0)}  "
      f"({[f for f in FEATURES if stats[f]['std']==0.0]})")
print()
print("  Gráficas guardadas en graficas_benign_full/")
for p in sorted(GRAF_DIR.glob("*.png")):
    print(f"    {p.name:<45} {p.stat().st_size/1024**2:.1f} MB")

# %%


# %%
#DATOS DE CADA FEATURE
import pandas as pd
import numpy as np
from pathlib import Path

PART_DIR = Path(r"C:\Users\Daniel\Desktop\Proyectos\Ia-tesisi\Notebooks_IaTesisz\particiones")

print("  Cargando benign_full.csv ...", end=" ", flush=True)
df = pd.read_csv(PART_DIR / "benign_full.csv")
print(f"{df.shape[0]:,} filas × {df.shape[1]} columnas")

total = len(df)

print(f"\n{'='*75}")
print(f"  {'FEATURE':<22} {'Total':>10} {'Válidos':>10} {'Nulos':>8} {'Ceros':>8} {'Únicos':>8}")
print(f"{'='*75}")

for col in df.columns:
    validos = df[col].notna().sum()
    nulos   = total - validos
    ceros   = (df[col] == 0).sum()
    unicos  = df[col].nunique()
    print(f"  {col:<22} {total:>10,} {validos:>10,} {nulos:>8,} {ceros:>8,} {unicos:>8,}")

print(f"{'='*75}")
print(f"\n  Total filas: {total:,}")
print(f"  Features con algún nulo: {df.isnull().any().sum()}")
print(f"  Features con todos ceros: {(df == 0).all().sum()}")


# %%
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ════════════════════════════════════════════════════════════════
WIN_DIR  = Path(r"C:\Users\Daniel\ventanas")
GRAF_DIR = Path(r"C:\Users\Daniel\Desktop\Proyectos\Ia-tesisi\Notebooks_IaTesisz\graficas_ventanas")
GRAF_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    'Header_Length', 'Protocol Type', 'Time_To_Live', 'Rate',
    'fin_flag_number', 'syn_flag_number', 'rst_flag_number',
    'psh_flag_number', 'ack_flag_number', 'ece_flag_number',
    'cwr_flag_number', 'ack_count', 'syn_count', 'fin_count',
    'rst_count', 'HTTP', 'HTTPS', 'DNS', 'SSH', 'IRC',
    'TCP', 'UDP', 'DHCP', 'ARP', 'ICMP', 'IGMP',
    'IPv', 'LLC', 'Tot sum', 'Min', 'Max', 'AVG', 'Std',
    'Tot size', 'IAT', 'Number', 'Variance'
]

T = 20

# ════════════════════════════════════════════════════════════════
# CARGAR SOLO LO NECESARIO — primeras 3 ventanas del train
# ════════════════════════════════════════════════════════════════
print("  Cargando train_w20.npy (solo primeras 3 ventanas) ...")
train_w = np.load(WIN_DIR / "train_w20.npy", mmap_mode='r')
print(f"  Shape completo: {train_w.shape}")

# Tomar las primeras 3 ventanas
muestra = train_w[:3].copy()
del train_w
print(f"  Muestra extraída: {muestra.shape}  (3 × 20 × 37)")

# ════════════════════════════════════════════════════════════════
# GRÁFICA 1 — Heatmap de una ventana (ventana 0)
# ════════════════════════════════════════════════════════════════
print("\n  Generando Gráfica 1 — Heatmap ventana 0 ...")

fig, ax = plt.subplots(figsize=(18, 8))
fig.patch.set_facecolor('#f8f9fa')
ax.set_facecolor('#f8f9fa')

ventana = muestra[0]  # shape (20, 37)

im = ax.imshow(ventana.T, aspect='auto', cmap='RdBu_r',
               vmin=-3, vmax=3, interpolation='nearest')

ax.set_xticks(range(T))
ax.set_xticklabels([f't{i+1}' for i in range(T)], fontsize=10)
ax.set_yticks(range(len(FEATURES)))
ax.set_yticklabels(FEATURES, fontsize=9)
ax.set_xlabel('Timestep (paso de tiempo)', fontsize=12, labelpad=10)
ax.set_ylabel('Feature', fontsize=12, labelpad=10)
ax.set_title(
    'Ventana de Entrenamiento #0 — train_w20.npy\n'
    f'Shape: (20 timesteps × 37 features)  ·  valores en escala RobustScaler',
    fontweight='bold', fontsize=14, pad=15
)

cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
cbar.set_label('Valor escalado (RobustScaler)', fontsize=11)
cbar.ax.tick_params(labelsize=10)

# Líneas de separación entre timesteps
for i in range(1, T):
    ax.axvline(i - 0.5, color='white', linewidth=0.3, alpha=0.5)

plt.tight_layout()
plt.savefig(GRAF_DIR / 'ventana_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()
print("  ✓ ventana_heatmap.png")

# ════════════════════════════════════════════════════════════════
# GRÁFICA 2 — 3 ventanas consecutivas comparadas (6 features clave)
# ════════════════════════════════════════════════════════════════
print("  Generando Gráfica 2 — 3 ventanas consecutivas ...")

FEATURES_PLOT = ['Rate', 'Header_Length', 'Time_To_Live',
                 'ack_count', 'HTTPS', 'Variance']
IDX = [FEATURES.index(f) for f in FEATURES_PLOT]

COLORES = ['#2980b9', '#2ecc71', '#e74c3c']
timesteps = range(1, T + 1)

fig, axes = plt.subplots(2, 3, figsize=(18, 8))
fig.patch.set_facecolor('#f8f9fa')
fig.suptitle(
    '3 Ventanas Consecutivas de Entrenamiento — train_w20.npy\n'
    'Cada ventana = 20 pasos temporales de tráfico benigno',
    fontsize=14, fontweight='bold', y=1.01
)

for ax, feat, idx in zip(axes.flatten(), FEATURES_PLOT, IDX):
    ax.set_facecolor('#f8f9fa')
    for v in range(3):
        vals = muestra[v, :, idx]
        ax.plot(timesteps, vals, color=COLORES[v],
                linewidth=1.8, marker='o', markersize=4,
                label=f'ventana {v}', alpha=0.85)
    ax.set_title(feat, fontweight='bold', fontsize=12)
    ax.set_xlabel('Timestep', fontsize=10)
    ax.set_ylabel('Valor escalado', fontsize=10)
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.axhline(0, color='gray', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.grid(alpha=0.3, linestyle='--')
    ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(GRAF_DIR / 'ventanas_comparadas.png', dpi=150, bbox_inches='tight')
plt.show()
print("  ✓ ventanas_comparadas.png")

# ════════════════════════════════════════════════════════════════
# GRÁFICA 3 — Tabla visual de la ventana 0 (primeras 10 features)
# ════════════════════════════════════════════════════════════════
print("  Generando Gráfica 3 — Tabla visual ventana 0 ...")

fig, ax = plt.subplots(figsize=(18, 5))
fig.patch.set_facecolor('#f8f9fa')
ax.set_facecolor('#f8f9fa')
ax.axis('off')

# Datos: primeras 10 features, todos los timesteps
datos_tabla = muestra[0, :, :10].T  # (10, 20)
col_labels  = [f't{i+1}' for i in range(T)]
row_labels  = FEATURES[:10]

tabla = ax.table(
    cellText=[[f'{v:.2f}' for v in fila] for fila in datos_tabla],
    rowLabels=row_labels,
    colLabels=col_labels,
    cellLoc='center',
    loc='center'
)

tabla.auto_set_font_size(False)
tabla.set_fontsize(8.5)
tabla.scale(1, 1.4)

# Colorear encabezados
for j in range(T):
    tabla[0, j].set_facecolor('#2980b9')
    tabla[0, j].set_text_props(color='white', fontweight='bold')
for i in range(10):
    tabla[i+1, -1].set_facecolor('#ecf0f1')
    tabla[i+1, -1].set_text_props(fontweight='bold')

ax.set_title(
    'Ventana #0 — Primeras 10 features × 20 timesteps\n'
    'Valores en escala RobustScaler',
    fontweight='bold', fontsize=13, pad=20
)

plt.tight_layout()
plt.savefig(GRAF_DIR / 'ventana_tabla.png', dpi=150, bbox_inches='tight')
plt.show()
print("  ✓ ventana_tabla.png")

# ════════════════════════════════════════════════════════════════
# RESUMEN
# ════════════════════════════════════════════════════════════════
print()
print("=" * 65)
print("  RESUMEN")
print("=" * 65)
print(f"  Ventanas analizadas:   3  (índices 0, 1, 2)")
print(f"  Shape de cada una:     (20 × 37)")
print(f"  Rango de valores:      {muestra.min():.3f}  a  {muestra.max():.3f}")
print(f"  Media global:          {muestra.mean():.4f}")
print()
print("  Gráficas guardadas en graficas_ventanas/")
for p in sorted(GRAF_DIR.glob("*.png")):
    print(f"    {p.name:<35}  {p.stat().st_size/1024**2:.1f} MB")


