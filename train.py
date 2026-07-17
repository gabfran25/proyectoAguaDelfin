import os
os.environ["KERAS_BACKEND"] = "torch"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pickle
import torch
import keras
from keras import layers, Model
from pathlib import Path
import time

# ════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ════════════════════════════════════════════════════════════════
WIN_DIR  = Path(r"C:\Users\Daniel\ventanas")
OUT_DIR  = Path(r"C:\Users\Daniel\Desktop\Proyectos\verano delfin\LSTM-AE\proyectoAguaDelfin\modelos")
OUT_DIR.mkdir(parents=True, exist_ok=True)

T        = 20    # timesteps por ventana
N_FEAT   = 37   # features
LATENT   = 32   # dimensión del espacio latente
EPOCHS   = 50
BATCH    = 256
LR       = 1e-3

# ════════════════════════════════════════════════════════════════
# VERIFICAR GPU
# ════════════════════════════════════════════════════════════════
print("=" * 65)
print("  ENTORNO")
print("=" * 65)
print(f"  Keras version:   {keras.__version__}")
print(f"  Keras backend:   {keras.backend.backend()}")
print(f"  PyTorch version: {torch.__version__}")
print(f"  CUDA disponible: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU:             {torch.cuda.get_device_name(0)}")
    print(f"  VRAM total:      {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print()

# ════════════════════════════════════════════════════════════════
# CARGAR DATOS
# ════════════════════════════════════════════════════════════════
print("=" * 65)
print("  CARGANDO DATOS")
print("=" * 65)

assert WIN_DIR.exists(), f"No se encontró {WIN_DIR}\nEjecuta el notebook de ventanas primero."

print(f"  train_w{T}.npy ...", end=" ")
X_train = np.load(WIN_DIR / f"train_w{T}.npy")
print(f"{X_train.shape}  dtype={X_train.dtype}")

print(f"  val_w{T}.npy   ...", end=" ")
X_val = np.load(WIN_DIR / f"val_w{T}.npy")
print(f"{X_val.shape}  dtype={X_val.dtype}")

assert X_train.shape[1] == T and X_train.shape[2] == N_FEAT, \
    f"Shape esperado (N,{T},{N_FEAT}), recibido {X_train.shape}"
print()

# ════════════════════════════════════════════════════════════════
# ARQUITECTURA LSTM-AUTOENCODER
# ════════════════════════════════════════════════════════════════
print("=" * 65)
print("  ARQUITECTURA  LSTM-AE")
print("=" * 65)

inputs = keras.Input(shape=(T, N_FEAT), name="entrada")

# Encoder
x = layers.LSTM(64, return_sequences=True, name="enc_lstm1")(inputs)
x = layers.LSTM(LATENT, return_sequences=False, name="enc_lstm2")(x)

# Bottleneck
codigo = layers.Dense(LATENT, activation="tanh", name="latente")(x)

# Decoder
x = layers.RepeatVector(T, name="repeat")(codigo)
x = layers.LSTM(LATENT, return_sequences=True, name="dec_lstm1")(x)
x = layers.LSTM(64, return_sequences=True, name="dec_lstm2")(x)
salida = layers.TimeDistributed(layers.Dense(N_FEAT), name="reconstruccion")(x)

model = Model(inputs=inputs, outputs=salida, name="LSTM_Autoencoder")
model.summary()
print()

# ════════════════════════════════════════════════════════════════
# COMPILAR Y ENTRENAR
# ════════════════════════════════════════════════════════════════
print("=" * 65)
print("  ENTRENAMIENTO")
print("=" * 65)
print(f"  Epochs:     {EPOCHS}")
print(f"  Batch size: {BATCH}")
print(f"  LR inicial: {LR}")
print(f"  Train:      {X_train.shape[0]:,} ventanas")
print(f"  Val:        {X_val.shape[0]:,} ventanas")
print()

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LR),
    loss="mse",
    metrics=["mae"],
)

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=8, restore_best_weights=True, verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6, verbose=1
    ),
    keras.callbacks.ModelCheckpoint(
        filepath=str(OUT_DIR / "lstm_ae_best.keras"),
        monitor="val_loss", save_best_only=True, verbose=0
    ),
]

t0 = time.time()
history = model.fit(
    X_train, X_train,
    validation_data=(X_val, X_val),
    epochs=EPOCHS,
    batch_size=BATCH,
    callbacks=callbacks,
    verbose=1,
)
elapsed = time.time() - t0

# ════════════════════════════════════════════════════════════════
# GUARDAR MODELO E HISTORIAL
# ════════════════════════════════════════════════════════════════
print()
print("=" * 65)
print("  GUARDANDO")
print("=" * 65)

model.save(OUT_DIR / "lstm_ae_final.keras")
print(f"  lstm_ae_final.keras guardado")

np.save(OUT_DIR / "history_loss.npy",     np.array(history.history["loss"]))
np.save(OUT_DIR / "history_val_loss.npy", np.array(history.history["val_loss"]))
print(f"  history_loss.npy guardado")

print()
print(f"  Tiempo de entrenamiento: {elapsed/60:.1f} min")
print(f"  Best val_loss: {min(history.history['val_loss']):.6f}")
print(f"  Epochs ejecutados: {len(history.history['loss'])}")
print()
print("  Siguiente paso: ejecutar evaluate.py")
