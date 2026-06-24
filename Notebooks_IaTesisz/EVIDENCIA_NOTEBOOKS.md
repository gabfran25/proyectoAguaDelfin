# EVIDENCIA — Notebooks y Código del Proyecto
## IDS para IoT con LSTM Autoencoder · CICIoT2023
---

# ══════════════════════════════════════════
# BLOQUE 1 — CONFIGURACIÓN DE DATOS
# ══════════════════════════════════════════
> Notebooks que procesaron, transformaron y guardaron archivos.
> No generan análisis — preparan la materia prima para el modelo.

---

## 📦 analisis_de_datos.ipynb
**Rol:** Pipeline original de preprocesamiento  
**Dato clave:** Split 70 / 15 / 15 · MinMaxScaler · T=10  

| Qué hace | Resultado |
|---|---|
| Carga 309 CSVs crudos de `CSV\CSV\` | Convierte a float32, filtra inf/NaN |
| Muestrea 2,000 registros por clase de ataque | Balancea las 33 clases |
| Split benigno: 70% train / 15% val / 15% test | Solo benigno para train y val |
| MinMaxScaler ajustado solo en train | Rango [0, 1] para todas las particiones |
| Ventanas deslizantes T=10, step=1 | Shape: (768,679 · 10 · 39) |

**Archivos generados:**
```
benign_raw.npy        (1,098,126 × 39)   163 MB
atk_raw.npy           (65,252    × 39)     9 MB
atk_labels.npy        (65,252,)             5 MB
scaler.pkl            MinMaxScaler          0 MB
partition_train.csv   768,688 filas       197 MB
partition_val.csv     164,719 filas        43 MB
partition_test_benign.csv  164,719 filas   43 MB
partition_attacks.csv  65,252 filas        17 MB
```

---

## 📦 analisis_de_datos2.ipynb
**Rol:** Pipeline mejorado con split 80/20 y muestreo proporcional  
**Dato clave:** Split 80 / 20 · evaluación 50% benigno / 50% ataque  

| Qué hace | Resultado |
|---|---|
| Reconstruye split 80/20 desde `benign_full.csv` | +109,812 filas de entrenamiento vs pipeline original |
| Muestreo proporcional de ataques | Clases grandes tienen más peso (DDoS-ICMP: 32,528 muestras) |
| Genera `test_full.csv` balanceado | 50% benigno / 50% ataque (219,626 vs 219,604) |
| Mapea `CSV\CSV\` en objeto `catalogo` | 34 clases · 309 archivos · 8.33 GB totales |

**Archivos generados:**
```
particiones/
├── benign_full.csv        1,098,126 filas    217 MB  ← dataset base completo
├── benign_train_80.csv      878,500 filas    174 MB  ← entrenamiento
├── benign_val_20.csv        219,626 filas     43 MB  ← test benigno
├── attack_test.csv          219,604 filas     47 MB  ← test ataques
└── test_full.csv            439,230 filas     95 MB  ← test combinado 50/50
```

**Variables clave generadas en memoria:**
```python
catalogo       → dict  34 clases con rutas, tamaños y tipo
CLASES_ATAQUE  → list  33 nombres de ataque ordenados
df_catalogo    → DataFrame resumen del dataset
FEATURES       → list  39 columnas del dataset
GRUPOS_OSI     → dict  features agrupadas por capa (validado)
```

---

## 📦 analisis_de_datos3.ipynb
**Rol:** Normalización RobustScaler + ventanas deslizantes T=20  
**Dato clave:** RobustScaler · T=20 · step=20 · 37 features (sin Telnet/SMTP)  

| Qué hace | Resultado |
|---|---|
| Aplica RobustScaler sobre particiones | Fórmula: `(X - Q2) / (Q3 - Q1)` |
| Elimina Telnet y SMTP | Ambas siempre = 0, no aportan información |
| Ventanas T=20, step=20, sin overlap | 878,500 filas → 43,925 ventanas |

**Archivos generados:**
```
normalizacion/
├── train_norm.npy   (878,500 × 37)   float32
├── val_norm.npy     (219,626 × 37)   float32
└── atk_norm.npy     (219,604 × 37)   float32

ventanas/
├── train_w20.npy         (43,925 × 20 × 37)   float32   124 MB
├── val_w20.npy           (10,981 × 20 × 37)   float32    31 MB
├── atk_w20.npy           (10,980 × 20 × 37)   float32    31 MB
└── atk_labels_w20.npy    (10,980,)             str         1 MB
```

---

# ══════════════════════════════════════════
# BLOQUE 2 — ANÁLISIS DE DATOS
# ══════════════════════════════════════════
> Código que observó, midió y visualizó sin modificar ningún archivo.
> Generan comprensión del dataset y sus características.

---

## 🔍 analisis_de_datos_prac.ipynb
**Rol:** EDA completo del dataset CICIoT2023  
**Dato clave:** 46,775,660 registros totales · 2.35% benigno / 97.65% ataque  

| Qué analiza | Hallazgo clave |
|---|---|
| Distribución de clases | DDoS-ICMP_Flood domina con 7.2M registros |
| Estadísticas por feature | Telnet y SMTP siempre = 0 |
| Heatmap capas 2/3 por clase | DDoS-ICMP tiene ARP 80× mayor que benigno |
| TTL por tipo de ataque | Port Scan eleva TTL a 0.92 vs 0.45 benigno |

**Gráficas generadas:**
```
distribucion_clases.png    → 34 clases en escala logarítmica
division_datos.png         → pie charts split 70/15/15
analisis_capas_2_3.png     → heatmap features Capa2/Capa3 por clase
```

---

## 🔍 analisis_de_datos3.ipynb — sección métricas
**Rol:** Evaluación estadística de las particiones normalizadas  
**Dato clave:** RobustScaler · valores en desviaciones relativas a mediana benigna  

| Qué analiza | Hallazgo clave |
|---|---|
| Media por feature en train/val/atk | Train y Val prácticamente iguales → normalización consistente |
| Std por feature | Ataques tienen mayor dispersión en flags TCP |
| KDE global de valores | Distribución benigna concentrada cerca de 0 |
| Diferencia de medias atk − train | Features con mayor separación: candidatas detectoras |

**Gráficas generadas:**
```
graficas_metricas/
├── media_por_feature.png       → barras Train vs Val vs Ataque
├── std_por_feature.png         → dispersión por partición
├── distribucion_global.png     → KDE de los 3 conjuntos
└── diferencia_medias.png       → Δ(ataque − benigno) por feature
```

---

## 🔍 Código: features agrupadas por capa OSI
**Rol:** Visualización organizada de las 39 features por capa de red  
**Dato clave:** 5 grupos · validación automática de cobertura completa  

```
Capa 2 — Enlace:        2 features  (ARP, LLC)
Capa 3 — Red:           5 features  (Protocol Type, TTL, ICMP, IGMP, IPv)
Capa 4 — Transporte:   14 features  (TCP, UDP, flags, contadores)
Capa 7 — Aplicación:    8 features  (HTTP, HTTPS, DNS, SSH, IRC, DHCP, Telnet, SMTP)
Estadísticas de Flujo: 10 features  (Rate, Tot sum, Min, Max, AVG, Std, Tot size, IAT, Number, Variance)
```

**Gráficas generadas:**
```
features_por_grupo.png       → heatmap 34 clases × features por grupo
features_comparacion.png     → barras benigno vs ataque por grupo
```

---

## 🔍 Código: análisis de benign_full.csv
**Rol:** Análisis exhaustivo del dataset base benigno en crudo  
**Dato clave:** 1,098,126 filas · 39 features · valores sin normalizar  

| Qué analiza | Resultado |
|---|---|
| Media por feature | Valores en escala original del tráfico de red |
| Std por feature | Mide variabilidad natural del tráfico benigno |
| KDE global | Distribución real antes de cualquier transformación |
| IQR por feature (P25–P75) | Identifica features con alta/baja variabilidad |

**Gráficas generadas:**
```
graficas_benign_full/
├── benign_media_por_feature.png   → media cruda por feature
├── benign_std_por_feature.png     → desviación estándar cruda
├── benign_distribucion_global.png → KDE global (clip P1–P99)
└── benign_iqr_por_feature.png     → rango intercuartil + mediana
```

---

## 🔍 Código: análisis Telnet y SMTP
**Rol:** Análisis exhaustivo de las dos features constantes  
**Dato clave:** Ambas = 0 en el 100% de los registros benignosmuerte  

| Métrica | Telnet | SMTP |
|---|---|---|
| Valores únicos | 1 | 1 |
| Valores = 0 | 1,098,126 (100%) | 1,098,126 (100%) |
| Media | 0.000000 | 0.000000 |
| Std | 0.000000 | 0.000000 |
| Conclusión | ⚠ Eliminar del modelo | ⚠ Eliminar del modelo |

**Gráficas generadas:**
```
graficas_benign_full/
└── analisis_telnet_smtp.png   → frecuencia · boxplot · pie ceros/no-ceros
```

---

## 🔍 Código: reporte para residencias
**Rol:** Visualizaciones de presentación para el folleto/reporte  
**Dato clave:** Fuentes grandes · bien segmentado · para audiencia no técnica  

**Gráficas generadas:**
```
reporte_clases.png       → barras horizontales 34 clases por MB (verde/rojo)
reporte_proporcion.png   → donut benigno/ataque + top15 archivos
reporte_features_osi.png → barras verticales features por capa OSI
```

---

# ══════════════════════════════════════════
# BLOQUE 3 — PENDIENTE
# ══════════════════════════════════════════

## ⚠️ LSTM_AE.ipynb — VACÍO
**Rol:** Entrenamiento y evaluación del modelo principal  
**Estado:** Archivo creado pero sin contenido  

**Lo que debe contener:**
```
1. Cargar ventanas desde C:\Users\Daniel\ventanas\
2. Definir arquitectura LSTM Autoencoder
3. Entrenar sobre train_w20.npy (solo benigno)
4. Calibrar umbral θ = μ + k·σ sobre val_w20.npy
5. Evaluar sobre atk_w20.npy → F1, Precision, Recall, FPR
6. Guardar modelo entrenado (.h5 o .pt)
```

---

*Generado el 2026-05-22 · Proyecto de tesis IDS-IoT*
