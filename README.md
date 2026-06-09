# 📊 Proyecto 2 — Análisis Inicial y Selección de Problema

**Bootcamp:** Data Science & Machine Learning — Skillnest / Sonda  
**Etapa:** Parte I — Búsqueda, EDA inicial y Selección de Problema  
**Herramientas:** Python · Pandas · NumPy · Matplotlib · Seaborn · Scikit-learn · Google Colab  

---

## 📁 Estructura del repositorio

```
Proyecto2/
│
├── datasets/
│   ├── dataset1.csv          ← Bank Customer Churn
│   ├── dataset2.csv          ← Heart Disease
│   ├── dataset3.csv          ← Video Game Sales & Metacritic
│   └── dataset4.csv          ← Most Streamed Spotify Songs 2024
│
├── EDA/
│   ├── EDA_dataset1.ipynb    ← EDA Bank Customer Churn (df_bank)
│   ├── EDA_dataset2.ipynb    ← EDA Heart Disease (df_heart)
│   ├── EDA_dataset3.ipynb    ← EDA Video Game Sales (df_game)
│   └── EDA_dataset4.ipynb    ← EDA Spotify 2024 (df_spotify)
│   └── Proyecto2_EDA_Completo.ipynb     ← EDA de los 4 datasets + selección de problema
│
├── selected_dataset/
│   ├── selected_dataset.csv  ← Customer-Churn-Records.csv
│   └── EDA_selected_dataset.ipynb  ← EDA completo del dataset elegido
│
└── README.md

---
---

## 🗂️ Datasets analizados

| Variable | Dataset | Fuente | Dominio | Problema |
|---|---|---|---|---|
| `df_bank` | Bank Customer Churn | [Kaggle](https://www.kaggle.com/datasets/radheshyamkollipara/bank-customer-churn) | Banca / Finanzas | Clasificación |
| `df_heart` | Heart Disease | [Kaggle](https://www.kaggle.com/datasets/oktayrdeki/heart-disease) | Salud / Medicina | Clasificación |
| `df_game` | Video Game Sales + Metacritic | [Kaggle](https://www.kaggle.com/datasets/meruvakodandasuraj/video-game-sales-and-metacritic-intelligence-198026) | Entretenimiento | Regresión |
| `df_spotify` | Most Streamed Spotify Songs 2024 | [Kaggle](https://www.kaggle.com/datasets/nelgiriyewithana/most-streamed-spotify-songs-2024) | Música / Streaming | Regresión |

---

## 🔍 Resumen del EDA por dataset

### 🏦 df_bank — Bank Customer Churn
- **Tamaño:** 10.000 filas × 15 columnas útiles
- **Nulos:** ✅ Ninguno
- **Duplicados:** ✅ Ninguno
- **Desafío principal:** Desbalance de clases (~20% churn vs ~80% no churn)
- **Hallazgo clave:** `Complain` presenta alta correlación con `Exited` → posible data leakage a revisar en fase de modelado
- **Tipo de problema:** Clasificación binaria

### 🫀 df_heart — Heart Disease
- **Tamaño:** ~1.000 filas × 21 columnas
- **Nulos:** ⚠️ Presentes en múltiples columnas (hasta 25.9% en `Alcohol Consumption`) → imputados con mediana/moda según tipo de variable
- **Duplicados:** ✅ Ninguno
- **Desafío principal:** Imputación cuidadosa + múltiples variables categóricas binarias (Yes/No)
- **Hallazgo clave:** `CRP Level` y `Triglyceride Level` presentan distribuciones fuertemente sesgadas → transformación log recomendada
- **Tipo de problema:** Clasificación binaria

### 🎮 df_game — Video Game Sales & Metacritic
- **Tamaño:** 50.000 filas × 34 columnas útiles
- **Nulos:** ⚠️ Presentes en scores de Metacritic y año → imputados con mediana/Unknown
- **Duplicados:** Verificados y tratados
- **Desafío principal:** Asimetría extrema del target (`global_sales_million`) + alta cardinalidad en `publisher` (500+) y `platform` (30+)
- **Hallazgo clave:** Ventas con skewness > 10 → transformación log(1+x) necesaria para modelado
- **Tipo de problema:** Regresión

### 🎵 df_spotify — Most Streamed Spotify Songs 2024
- **Tamaño:** ~4.600 filas × 29 columnas
- **Nulos:** ⚠️ Plataformas secundarias (Pandora, Soundcloud, TIDAL) con nulos esperables
- **Duplicados:** ✅ Ninguno
- **Desafío principal:** 15 columnas numéricas almacenadas como strings con separadores de miles → conversión obligatoria antes de cualquier análisis
- **Hallazgo clave:** Alta multicolinealidad entre plataformas (Spotify, YouTube, TikTok, Shazam)
- **Tipo de problema:** Regresión

---

## ✅ Dataset seleccionado — Bank Customer Churn (`df_bank`)

### 🎯 Problemática: Clasificación binaria — Predicción de abandono de clientes (Churn)

**¿Qué es el churn?**  
El churn bancario ocurre cuando un cliente decide abandonar el banco. Predecirlo de forma anticipada permite que la institución tome medidas proactivas para retenerlo, como ofrecer beneficios, mejorar la atención o ajustar condiciones contractuales.

### ¿Por qué se eligió este dataset?

**1. Relevancia del problema de negocio**  
Adquirir un cliente nuevo cuesta entre 5 y 7 veces más que retener uno existente. Un modelo predictivo de churn tiene impacto directo en los ingresos y es uno de los casos de uso más demandados en la industria financiera.

**2. Riqueza y diversidad de features**

| Tipo | Variables |
|---|---|
| Demográficas | `Age`, `Gender`, `Geography` |
| Financieras | `Balance`, `CreditScore`, `EstimatedSalary` |
| Comportamiento | `IsActiveMember`, `NumOfProducts`, `Tenure`, `Complain` |
| Satisfacción | `Satisfaction Score`, `Point Earned`, `Card Type` |

**3. Desafíos técnicos equilibrados**

| Desafío | Nivel | Tratamiento |
|---|---|---|
| Desbalance de clases (~20% churn) | ⚠️ Medio | SMOTE o `class_weight='balanced'` |
| Variables categóricas | ✅ Manejable | One-Hot Encoding / Label Encoding |
| Balance bimodal (muchos clientes con $0) | ⚠️ Interesante | Análisis diferenciado |
| Sin nulos ni duplicados | ✅ Facilita | No requiere imputación |

**4. Modelos candidatos para la Parte II**

| Modelo | Rol |
|---|---|
| Logistic Regression | Baseline interpretable |
| Random Forest Classifier | Modelo principal |
| XGBoost / Gradient Boosting | Modelo avanzado |

**Métrica principal:** F1-Score (por desbalance de clases) + AUC-ROC

### ¿Por qué se descartaron los otros datasets?

| Dataset | Razón de descarte |
|---|---|
| `df_heart` | Tamaño reducido (~1.000 filas) — menor volumen para generalización del modelo |
| `df_game` | Asimetría extrema del target + alta cardinalidad dificultan el modelado de regresión |
| `df_spotify` | Multicolinealidad extrema entre plataformas + conversión de tipos añade complejidad sin valor diferencial |

---

## 🧰 Tecnologías utilizadas

| Librería | Uso |
|---|---|
| `pandas` | Manipulación y limpieza de datos |
| `numpy` | Operaciones numéricas |
| `matplotlib` | Visualizaciones base |
| `seaborn` | Visualizaciones estadísticas |
| `scipy` | Tests estadísticos (Shapiro-Wilk, Pearson) |
| `google.colab` | Entorno de ejecución y carga de credenciales |
| `kaggle` | Descarga automatizada de datasets |

---

## ▶️ Cómo ejecutar el notebook

1. Abrir `Proyecto2_EDA_Completo.ipynb` en Google Colab
2. En la celda de configuración, subir el archivo `kaggle.json` cuando se solicite
3. Ejecutar todas las celdas en orden (`Runtime → Run all`)

---

*Proyecto desarrollado como parte del Bootcamp de Data Science & Machine Learning — Skillnest / Sonda*
