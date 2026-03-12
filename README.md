# Caso de Estudio 1.6 — Análisis Predictivo de Entregas E-Commerce

> **Metodología:** CRISP-DM | **Universidad Católica Boliviana "San Pablo"** | **Materia:** Machine Learning | **Año:** 2026

## Integrantes

- Jean Marco Fernandez Silva
- Sergio Alejandro Arias Mayta
- Marvin Larry Mollo Ramirez
- Jaime Ignacio Huaycho Clavel
- Sergio Alexander Mendoza Choque

---

## Objetivo del Caso

Este caso de estudio aplica la metodología **CRISP-DM** sobre un dataset de envíos de e-commerce para abordar dos problemas de aprendizaje supervisado:

1. **Regresión:** Predecir el `Cost_of_the_Product` a partir de variables logísticas y del cliente.
2. **Clasificación:** Predecir si un pedido llegó a tiempo (`Reached.on.Time_Y.N`) comparando un **Árbol de Decisión** y un **Random Forest**.
3. **Análisis descriptivo:** Descubrir patrones ocultos en entregas y desempeño logístico para optimizar operaciones.

El proyecto transita de un modelo reactivo a uno **predictivo y proactivo**, permitiendo tomar decisiones antes de que los incidentes impacten al cliente.

---

## Dataset

**Nombre:** Customer Analytics — E-Commerce Shipping Dataset
**Fuente:** Kaggle (`prachi13/customer-analytics`)
**Enlace:** [https://www.kaggle.com/datasets/prachi13/customer-analytics](https://www.kaggle.com/datasets/prachi13/customer-analytics)
**Registros:** 10,999 filas × 12 columnas

| Variable | Tipo | Descripción |
|---|---|---|
| `Warehouse_block` | Categórica nominal | Bloque del almacén (A–F). El bloque F concentra el 33.3% de envíos |
| `Mode_of_Shipment` | Categórica nominal | Modo de envío (Ship 67.8%, Flight, Road) |
| `Customer_care_calls` | Numérica | Llamadas al servicio al cliente |
| `Customer_rating` | Numérica | Calificación del cliente (1–5) |
| `Cost_of_the_Product` | Numérica | **Target regresión** — costo del producto |
| `Prior_purchases` | Numérica | Compras previas del cliente (outliers conservados por justificación de negocio) |
| `Product_importance` | Ordinal | Importancia: low (5,297) / medium (4,754) / high (948) |
| `Gender` | Categórica | Género del cliente *(eliminada: bajo poder explicativo)* |
| `Discount_offered` | Numérica | Descuento ofrecido *(transformada → `High_Discount`)* |
| `Weight_in_gms` | Numérica | Peso del producto en gramos |
| `Reached.on.Time_Y.N` | Binaria | **Target clasificación** — 0: llegó a tiempo (4,436 / 40.3%), 1: retraso (6,563 / 59.7%) |

---

## Marco Metodológico: CRISP-DM

| Fase | Aplicación en este Proyecto |
|---|---|
| 1. Business Understanding | Definición del problema logístico, objetivos de regresión, clasificación y análisis |
| 2. Data Understanding | EDA: distribución del target, correlaciones, análisis de variables clave |
| 3. Data Preparation | Limpieza, codificación, ingeniería de características (`High_Discount`) |
| 4. Modeling | Regresión (OLS, Ridge, Lasso) y Clasificación (Árbol de Decisión, Random Forest) |
| 5. Evaluation | Métricas MAE, RMSE, R², Accuracy, Precision, Recall, F1-Score, matrices de confusión |
| 6. Deployment | Recomendaciones estratégicas para implementación productiva |

---

## Instrucciones para Reproducir

### 1. Requisitos previos

```bash
pip install pandas numpy matplotlib seaborn scikit-learn kagglehub jupyter
```

### 2. Configurar credenciales de Kaggle

El dataset se descarga automáticamente con `kagglehub`. Asegúrate de tener tu token configurado en `~/.kaggle/kaggle.json`.

### 3. Ejecutar el notebook

1. Instala la extensión **Jupyter** en VS Code (`Ctrl+Shift+X`)
2. Abre el archivo `Codigo.ipynb`
3. Selecciona el kernel de Python con las dependencias instaladas
4. Ejecuta todas las celdas con **Run All** (`Ctrl+Alt+R`)

---

## Flujo del Notebook

```
1. Descarga del dataset (kagglehub)
2. Exploración inicial (df.info(), value_counts())
3. Eliminación de columnas irrelevantes (ID, Gender)
4. Ingeniería de variables:
   └── Discount_offered → High_Discount (binaria, umbral > 10)
5. Codificación de variables categóricas:
   ├── OrdinalEncoder: Product_importance (low=0, medium=1, high=2)
   └── OneHotEncoder: Warehouse_block, Mode_of_Shipment
6. EDA:
   ├── Matriz de correlación (heatmap)
   ├── Distribución de descuentos vs entrega a tiempo (KDE)
   ├── Histogramas de variables numéricas
   └── Boxplots (detección de outliers)
7. Eliminación de Discount_offered (reemplazada por High_Discount)
8. Regresión: train_test_split (90/10 → 9,899 train / 1,100 test, estratificado por bins)
   ├── Regresión Lineal (OLS, baseline) + StandardScaler
   ├── Ridge (α=1.0, penalización L2)
   └── Lasso (α=1.0, max_iter=1000, penalización L1)
9. Clasificación: train_test_split (85/15 → 9,349 train / 1,650 test, estratificado)
   ├── Árbol de Decisión (criterion=gini, max_depth=8, min_samples_split=10, min_samples_leaf=10)
   └── Random Forest (n_estimators=150, max_depth=8, max_features=0.5, class_weight='balanced')
```

---

## Decisiones de Preprocesamiento

### Variables eliminadas
- **`ID`**: Identificador sin valor predictivo.
- **`Gender`**: Sin correlación relevante con los targets.
- **`Discount_offered`** *(post-EDA)*: Reemplazada por `High_Discount` (1 si descuento > 10, 0 si no) para reducir ruido y evitar redundancia con la variable derivada.

### Codificación
- **`Product_importance`** → `OrdinalEncoder` con orden explícito `['low'=0, 'medium'=1, 'high'=2]`, capturando la jerarquía de negocio.
- **`Warehouse_block`** y **`Mode_of_Shipment`** → `OneHotEncoder` con `handle_unknown='ignore'`, evitando que el modelo asuma jerarquías falsas.

### Calidad de datos
- Valores nulos: **ninguno encontrado** (`isnull().sum()`)
- Duplicados: **dataset limpio** (`duplicated().sum()`)
- Outliers en `Prior_purchases`: **conservados** por coherencia con el modelo de negocio.

### Estandarización
- `StandardScaler` aplicado dentro de cada `Pipeline` para evitar data leakage.

---

## Resultados

### Regresión — Predicción de `Cost_of_the_Product`

División: 90% entrenamiento (9,899 muestras) — 10% prueba (1,100 muestras)

| Modelo | MAE | RMSE | R² |
|---|---|---|---|
| Regresión Lineal (OLS) | 36.50 | 43.87 | 0.1785 |
| Ridge (α=1.0) | 36.50 | 43.87 | ≈ 0.1785 |
| Lasso (α=1.0) | 36.91 | 44.11 | — |

> El R² de ~0.18 indica que los modelos explican apenas el 17.85% de la variabilidad del costo. La similitud entre OLS y Ridge confirma que la limitación es **informativa**, no de optimización: el dataset no contiene variables suficientemente relacionadas con la formación del precio (categoría, marca, tipo de producto).

---

### Clasificación — Predicción de `Reached.on.Time_Y.N`

División: 85% entrenamiento (9,349 muestras) — 15% prueba (1,650 muestras)

#### Árbol de Decisión (`criterion=gini`, `max_depth=8`)

**Accuracy: 67.94%**

| Clase | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| 0 (A tiempo) | 0.56 | 0.94 | 0.70 | 665 |
| 1 (Retraso) | 0.93 | 0.50 | 0.65 | 985 |
| **Weighted avg** | **0.78** | **0.68** | **0.67** | **1,650** |

**Matriz de confusión:**

|  | Pred. 0 | Pred. 1 |
|---|---|---|
| **Real 0** | TN = 627 | FP = 38 |
| **Real 1** | FN = 491 | TP = 494 |

> Casi infalible detectando retrasos reales (recall 0.94), pero con precisión moderada del 56% en esas predicciones. Para entregas a tiempo, precisión excelente (0.93) pero solo acierta en la mitad de los casos reales.

---

#### Random Forest (`n_estimators=150`, `max_features=0.5`, `class_weight='balanced'`)

**Accuracy: 67.27%**

| Clase | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| 0 (A tiempo) | 0.55 | 0.98 | 0.71 | 665 |
| 1 (Retraso) | 0.97 | 0.46 | 0.63 | 985 |
| **Weighted avg** | **0.80** | **0.67** | **0.66** | **1,650** |

**Matriz de confusión:**

|  | Pred. 0 | Pred. 1 |
|---|---|---|
| **Real 0** | TN = 652 | FP = 13 |
| **Real 1** | FN = 527 | TP = 458 |

> Reduce drásticamente los Falsos Positivos (38 → 13) respecto al Árbol de Decisión, alcanzando una precisión del 97% para la clase 1. El recall de la clase 1 cae al 46%.

---

### Comparación y Trade-offs

| Modelo | Accuracy | Precision (clase 1) | Recall (clase 1) | FP | FN |
|---|---|---|---|---|---|
| Árbol de Decisión | 67.94% | 0.93 | 0.50 | 38 | 491 |
| Random Forest | 67.27% | 0.97 | 0.46 | 13 | 527 |

| Tipo de Error | Árbol | Random Forest | Impacto de negocio |
|---|---|---|---|
| Falsos Negativos (FN) | 491 | 527 | Retrasos no detectados — riesgo reputacional alto |
| Falsos Positivos (FP) | 38 | 13 | Alertas innecesarias — costo operativo menor |

> **Regla de decisión:** Si es prioritario **evitar promesas incumplidas** (minimizar FP), el Random Forest es preferible. Si se requiere **identificar la mayor cantidad de retrasos posibles** (maximizar recall), el Árbol de Decisión tiene ventaja.

---

## Conclusiones (CRISP-DM: Deployment)

El mayor valor del proyecto reside en el **modelo de clasificación**, que permite transitar de una gestión logística reactiva a una **predictiva y proactiva**:

- El sistema puede anticipar la mayoría de incumplimientos y activar acciones preventivas: priorización de envíos, reasignación de recursos o comunicación anticipada con el cliente.
- Los modelos de **regresión no se recomiendan para producción** en el estado actual (R² ≈ 0.18); se requieren variables comerciales adicionales.
- Ambos clasificadores ofrecen perspectivas **complementarias** para la toma de decisiones operativas.

### Recomendaciones estratégicas

- Incorporar **feature engineering** adicional (interacciones entre peso y modo de envío, categoría de producto, marca).
- Probar **Gradient Boosting** (XGBoost, LightGBM) para ambas tareas.
- Aplicar **ajuste de umbral de decisión** en clasificación para balancear recall entre clases según prioridad de negocio.
- Para regresión, explorar modelos no lineales (Random Forest Regressor, SVR) con variables de precio más directas.
- Considerar **GridSearchCV** para optimizar hiperparámetros, especialmente el alpha de Lasso.

---

## Estructura del Proyecto

```
Codigo.ipynb                          ← Notebook principal
CRISP_DM_Ecommerce_Logistics.docx     ← Informe técnico CRISP-DM
CRISP_DM_Entregas_Ecommerce.pptx      ← Presentación ejecutiva
README.md                             ← Este archivo
```

## Dependencias

```
pandas
numpy
matplotlib
seaborn
scikit-learn
kagglehub
jupyter
```
