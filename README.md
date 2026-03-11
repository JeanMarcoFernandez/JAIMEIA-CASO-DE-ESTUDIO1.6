# Caso de Estudio 1.6 — Regresión y Clasificación en Datos de E-Commerce

## Objetivo del Caso

Este caso de estudio aborda dos problemas de aprendizaje supervisado sobre un dataset de envíos de e-commerce:

1. **Regresión:** Predecir el `Cost_of_the_Product` usando variables como bloque de almacén, modo de envío, calificaciones del cliente, peso del paquete e importancia del producto.
2. **Clasificación:** Predecir si un pedido llegó a tiempo (`Reached.on.Time_Y.N`) comparando un **Árbol de Decisión** y un **Random Forest**.

Se aplica un pipeline completo que incluye EDA, ingeniería de variables, codificación de categóricas, estandarización y entrenamiento con múltiples modelos.

---

## Dataset

**Nombre:** Customer Analytics — E-Commerce Shipping Dataset  
**Fuente:** Kaggle (`prachi13/customer-analytics`)  
**Enlace:** [https://www.kaggle.com/datasets/prachi13/customer-analytics](https://www.kaggle.com/datasets/prachi13/customer-analytics)  
**Registros:** 10,999 filas × 12 columnas

| Variable | Tipo | Descripción |
|---|---|---|
| `Warehouse_block` | Categórica nominal | Bloque del almacén (A–F) |
| `Mode_of_Shipment` | Categórica nominal | Modo de envío (Ship, Flight, Road) |
| `Customer_care_calls` | Numérica | Llamadas al servicio al cliente |
| `Customer_rating` | Numérica | Calificación del cliente (1–5) |
| `Cost_of_the_Product` | Numérica | **Target regresión** — costo del producto |
| `Prior_purchases` | Numérica | Compras previas del cliente |
| `Product_importance` | Ordinal | Importancia del producto (low / medium / high) |
| `Gender` | Categórica | Género del cliente *(eliminada en preprocesamiento)* |
| `Discount_offered` | Numérica | Descuento ofrecido *(binarizada → `High_Discount`)* |
| `Weight_in_gms` | Numérica | Peso del producto en gramos |
| `Reached.on.Time_Y.N` | Binaria | **Target clasificación** — ¿Entregado a tiempo? |

---

## Instrucciones para Reproducir

### 1. Requisitos previos

```bash
pip install pandas numpy matplotlib seaborn scikit-learn kagglehub jupyter
```

### 2. Configurar credenciales de Kaggle

El dataset se descarga automáticamente con `kagglehub`. Asegúrate de tener tu token de Kaggle configurado (`~/.kaggle/kaggle.json`).

### 3. Ejecutar el notebook

1. Instala la extensión **Jupyter** en VS Code (`Ctrl+Shift+X`)
2. Abre el archivo `Caso_Estudio1_6_v4.ipynb`
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
8. Regresión: train_test_split (90/10, estratificado por bins)
   ├── Regresión Lineal + StandardScaler
   ├── Ridge (α=1.0)
   └── Lasso (α=1.0, max_iter=1000)
9. Clasificación: train_test_split (85/15, estratificado)
   ├── Árbol de Decisión (max_depth=8)
   └── Random Forest (n_estimators=150, class_weight='balanced')
```

---

## Decisiones de Preprocesamiento

### Variables eliminadas
- **`ID`**: Identificador sin valor predictivo.
- **`Gender`**: Sin correlación relevante con los targets.
- **`Discount_offered`** *(post-EDA)*: Reemplazada por la variable binaria `High_Discount` (1 si descuento > 10, 0 si no), reduciendo ruido al modelo.

### Codificación
- **`Product_importance`** → `OrdinalEncoder` con orden explícito `['low', 'medium', 'high']`
- **`Warehouse_block`** y **`Mode_of_Shipment`** → `OneHotEncoder` con `handle_unknown='ignore'`

### Estandarización
- `StandardScaler` aplicado dentro de cada `Pipeline` para evitar data leakage.

---

## Resultados

### Regresión — Predicción de `Cost_of_the_Product`

| Modelo | MAE | RMSE |
|---|---|---|
| Regresión Lineal | ~0.77 | ~0.93 |
| Ridge (α=1.0) | ~0.77 | ~0.93 |
| Lasso (α=1.0) | ~0.84 | ~0.98 |

**R² ≈ 0.10** en todos los modelos.

> Los tres modelos presentan un desempeño casi idéntico y un R² muy bajo (~10%). Esto indica que las variables disponibles no tienen suficiente poder explicativo para predecir el costo del producto. El problema no es el tipo de modelo ni la regularización, sino la limitada capacidad informativa del dataset para esta tarea.

---

### Clasificación — Predicción de `Reached.on.Time_Y.N`

#### Árbol de Decisión (`max_depth=8`)

**Accuracy: ~67.9%**

| Clase | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| 0 (Retraso) | 0.56 | 0.94 | 0.70 | 665 |
| 1 (A tiempo) | 0.93 | 0.50 | 0.65 | 985 |

**Matriz de confusión:**

|  | Pred. 0 | Pred. 1 |
|---|---|---|
| **Real 0** | 627 (TN) | 38 (FP) |
| **Real 1** | 491 (FN) | 494 (TP) |

> El modelo es casi infalible detectando retrasos reales (recall 0.94), pero cuando predice un retraso, solo acierta el 56% de las veces. Para entregas a tiempo, su precisión es excelente (0.93), aunque solo se "atreve" a predecirlas en la mitad de los casos reales.

---

#### Random Forest (`n_estimators=150`, `class_weight='balanced'`)

**Accuracy: ~67.3%**

| Clase | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| 0 (Retraso) | 0.55 | 0.98 | 0.71 | 665 |
| 1 (A tiempo) | 0.97 | 0.46 | 0.63 | 985 |

**Matriz de confusión:**

|  | Pred. 0 | Pred. 1 |
|---|---|---|
| **Real 0** | 652 (TN) | 13 (FP) |
| **Real 1** | 527 (FN) | 458 (TP) |

> El Random Forest reduce drásticamente los Falsos Positivos (38 → 13) respecto al Árbol de Decisión, alcanzando una precisión del 97% para la clase 1. Sin embargo, su recall para entregas a tiempo cae al 46%. Ambos modelos presentan un accuracy similar (~67%), pero con perfiles de error distintos.

---

### Comparación de Modelos de Clasificación

| Modelo | Accuracy | Precision (clase 1) | Recall (clase 1) | F1 (clase 1) |
|---|---|---|---|---|
| Árbol de Decisión | 67.9% | 0.93 | 0.50 | 0.65 |
| Random Forest | 67.3% | 0.97 | 0.46 | 0.63 |

> La elección entre modelos depende del costo de negocio de cada tipo de error: si es más crítico evitar promesas incumplidas (falsos positivos), el Random Forest es preferible. Si se requiere identificar la mayor cantidad posible de entregas exitosas, el Árbol de Decisión tiene mejor recall.

---

## Conclusiones

- La **regresión lineal** con y sin regularización (Ridge/Lasso) no logra explicar el costo del producto con las variables disponibles (R² ≈ 10%). Se requieren features adicionales como categoría, marca o tipo de producto.
- El **Árbol de Decisión** y el **Random Forest** alcanzan una accuracy similar (~67%), pero con distintos trade-offs entre precisión y recall por clase.
- El dataset presenta un **desbalance moderado** entre clases (más pedidos a tiempo que retrasados), lo que justifica el uso de `class_weight='balanced'` en Random Forest.
- `Prior_purchases` presenta outliers notables, pero son coherentes con el modelo de negocio y se mantuvieron en el análisis.

### Recomendaciones
- Incorporar **feature engineering** adicional (e.g., interacciones entre peso y modo de envío).
- Probar **Gradient Boosting** (XGBoost, LightGBM) para ambas tareas.
- Aplicar **ajuste de umbral de decisión** para balancear recall entre clases según prioridad de negocio.
- Para regresión, explorar modelos no lineales (Random Forest Regressor, SVR) o incorporar variables más directamente relacionadas con el precio.

---

## Estructura del Proyecto

```
Caso_Estudio1_6_v4.ipynb   ← Notebook principal
README.md                  ← Este archivo
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
