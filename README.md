#  Caso de Estudio 1.6 — Regresión y Clasificación en Datos de E-Commerce

## Objetivo del Caso

Este caso de estudio aborda dos problemas de aprendizaje supervisado sobre un dataset de envíos de e-commerce:

1. **Regresión:** Predecir el `Cost_of_the_Product` (costo del producto) a partir de variables como el bloque de almacén, modo de envío, calificaciones del cliente, peso del paquete e importancia del producto.
2. **Clasificación:** Predecir si un pedido llegó a tiempo (`Reached.on.Time_Y.N`) utilizando las mismas variables, comparando modelos de Árbol de Decisión y Random Forest.

Se aplica un pipeline completo de EDA, codificación de variables categóricas, estandarización y entrenamiento con múltiples modelos para comparar su desempeño.

---


## Dataset

**Nombre:** Customer Analytics — E-Commerce Shipping Dataset  
**Fuente:** Kaggle (`prachi13/customer-analytics`)  
**Enlace de descarga:** [https://www.kaggle.com/datasets/prachi13/customer-analytics](https://www.kaggle.com/datasets/prachi13/customer-analytics)

El dataset contiene **10,999 registros** y 12 columnas con información sobre envíos de una empresa de comercio electrónico:

| Variable | Tipo | Descripción |
|---|---|---|
| `Warehouse_block` | Categórica | Bloque del almacén (A–F) |
| `Mode_of_Shipment` | Categórica | Modo de envío (Ship, Flight, Road) |
| `Customer_care_calls` | Numérica | Llamadas al servicio al cliente |
| `Customer_rating` | Numérica | Calificación del cliente (1–5) |
| `Cost_of_the_Product` | Numérica | Costo del producto (target regresión) |
| `Prior_purchases` | Numérica | Compras previas del cliente |
| `Product_importance` | Ordinal | Importancia del producto (low/medium/high) |
| `Gender` | Categórica | Género del cliente |
| `Discount_offered` | Numérica | Descuento ofrecido |
| `Weight_in_gms` | Numérica | Peso del producto en gramos |
| `Reached.on.Time_Y.N` | Binaria | ¿Entregado a tiempo? (target clasificación) |

---

## Instrucciones para Reproducir

### 1. Requisitos previos

Abre la terminal en VS Code (Ctrl+Ñ o Terminal → New Terminal) y ejecuta:

pip install pandas numpy matplotlib seaborn scikit-learn kagglehub jupyter


### 2. Ejecutar el notebook

1. Instala la extensión Jupyter desde el panel de extensiones (Ctrl+Shift+X)
2. Abre el archivo Copia_de_Copia_de_Caso_Estudio1_6.ipynb desde el explorador de archivos
3. Selecciona el kernel de Python en la esquina superior derecha → elige tu entorno con las dependencias instaladas
4. Ejecuta todas las celdas con el botón Run All (▶▶) o Ctrl+Alt+R

El flujo del notebook es:

1. Descarga y carga del dataset
2. Exploración (EDA): distribuciones, boxplots, matriz de correlación
3. Preprocesamiento: OrdinalEncoder para Product_importance, OneHotEncoder para variables nominales, eliminación de ID
4. Estandarización con StandardScaler
5. Entrenamiento y evaluación de modelos de regresión (Linear, Ridge, Lasso)
6. Entrenamiento y evaluación de modelos de clasificación (Árbol de Decisión, Random Forest)
---

## Resultados Principales

### Regresión — Predicción de `Cost_of_the_Product`

| Modelo | MAE | RMSE | R² |
|---|---|---|---|
| Regresión Lineal | 0.7684 | 0.9271 | 0.1043 |
| Ridge (α=1.0) | 0.7684 | 0.9271 | — |
| Lasso (α=1.0) | 0.8366 | 0.9797 | — |

> Los tres modelos presentan desempeño similar y bajo R² (~10%), lo que indica que las variables disponibles no tienen suficiente poder explicativo para predecir el costo del producto. El problema no es el modelo sino las features del dataset.

### Clasificación — Predicción de `Reached.on.Time_Y.N`

| Modelo | Accuracy |
|---|---|
| Árbol de Decisión | 64.1% |
| Random Forest | — |

**Árbol de Decisión — Reporte de Clasificación:**

| Clase | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| 0 (No a tiempo) | 0.56 | 0.53 | 0.54 | 887 |
| 1 (A tiempo) | 0.69 | 0.72 | 0.70 | 1,313 |

> El modelo clasifica mejor los pedidos entregados a tiempo (clase 1) que los no entregados. Se recomienda aplicar técnicas de balanceo o ajuste de umbrales para mejorar el recall de la clase 0.

---

## Conclusiones

- La **regresión lineal** con o sin regularización (Ridge/Lasso) no logra explicar adecuadamente el costo del producto con las variables disponibles.
- El **Árbol de Decisión** alcanza un 64% de accuracy en clasificación, con mejor desempeño en la clase mayoritaria (pedidos a tiempo).
- Se recomienda explorar **modelos de ensamble** (Random Forest, Gradient Boosting) y feature engineering para mejorar ambas tareas.