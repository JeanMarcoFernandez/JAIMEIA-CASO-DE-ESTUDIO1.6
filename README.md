# Caso de Estudio 6 — Análisis Predictivo de Entregas E-Commerce

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
| `Cost_of_the_Product` | Numérica | **Target regresión** — costo del producto (96–310) |
| `Prior_purchases` | Numérica | Compras previas del cliente |
| `Product_importance` | Ordinal | Importancia: low (5,297) / medium (4,754) / high (948) |
| `Gender` | Categórica | Género del cliente *(eliminada: bajo poder explicativo)* |
| `Discount_offered` | Numérica | Descuento ofrecido |
| `Weight_in_gms` | Numérica | Peso del producto en gramos |
| `Reached.on.Time_Y.N` | Binaria | **Target clasificación** — 0: llegó a tiempo (4,436 / 40.3%), 1: retraso (6,563 / 59.7%) |

---

## Marco Metodológico: CRISP-DM

| Fase | Aplicación en este Proyecto |
|---|---|
| 1. Business Understanding | Definición del problema logístico, objetivos de regresión, clasificación y análisis |
| 2. Data Understanding | EDA: distribución del target, correlaciones, análisis de variables clave |
| 3. Data Preparation | Limpieza, codificación, ingeniería de características |
| 4. Modeling | Regresión Lineal, Árbol de Decisión, Random Forest |
| 5. Evaluation | Métricas MAE, RMSE, R², Accuracy, Precision, Recall, F1-Score, matrices de confusión |
| 6. Deployment | Recomendaciones estratégicas para implementación productiva |

---

## Instrucciones para Ejecutar

### 1. Acceder a Google Colab

1. Ve a [Google Colab](https://colab.research.google.com)
2. Carga el archivo: `Caso de estudio - 6 JaimeIA.ipynb`
3. Ejecuta todas las celdas con **Runtime → Run all** (`Ctrl+F9`)

No se requiere instalar dependencias: pandas, numpy, matplotlib, seaborn y scikit-learn ya están preinstalados en Colab.

---

## Resultados

### Regresión — Predicción de `Cost_of_the_Product`

**División de datos:** 80% entrenamiento (8,799 muestras) — 20% prueba (2,200 muestras)

| Métrica | Valor |
|---|---|
| **MSE** | 1961.77 |
| **RMSE** | 44.29 |
| **R² Score** | 0.11 |

**Interpretación:** El modelo explica apenas el 11% de la variabilidad en el costo del producto. Esta limitación es informativa: el dataset carece de variables directamente relacionadas con la formación del precio (categoría de producto, marca, tipo). El error promedio de predicción es de ±44.29 unidades monetarias.

---

### Clasificación — Predicción de `Reached.on.Time_Y.N`

**División de datos:** 80% entrenamiento (8,799 muestras) — 20% prueba (2,200 muestras)

#### Árbol de Decisión

**Accuracy: 68%**

| Clase | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| 0 (A tiempo) | 0.57 | 0.91 | 0.70 | 895 |
| 1 (Retraso) | 0.90 | 0.52 | 0.66 | 1,305 |
| **Weighted avg** | **0.76** | **0.68** | **0.67** | **2,200** |

**Análisis:** El modelo tiene un recall excelente (0.91) para detectar entregas a tiempo, pero una precisión moderada (0.57) cuando predice retrasos. Esto significa que identifica correctamente la mayoría de entregas puntuales, pero comete errores al predecir retrasos.

---

#### Random Forest

**Accuracy: 68%**

| Clase | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| 0 (A tiempo) | 0.57 | 0.91 | 0.70 | 895 |
| 1 (Retraso) | 0.90 | 0.52 | 0.66 | 1,305 |
| **Weighted avg** | **0.76** | **0.68** | **0.67** | **2,200** |

**AUC Score: 0.75**

**Análisis:** El Random Forest mantiene un desempeño similar al Árbol de Decisión en términos de accuracy, pero ofrece mayor robustez. El AUC de 0.75 indica una buena capacidad discriminativa entre clases.

---

## Comparación de Modelos de Clasificación

| Métrica | Árbol de Decisión | Random Forest |
|---|---|---|
| Accuracy | 68% | 68% |
| Precision (clase 1) | 0.90 | 0.90 |
| Recall (clase 1) | 0.52 | 0.52 |
| F1-Score (clase 1) | 0.66 | 0.66 |
| AUC | — | 0.75 |

Ambos modelos presentan rendimiento similar. El Random Forest proporciona mayor estabilidad y mejor generalización, mientras que el Árbol de Decisión es más interpretable.

---

## Decisiones de Preprocesamiento

### Variables eliminadas
- **`ID`**: Identificador sin valor predictivo.
- **`Gender`**: Sin correlación relevante con los targets.

### Codificación
- **`Product_importance`** → `OrdinalEncoder` con orden explícito (`low=0, medium=1, high=2`), capturando la jerarquía de negocio.
- **`Warehouse_block`** y **`Mode_of_Shipment`** → `OneHotEncoder`, evitando jerarquías falsas.

### Calidad de datos
- Valores nulos: **ninguno encontrado**
- Duplicados: **dataset limpio**
- Outliers en `Prior_purchases`: **conservados** por coherencia comercial

---

## Conclusiones

El mayor valor del proyecto reside en el **modelo de clasificación**, que permite transitar de una gestión logística reactiva a una **predictiva y proactivo**:

- Los modelos de clasificación (Árbol de Decisión y Random Forest) alcanzan 68% de accuracy, permitiendo identificar entregas a tiempo con alto recall (0.91).
- El modelo de **regresión no se recomienda para producción** en el estado actual (R² = 0.11); se requieren variables comerciales adicionales.
- Ambos clasificadores ofrecen perspectivas **complementarias** para optimizar operaciones logísticas.

### Recomendaciones estratégicas

- Incorporar **feature engineering** adicional (interacciones entre variables, categoría de producto).
- Probar **Gradient Boosting** (XGBoost, LightGBM) para ambas tareas.
- Aplicar **ajuste de umbral de decisión** en clasificación para balancear recall según prioridad operativa.
- Para regresión, explorar modelos no lineales (Random Forest Regressor) con variables de precio más directas.

---

## Estructura del Proyecto

```
Caso de estudio - 6 JaimeIA.ipynb    ← Notebook principal (ejecutar en Colab)
README.md                             ← Este archivo
```

## Tecnologías

- **Google Colab** (entorno de ejecución)
- **Pandas** (manipulación de datos)
- **NumPy** (operaciones numéricas)
- **Matplotlib & Seaborn** (visualización)
- **Scikit-learn** (modelado ML)
