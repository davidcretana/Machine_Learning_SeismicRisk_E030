# Pipeline Híbrido de Machine Learning Físicamente Informado (Norma E.030)

Se implementó una arquitectura de **Machine Learning de Dos Etapas** que integra principios de ingeniería estructural con algoritmos avanzados de ensamble (**Gradient Boosting** y **Random Forest**) sobre la base de datos `dataset_e030_v2.csv` (2,500 edificaciones sintéticas peruanas).

---

## 1. Arquitectura del Modelo
1. **Etapa 1 (Regresión de Demandas Físicas)**: Estima simultáneamente la Cortante Basal ($V_x, V_y$) y la Deriva Inelástica Máxima ($\Delta$) a partir de las propiedades geométricas y de sitio ($X$).
2. **Etapa 2 (Clasificación de Peligrosidad)**: Combina las entradas iniciales $X$ con las demandas predichas ($\hat{V}_x, \hat{V}_y, \hat{\Delta}$) para clasificar el nivel de peligro sísmico (`Bajo`, `Medio`, `Alto`, `Critico`).

---

## 2. Métricas de Rendimiento en Prueba

### A. Regresión de Demandas (Etapa 1)
* **Cortante Basal $V_x$**: $R^2 = 0.9199$ | RMSE = 281.60 Ton
* **Cortante Basal $V_y$**: $R^2 = 0.9224$
* **Deriva Inelástica $\Delta$**: $R^2 = 0.9223$ | RMSE = 0.00050

### B. Clasificación de Peligrosidad Sísmica (Etapa 2)
* **Exactitud Global (Accuracy)**: **87.40%**
* **F1-Score Ponderado**: **0.8741**

---

## 3. Matriz de Confusión (Conteo en Muestra de Prueba)

| Real \ Predicho | Alto | Bajo | Critico | Medio |
| :--- | :---: | :---: | :---: | :---: |
| **Alto** | 118 | 6 | 5 | 6 |
| **Bajo** | 1 | 205 | 0 | 16 |
| **Critico** | 8 | 0 | 18 | 2 |
| **Medio** | 12 | 5 | 2 | 96 |

---

## 4. Top Variables Dominantes (*Feature Importance*)
1. **`Pred_Deriva`**: 33.96%
2. **`Factor_R_X`**: 18.31%
3. **`Factor_Cx`**: 10.16%
4. **`Irregularidad_Planta_Ip`**: 7.59%
5. **`Irregularidad_Altura_Ia`**: 7.07%

---
*Desarrollado para el proyecto de IA en Ingeniería Estructural (UPC).*
