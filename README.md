# Machine_Learning_SeismicRisk_E030

![Build Status](https://img.shields.io/badge/Status-Completed-success)
![Framework](https://img.shields.io/badge/Framework-Physics--Informed%20ML-blue)
![Code Standard](https://img.shields.io/badge/Norma-Peruvian%20E.030%20RNE-red)

---

## 🌐 English Version

### 1. Executive Summary & Overview
This repository implements a **Physics-Informed Machine Learning (PIML) framework** designed to forecast dynamic seismic demands and classify structural vulnerability in reinforced concrete and masonry buildings across Peru. Built strictly in accordance with the **Peruvian Seismic Design Code (*Norma Técnica E.030 - Reglamento Nacional de Edificaciones*)**, the project bridges empirical computational mechanics and statistical learning.

By deploying a **two-stage supervised architecture** coupled with **unsupervised dimensional discovery (PCA & K-Means)**, this model bypasses the heavy computational latency of traditional non-linear dynamic time-history and pushover analyses. It allows structural engineers, urban planners, and risk managers to evaluate building safety in real time during conceptual design or city-scale vulnerability assessments.

---

### 2. Parametric Dataset Architecture (`dataset_e030_v2.csv`)
A synthetic dataset of **2,500 building configurations** was parametrically generated using closed-form structural mechanics and spectral formulations derived from E.030. Each sample consists of **37 explicit variables**:

* **Seismic Hazard & Site Parameters**: Zone Factor ($Z \in \{0.10, 0.25, 0.35, 0.45\}g$), Soil Profile ($S_0, S_1, S_2, S_3$), Soil Factor ($S$), Platform Periods ($T_p, T_L$).
* **Occupancy & Importance**: Category Factor ($U \in \{1.0, 1.3, 1.5\}$).
* **Directional Structural System & Reduction Factors**: Basic reduction factors ($R_{0,X}, R_{0,Y}$ for RC Frames, Dual Systems, Structural Walls, Confined Masonry), Height Irregularity ($I_a$), Plan Irregularity ($I_p$), and Effective Response Modification Factors ($R_X = R_{0,X} \cdot I_a \cdot I_p$, $R_Y = R_{0,Y} \cdot I_a \cdot I_p$).
* **Geometric & Physical Attributes**: Number of stories ($N \in [2, 20]$), total height ($H$), footprint area ($A$), effective seismic weight ($P$), shear wall area ratio.
* **Dynamic & Spectral Demand Outputs**: Directional fundamental periods ($T_X, T_Y$), spectral acceleration coefficients ($C_X, C_Y$), pseudo-accelerations ($S_{a,X}, S_{a,Y}$), directional base shears ($V_X, V_Y$), shear-to-weight ratios ($V/P$), maximum inelastic inter-story drifts ($\Delta_X$), and normative drift limits ($\Delta_{\text{lim}}$).
* **Target Label**: Multiclass Hazard Index (`Low`, `Medium`, `High`, `Critical`).

---

### 3. Machine Learning Methodology & Techniques

#### 3.1. Unsupervised Discovery (PCA & K-Means)
To explore latent physical relationships without human bias:
* **Principal Component Analysis (PCA)**: Reduced the feature space from 37 variables to 5 principal components explaining **71.55% of total cumulative variance**. 
  * **PC1 (28.58%)**: Governed by structural scale, mass, and flexibility ($H$, $N$, $P$, $T_X, T_Y$).
  * **PC2 (18.15%)**: Governed by site acceleration demand, spectral response, and drift ratios ($S_a, Z, S, \Delta$).
* **K-Means Clustering ($K=4$)**: Identified natural cluster boundaries in the reduced PCA space that closely aligned with normative risk boundaries.

#### 3.2. Two-Stage Physics-Informed Supervised Pipeline
To preserve physical consistency ($V = \frac{Z \cdot U \cdot C \cdot S}{R} \cdot P$) while leveraging non-linear feature interactions:
* **Stage 1 — Demand Regression (Gradient Boosting / Random Forest)**: Predicts continuous physical responses ($V_X, V_Y, \Delta$).
  * **Directional Base Shear ($V_X$)**: $R^2 = 0.925$, $\text{RMSE} = 281.60\text{ Ton}$.
  * **Inelastic Drift ($\Delta$)**: $R^2 = 0.922$, $\text{RMSE} = 0.00050$.
* **Stage 2 — Hazard Classification (Ensemble Learning)**: Combines initial building parameters with Stage 1 predicted physical demands ($\hat{V}_X, \hat{V}_Y, \hat{\Delta}$) to predict the categorical hazard index.
  * **Global Classification Accuracy**: **87.40%** (Macro F1-Score: **0.83**, Weighted F1-Score: **0.87**).

---

### 4. Technical Stack & Tools
* **Programming Environment**: Python 3.12, Jupyter Notebook.
* **Data Processing & Analytics**: Pandas, NumPy.
* **Machine Learning Libraries**: Scikit-Learn (PCA, K-Means, GradientBoostingRegressor, RandomForestClassifier, StandardScaler).
* **Visualization Engine**: Matplotlib, Seaborn (Publication-grade dashboards at 150 DPI).
* **Methodological Framework**: Veridical Data Science (VDS) — Predictability, Computability, Stability (PCS).

---

### 5. Practical Engineering Applications
1. **Rapid Seismic Screening & Triage**: Rapidly evaluates thousands of existing or proposed buildings across coastal and Andean zones in Peru without generating full ETABS/SAP2000 models.
2. **Early Conceptual Design Optimization**: Enables structural designers to test variations in shear wall ratios, structural systems ($R_0$), or floor plans ($I_p$) in milliseconds, receiving immediate feedback on drift compliance and base shear demands.
3. **Urban Risk & Catastrophe Modeling**: Serves municipal authorities and insurance companies for regional seismic loss estimation and emergency response planning.

---

## 🇪🇸 Versión en Español

### 1. Resumen Ejecutivo y Visión General
Este repositorio implementa un **marco de Aprendizaje Automático Físicamente Informado (PIML)** diseñado para predecir demandas sísmicas dinámicas y clasificar la vulnerabilidad estructural en edificaciones de concreto armado y albañilería en el Perú. Desarrollado estrictamente bajo la **Norma Técnica E.030 de Diseño Sismorresistente del Reglamento Nacional de Edificaciones (RNE)**, el proyecto une la mecánica computacional con el aprendizaje estadístico.

Mediante una **arquitectura supervisada de dos etapas** combinada con **descubrimiento no supervisado de dimensionalidad (PCA y K-Means)**, este modelo evita la alta latencia computacional de los análisis dinámicos no lineales (tiempo-historia o pushover). Permite a ingenieros estructurales y gestores de riesgo evaluar la seguridad estructural en tiempo real durante la fase conceptual de diseño o en evaluaciones de vulnerabilidad a escala urbana.

---

### 2. Arquitectura del Dataset Paramétrico (`dataset_e030_v2.csv`)
Se generó paramétricamente una base de datos sintética de **2,500 configuraciones estructurales** utilizando ecuaciones de mecánica estructural y formulaciones espectrales de la E.030. Cada muestra incluye **37 variables explícitas**:

* **Peligro Sísmico y Sitio**: Factor de Zona ($Z \in \{0.10, 0.25, 0.35, 0.45\}g$), Perfil de Suelo ($S_0, S_1, S_2, S_3$), Factor de Suelo ($S$), Periodos de Plataforma ($T_p, T_L$).
* **Uso e Importancia**: Factor de Categoría ($U \in \{1.0, 1.3, 1.5\}$).
* **Sistema Estructural y Reducción Direccional**: Coeficientes básicos ($R_{0,X}, R_{0,Y}$ para Pórticos de Concreto, Sistemas Duales, Muros Estructurales, Albañilería Confinada), Irregularidad en Altura ($I_a$), Irregularidad en Planta ($I_p$), y Coeficientes Efectivos de Reducción ($R_X = R_{0,X} \cdot I_a \cdot I_p$, $R_Y = R_{0,Y} \cdot I_a \cdot I_p$).
* **Atributos Geométricos y Físicos**: Número de pisos ($N \in [2, 20]$), altura total ($H$), área en planta ($A$), peso sísmico efectivo ($P$), ratio de área de muros de corte.
* **Respuestas Dinámicas y Espectrales**: Periodos fundamentales por dirección ($T_X, T_Y$), factor de amplificación sísmica ($C_X, C_Y$), pseudoaceleraciones ($S_{a,X}, S_{a,Y}$), cortantes basales ($V_X, V_Y$), ratio cortante/peso ($V/P$), derivas inelásticas máximas ($\Delta_X$) y límites normativos ($\Delta_{\text{lim}}$).
* **Etiqueta Target**: Índice Multiclase de Peligrosidad (`Bajo`, `Medio`, `Alto`, `Critico`).

---

### 3. Metodología y Técnicas de Machine Learning

#### 3.1. Descubrimiento No Supervisado (PCA y K-Means)
* **Análisis de Componentes Principales (PCA)**: Redujo el espacio de características de 37 variables a 5 componentes principales que explican el **71.55% de la varianza acumulada total**.
  * **PC1 (28.58%)**: Dominado por la escala física, masa y flexibilidad ($H$, $N$, $P$, $T_X, T_Y$).
  * **PC2 (18.15%)**: Dominado por la demanda de aceleración, respuesta espectral y derivas ($S_a, Z, S, \Delta$).
* **Clustering K-Means ($K=4$)**: Identificó agrupaciones naturales en el espacio reducido de PCA con alta concordancia respecto a los límites de riesgo normativo.

#### 3.2. Pipeline Supervisado Físicamente Informado (Dos Etapas)
* **Etapa 1 — Regresión de Demandas (Gradient Boosting / Random Forest)**: Predice variables continuas de respuesta física ($V_X, V_Y, \Delta$).
  * **Cortante Basal Direccional ($V_X$)**: $R^2 = 0.925$, $\text{RMSE} = 281.60\text{ Ton}$.
  * **Deriva Inelástica ($\Delta$)**: $R^2 = 0.922$, $\text{RMSE} = 0.00050$.
* **Etapa 2 — Clasificación de Peligrosidad (Ensemble Learning)**: Combina las entradas iniciales con las demandas físicas predichas en la Etapa 1 ($\hat{V}_X, \hat{V}_Y, \hat{\Delta}$) para clasificar el nivel de peligro.
  * **Precisión Global de Clasificación**: **87.40%** (F1-Score Macro: **0.83**, F1-Score Ponderado: **0.87**).

---

### 4. Herramientas y Stack Tecnológico
* **Lenguaje y Entorno**: Python 3.12, Jupyter Notebook.
* **Procesamiento de Datos**: Pandas, NumPy.
* **Bibliotecas de Machine Learning**: Scikit-Learn (PCA, K-Means, GradientBoostingRegressor, RandomForestClassifier, StandardScaler).
* **Visualización de Datos**: Matplotlib, Seaborn (Dashboards de calidad de publicación a 150 DPI).
* **Marco Metodológico**: Veridical Data Science (VDS) — Predicción, Computabilidad, Estabilidad (PCS).

---

### 5. Aplicaciones Prácticas en Ingeniería Estructural
1. **Tamizado Sísmico Rápido (Triage)**: Evalúa rápidamente miles de edificios existentes o proyectados en la costa y sierra del Perú sin necesidad de crear modelos complejos en ETABS o SAP2000.
2. **Optimización en Pre-Diseño Estructural**: Permite a proyectistas probar variaciones en densidad de muros, sistemas estructurales ($R_0$) o irregularidades ($I_p$) en milisegundos, obteniendo retroalimentación inmediata sobre cumplimiento de derivas.
3. **Gestión del Riesgo Urbano**: Sirve a municipalidades y aseguradoras para la estimación de pérdidas sísmicas a nivel catastral y planes de respuesta ante emergencias.

---

### 📂 Repository Structure / Estructura del Repositorio
```text
Machine_Learning_SeismicRisk_E030/
├── README.md                           # Documentación principal (Bilingüe)
├── dataset_e030_v2.csv                 # Dataset sintético paramétrico (37 variables, 2500 casos)
├── entrenamiento_supervisado_e030-v2.py# Script de entrenamiento supervisado en Python
├── pca_clustering_e030-v2.py           # Script de análisis no supervisado (PCA + K-Means)
├── supervisado_ml_dashboard-v2.png     # Dashboard gráfico de desempeño del modelo supervisado
├── pca_clustering_dashboard-v2.png     # Dashboard gráfico de análisis no supervisado
├── reporte_supervisado_ml-v2.md        # Reporte técnico cuantitativo del modelo supervisado
└── reporte_pca_clustering-v2.md        # Reporte técnico del análisis no supervisado
```
