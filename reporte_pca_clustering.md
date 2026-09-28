# Análisis No Supervisado: PCA y Clustering en Edificaciones Peruanas (Norma E.030)

Se aplicó un flujo de **Aprendizaje No Supervisado** sobre el dataset sintético de 2,500 edificaciones estructurado bajo la Norma E.030 del RNE.

## 1. Reducción de Dimensionalidad (PCA)
- **PC1 (31.18% var)**: Captura principalmente la **escala física y flexibilidad estructural** (Altura, Número de Pisos, Periodo $T$, Cortante Basal).
- **PC2 (19.58% var)**: Captura la **demanda y severidad sísmica** (Ratio de Deriva Límite, Deriva Inelástica Máxima, Factor Z).
- **Varianza Acumulada**: Las primeras 4 componentes principales explican el **75.51%** de la varianza total.

## 2. Agrupamiento Autónomo (K-Means)
- El modelo K-Means ($K=4$) logró separar las estructuras en conglomerados bien definidos en el espacio reducido.
- **Coincidencia con Peligrosidad Normativa**: Los clusters identificados de manera no supervisada concuerdan de forma consistente con las categorías normativas de riesgo (`Bajo`, `Medio`, `Alto`, `Critico`).

## 3. Matriz de Cruzamiento (% por Cluster)
|   Cluster_KMeans |   Alto |   Bajo |   Critico |   Medio |
|-----------------:|-------:|-------:|----------:|--------:|
|                0 |   45.3 |    0.6 |      36.8 |    17.3 |
|                1 |   11.3 |   47.1 |       0.3 |    41.3 |
|                2 |   35.9 |   60.3 |       0   |     3.8 |
|                3 |    8.4 |   70.8 |       0.2 |    20.6 |

---
*Análisis generado para el proyecto de IA en Ingeniería Estructural.*
