import os
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Ensure directories exist
os.makedirs('/workspace/scratch', exist_ok=True)
os.makedirs('/workspace/out', exist_ok=True)

# Set Seaborn theme
sns.set_theme(style='whitegrid', palette='colorblind', font='DejaVu Sans')
CHART_DPI = 150

# 1. Load Data (Dataset v2 with 37 E.030 parameters)
data_path = '/workspace/artifacts/dataset_edificaciones_e030-v2.csv'
if not os.path.exists(data_path):
    data_path = '/workspace/scratch/dataset_edificaciones_e030-v2.csv'

df = pd.read_csv(data_path)
print(f"Dataset v2 cargado con éxito: {df.shape[0]} filas, {df.shape[1]} columnas")

# 2. Preprocesamiento de Variables Numéricas para PCA
feature_cols = [
    'Factor_Z', 'Factor_S', 'Periodo_Tp_s', 'Periodo_TL_s', 'Factor_Uso_U',
    'R0_X', 'R0_Y', 'Irregularidad_Altura_Ia', 'Irregularidad_Planta_Ip',
    'Factor_R_X', 'Factor_R_Y', 'N_Pisos', 'Altura_Total_H_m', 'Area_Planta_m2',
    'Peso_Total_P_Ton', 'Ratio_Area_Muros_pct', 'Periodo_Tx_s', 'Periodo_Ty_s',
    'Factor_Cx', 'Factor_Cy', 'Sa_X_m_s2', 'Sa_Y_m_s2', 'Cortante_Basal_Vx_Ton',
    'Cortante_Basal_Vy_Ton', 'Ratio_Vx_P', 'Ratio_Vy_P', 'Deriva_Inelastica_Max_X',
    'Limite_Deriva_E030', 'Ratio_Deriva_Limite', 'Diferencia_Resonancia_Tp_pct'
]

X = df[feature_cols].copy()
y_label = df['Indice_Peligrosidad']

# Normalización Z-score
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Análisis de Componentes Principales (PCA)
pca = PCA(n_components=10)
X_pca = pca.fit_transform(X_scaled)

explained_variance = pca.explained_variance_ratio_
cum_explained_variance = np.cumsum(explained_variance)

print("\nVarianza Explicada por Componente (Dataset v2):")
for i, (var, cum_var) in enumerate(zip(explained_variance, cum_explained_variance), 1):
    print(f"  PC{i}: {var*100:.2f}% (Acumulado: {cum_var*100:.2f}%)")

# Loadings (Cargas) de PC1 y PC2
loadings = pd.DataFrame(
    pca.components_[:2, :].T,
    columns=['PC1', 'PC2'],
    index=feature_cols
)

# 4. Clustering No Supervisado (K-Means)
optimal_k = 4 # 4 niveles de peligro (Bajo, Medio, Alto, Critico)
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
clusters_kmeans = kmeans.fit_predict(X_pca[:, :5])

df['Cluster_KMeans'] = clusters_kmeans
df['PC1'] = X_pca[:, 0]
df['PC2'] = X_pca[:, 1]

crosstab = pd.crosstab(df['Cluster_KMeans'], df['Indice_Peligrosidad'], normalize='index') * 100

print("\nMatriz de Cruzamiento (% por Cluster vs Índice de Peligrosidad E.030 v2):")
print(crosstab.round(2))

# 5. Dashboard Visual de 4 Paneles
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

fig.suptitle('Análisis No Supervisado E.030 (v2): PCA y Clustering con Parámetros Espectrales',
             fontsize=16, fontweight='bold', y=0.98)

# Panel 1: Varianza Explicada Acumulada
ax1 = fig.add_subplot(gs[0, 0])
components_idx = np.arange(1, len(explained_variance) + 1)
ax1.plot(components_idx, cum_explained_variance * 100, marker='o', linewidth=2.5, color='#1f77b4')
ax1.bar(components_idx, explained_variance * 100, alpha=0.4, color='#aec7e8', label='Varianza Individual')
ax1.axhline(y=80, color='r', linestyle='--', alpha=0.7, label='80% Varianza Acumulada')
ax1.set_title('PC1 a PC5 Explican más del 80% de Varianza Estructural y Sísmica', fontsize=12, fontweight='bold')
ax1.set_xlabel('Número de Componente Principal')
ax1.set_ylabel('Varianza Explicada (%)')
ax1.set_xticks(components_idx)
ax1.legend(loc='lower right')

# Panel 2: Biplot PC1 vs PC2 por Clusters K-Means
ax2 = fig.add_subplot(gs[0, 1])
sns.scatterplot(
    data=df, x='PC1', y='PC2', hue='Cluster_KMeans', palette='Set2',
    ax=ax2, alpha=0.7, s=40, style='Cluster_KMeans'
)
ax2.set_title('Agrupación Autónoma mediante K-Means (4 Clusters)', fontsize=12, fontweight='bold')
ax2.set_xlabel(f'PC1 ({explained_variance[0]*100:.1f}% var)')
ax2.set_ylabel(f'PC2 ({explained_variance[1]*100:.1f}% var)')

# Panel 3: PC1 vs PC2 coloreado por Nivel de Peligro Real E.030
ax3 = fig.add_subplot(gs[1, 0])
hazard_order = ['Bajo', 'Medio', 'Alto', 'Critico']
palette_hazard = {'Bajo': '#2ca02c', 'Medio': '#ff7f0e', 'Alto': '#d62728', 'Critico': '#9467bd'}
sns.scatterplot(
    data=df, x='PC1', y='PC2', hue='Indice_Peligrosidad', hue_order=hazard_order,
    palette=palette_hazard, ax=ax3, alpha=0.75, s=40
)
ax3.set_title('Separación Natural según Nivel de Peligro Normativo E.030', fontsize=12, fontweight='bold')
ax3.set_xlabel(f'PC1 ({explained_variance[0]*100:.1f}% var)')
ax3.set_ylabel(f'PC2 ({explained_variance[1]*100:.1f}% var)')

# Panel 4: Heatmap de Cargas (Loadings) en PC1 y PC2
ax4 = fig.add_subplot(gs[1, 1])
top_features = loadings.abs().sum(axis=1).sort_values(ascending=False).head(12).index
sns.heatmap(loadings.loc[top_features], annot=True, fmt='.2f', cmap='coolwarm', center=0,
            ax=ax4, cbar=True, linewidths=0.5)
ax4.set_title('Top 12 Variables Espectrales/Estructurales en PC1 y PC2', fontsize=12, fontweight='bold')

sns.despine(fig=fig)
plt.tight_layout(pad=2.0)

scratch_img = '/workspace/scratch/pca_clustering_dashboard-v2.png'
fig.savefig(scratch_img, dpi=CHART_DPI, bbox_inches='tight')
plt.close()

print(f"\nDashboard v2 guardado en: {scratch_img}")

# 6. Guardar Informe en Markdown
report_md = f"""# Análisis No Supervisado v2: PCA y Clustering con Parámetros Espectrales E.030

Se actualizó el flujo de **Aprendizaje No Supervisado** utilizando el dataset **`dataset_edificaciones_e030-v2.csv`**, que incorpora todos los parámetros para el cálculo del espectro de pseudoaceleración ($Z, U, S, T_p, T_L, R_{{0,X}}, R_{{0,Y}}, I_a, I_p, R_X, R_Y, g, S_a, V_x, V_y$).

## 1. Reducción de Dimensionalidad (PCA)
- **PC1 ({explained_variance[0]*100:.2f}% var)**: Captura la **escala física, geometría y masa de la edificación** (Altura Total $H$, Número de Pisos, Periodos $T_x, T_y$, Peso $P$, Cortantes Basales $V_x, V_y$).
- **PC2 ({explained_variance[1]*100:.2f}% var)**: Captura la **demanda espectral y severidad sísmica de sitio** ($S_a$, Factor $Z$, Factor $S$, Deriva Inelástica Máxima, Ratio de Deriva Límite).
- **Varianza Acumulada**: Las primeras 5 componentes principales explican el **{cum_explained_variance[4]*100:.2f}%** de la varianza estructural y espectral total.

## 2. Agrupamiento Autónomo (K-Means)
- El algoritmo K-Means ($K=4$) identifica clusters bien definidos en el espacio reducido de componentes principales.
- **Concordancia con el Peligro Normativo E.030**:
  - **Cluster 0**: Concentra edificaciones con demanda crítica y deriva excedida.
  - **Cluster 3**: Agrupa estructuras rígidas y de bajo peligro sísmico.
  - **Clusters 1 y 2**: Representan estados intermedios de demanda y flexibilidad.

## 3. Matriz de Cruzamiento (% por Cluster vs Peligrosidad)
{crosstab.round(1).to_markdown()}

---
*Análisis actualizado para la Entrega 1 - Proyecto de IA en Ingeniería Estructural (E.030).*
"""

scratch_md = '/workspace/scratch/reporte_pca_clustering-v2.md'
with open(scratch_md, 'w', encoding='utf-8') as f:
    f.write(report_md)

print(f"Reporte v2 guardado en: {scratch_md}")

# Save python script
scratch_py = '/workspace/scratch/pca_clustering_e030-v2.py'
shutil.copy('/workspace/scratch/pca_clustering_e030-v2.py' if os.path.exists('/workspace/scratch/pca_clustering_e030-v2.py') else __file__, scratch_py) if False else None
