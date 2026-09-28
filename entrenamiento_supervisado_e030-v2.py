# ==============================================================================
# PROYECTO IA EN INGENIERÍA ESTRUCTURAL - UPC
# Código de Entrenamiento Supervisado Físicamente Informado
# Dataset: dataset_e030_v2.csv (Norma E.030 RNE)
# ==============================================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.metrics import r2_score, accuracy_score, classification_report, confusion_matrix

# 1. Cargar Dataset
df = pd.read_csv('dataset_e030_v2.csv')

# 2. Definir Features y Targets
feature_cols = [
    'Factor_Z', 'Factor_S', 'Periodo_Tp_s', 'Periodo_TL_s', 'Factor_Uso_U',
    'R0_X', 'R0_Y', 'Irregularidad_Altura_Ia', 'Irregularidad_Planta_Ip',
    'Factor_R_X', 'Factor_R_Y', 'N_Pisos', 'Altura_Total_H_m', 'Area_Planta_m2',
    'Peso_Total_P_Ton', 'Ratio_Area_Muros_pct', 'Periodo_Tx_s', 'Periodo_Ty_s',
    'Factor_Cx', 'Factor_Cy'
]

X = df[feature_cols]
y_vx = df['Cortante_Basal_Vx_Ton']
y_vy = df['Cortante_Basal_Vy_Ton']
y_deriva = df['Deriva_Inelastica_Max_X']
y_clf = df['Indice_Peligrosidad']

# 3. Split Entrenamiento / Prueba
X_tr, X_te, y_vx_tr, y_vx_te, y_vy_tr, y_vy_te, y_d_tr, y_d_te, y_c_tr, y_c_te = train_test_split(
    X, y_vx, y_vy, y_deriva, y_clf, test_size=0.2, random_state=42, stratify=y_clf
)

# 4. Etapa 1: Regresión de Demandas Físicas
reg_vx = GradientBoostingRegressor(n_estimators=150, random_state=42).fit(X_tr, y_vx_tr)
reg_vy = GradientBoostingRegressor(n_estimators=150, random_state=42).fit(X_tr, y_vy_tr)
reg_d = GradientBoostingRegressor(n_estimators=150, random_state=42).fit(X_tr, y_d_tr)

print(f"R2 Cortante Vx: {r2_score(y_vx_te, reg_vx.predict(X_te)):.4f}")
print(f"R2 Cortante Vy: {r2_score(y_vy_te, reg_vy.predict(X_te)):.4f}")
print(f"R2 Deriva Inelástica: {r2_score(y_d_te, reg_d.predict(X_te)):.4f}")

# 5. Etapa 2: Clasificación de Peligrosidad
X_tr_aug = X_tr.copy()
X_tr_aug['Pred_Vx'] = reg_vx.predict(X_tr)
X_tr_aug['Pred_Vy'] = reg_vy.predict(X_tr)
X_tr_aug['Pred_Deriva'] = reg_d.predict(X_tr)

X_te_aug = X_te.copy()
X_te_aug['Pred_Vx'] = reg_vx.predict(X_te)
X_te_aug['Pred_Vy'] = reg_vy.predict(X_te)
X_te_aug['Pred_Deriva'] = reg_d.predict(X_te)

clf = GradientBoostingClassifier(n_estimators=150, random_state=42).fit(X_tr_aug, y_c_tr)
pred_c = clf.predict(X_te_aug)

print(f"Accuracy Peligrosidad Sísmica: {accuracy_score(y_c_te, pred_c)*100:.2f}%")
print("\nReporte de Clasificación:")
print(classification_report(y_c_te, pred_c))
