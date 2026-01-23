# customer-churn
Las categorías numéricas no siguen una distribución normal
Las categorías binarias con datos "Yes" y "No" fueron transformadas a 0/1
Las categorías multi-clase fueron manejados con OneHotEncoder
Baseline con LogisticRegression, pero RandomForest es mejor modelo para churn por el mejor recall y AUC
No hay outliers

metricas:
accuracy: Porcentaje total de predicciones correctas (que tan bien clasifica en general)
roc-auc: Área bajo la curva ROC. Mide la capacidad del modelo para separar clases (0.5 significa modelo aleatorio, 1.0 separación perfecta)
f1-macro: Media del F1-score de cada clase. Mide el balance global del modelo. Si este valor es alto, el modelo no se centra solo en la clase mayoritaria.
precision-yes: De los que el modelo predice como churn, ¿cuántos lo son realmente?
recall-yes: De los que realmente son churn, ¿cuántos detecta el modelo?
f1-yes: Media entre precisión y recall para churn. Dice si el modelo está equilibrado para la clase importante.

Since churn is the minority class, I focused on recall and F1-score for the positive class, while using ROC-AUC to evaluate overall separability.