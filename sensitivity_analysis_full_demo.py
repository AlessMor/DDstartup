# sensitivity_analysis_full_demo.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_regression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.cluster import KMeans

import shap
import plotly.express as px

# ----------------------------
# 1. Dummy dataset
# ----------------------------
X, y = make_regression(n_samples=1000, n_features=6, noise=0.3, random_state=42)
columns = [f"feature_{i}" for i in range(X.shape[1])]
X = pd.DataFrame(X, columns=columns)
y = pd.Series(y, name="target")

# ----------------------------
# 2. Fit surrogate model
# ----------------------------
model = HistGradientBoostingRegressor(random_state=0)
model.fit(X, y)

# ----------------------------
# 3. Permutation importance
# ----------------------------
res = permutation_importance(model, X, y, n_repeats=10, random_state=0, n_jobs=-1)
importances = pd.Series(res.importances_mean, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(6,4))
importances.plot.barh()
plt.title("Permutation Importance")
plt.xlabel("Mean importance (ΔR²)")
plt.tight_layout()
plt.show()

# ----------------------------
# 4. SHAP values (summary + dependence)
# ----------------------------
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

shap.summary_plot(shap_values, X)  # global feature impact

# Example dependence plot for top feature
top_feature = importances.index[0]
shap.dependence_plot(top_feature, shap_values, X)

# ----------------------------
# 5. Effect-size heatmap (per quartile)
# ----------------------------
df = X.copy()
df["quartile"] = pd.qcut(y, 4, labels=["Q1","Q2","Q3","Q4"])

def cohen_d(a, b):
    na, nb = len(a), len(b)
    pooled = np.sqrt(((a.std(ddof=1)**2)*(na-1)+(b.std(ddof=1)**2)*(nb-1))/(na+nb-2))
    return (a.mean() - b.mean())/pooled

effects = pd.DataFrame(index=X.columns, columns=["Q1","Q2","Q3","Q4"])
for label in ["Q1","Q2","Q3","Q4"]:
    mask = df["quartile"]==label
    for col in X.columns:
        effects.loc[col,label] = cohen_d(X.loc[mask,col], X.loc[~mask,col])
effects = effects.astype(float)

plt.figure(figsize=(6,4))
sns.heatmap(effects, center=0, cmap="vlag", annot=True, fmt=".2f")
plt.title("Effect Size (Cohen's d) per Quartile")
plt.tight_layout()
plt.show()

# ----------------------------
# 7. Cluster analysis + quartile distribution
# ----------------------------
kmeans = KMeans(n_clusters=5, random_state=0).fit(X)
df['cluster'] = kmeans.labels_
pd.crosstab(df['cluster'], df['quartile'], normalize='index').plot.bar(stacked=True, figsize=(6,4))
plt.title("Quartile Distribution per Cluster")
plt.ylabel("Proportion")
plt.tight_layout()
plt.show()

# ----------------------------
# 8. SHAP interaction heatmap
# ----------------------------
shap_inter = explainer.shap_interaction_values(X)
S = np.abs(shap_inter).mean(axis=0)
plt.figure(figsize=(6,5))
sns.heatmap(S, xticklabels=X.columns, yticklabels=X.columns, cmap="viridis")
plt.title("Mean Absolute SHAP Interaction Values")
plt.tight_layout()
plt.show()

# ----------------------------
# 9. Partial dependence plot (PDP) for top 2 features
# ----------------------------
fig, ax = plt.subplots(figsize=(6,4))
PartialDependenceDisplay.from_estimator(model, X, [top_feature], ax=ax)
plt.show()

# 2D PDP for top two features
fig, ax = plt.subplots(figsize=(6,4))
PartialDependenceDisplay.from_estimator(model, X, [(top_feature, importances.index[1])], ax=ax)
plt.show()
