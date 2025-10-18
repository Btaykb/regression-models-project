from pathlib import Path
from sklearn.discriminant_analysis import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score

from plot_utils import plot_pred_vs_actual

df = fetch_california_housing(as_frame=True)
X = df.data
y = df.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), X.columns)
])

model = RandomForestRegressor(n_estimators=100)

pipeline = Pipeline(steps=[
    ('regressor', model)
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

plots_dir = Path(__file__).resolve().parent / "result-plots"
plots_dir.mkdir(parents=True, exist_ok=True)
save_path = plots_dir / "random_forest_result_plot.png"
plot_pred_vs_actual(y_test, y_pred, str(save_path))
