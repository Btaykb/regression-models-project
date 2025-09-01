from sklearn.discriminant_analysis import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV

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

# Include model tuning

param_grid = {
    "regressor__n_estimators": [100, 200, 400, 500],
    "regressor__max_depth": [None, 10, 20, 30],
    "regressor__min_samples_split": [2, 5],
    "regressor__min_samples_leaf": [1, 2]
}

grid_search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_grid,
    cv=5,
    n_jobs=-1,
    n_iter=50,
    verbose=2
)

grid_search.fit(X_train, y_train)
y_pred_tuned = grid_search.predict(X_test)
score_tuned = r2_score(y_test, y_pred_tuned)
print(f"R2 Score for tuned model: {score_tuned}")
