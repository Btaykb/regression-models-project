from sklearn.discriminant_analysis import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score

df = fetch_california_housing(as_frame=True)
X = df.data
y = df.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), X.columns)
])

param_grid = {
    'svr__C': [0.1, 1, 10, 100],
    'svr__gamma': ['scale', 0.01, 0.1, 1],
    'svr__epsilon': [0.01, 0.1, 0.2]
}

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('svr', SVR(kernel='rbf')),
])

grid_search = GridSearchCV(pipeline, param_grid, cv=3,
                           n_jobs=-1, scoring='r2', verbose=2)
grid_search.fit(X_train, y_train)

print("Best params:", grid_search.best_params_)
print("Best CV score:", grid_search.best_score_)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
score = r2_score(y_test, y_pred)
print(score)
