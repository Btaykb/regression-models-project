from sklearn.discriminant_analysis import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score
from data_preprocessing import load_data, encode_categorical

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
score = r2_score(y_test, y_pred)
print(score)
