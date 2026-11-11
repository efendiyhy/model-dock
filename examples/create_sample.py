"""Optional reproducible example to create a pre-trained iris model (not a platform training feature)."""
from pathlib import Path
import joblib
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=400, random_state=42))
model.fit(X, y)
output = Path(__file__).with_name('iris_classifier.joblib')
joblib.dump(model, output)
print(f'Saved {output}')
