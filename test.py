import joblib
import pandas as pd

clf = joblib.load('model.joblib')
vec = joblib.load('vec.joblib')

X_new = vec.transform(['Text1', 'Text2'])

y_pred = clf.predict(X_new)

print(y_pred)