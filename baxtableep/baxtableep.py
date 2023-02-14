import numpy as np
import joblib

class Baxtableep:
	def __init__(self) -> None:
		self.clf = joblib.load('data/model.joblib')
		self.vec = joblib.load('data/vec.joblib')

	def predict(self, texts: list[str]) -> list[int]:
		X_new = self.vec.transform(texts)
		y_pred = self.clf.predict(X_new)
		return y_pred
	
	def predict_probs(self, texts: list[str]) -> np.ndarray:
		X_new = self.vec.transform(texts)
		y_pred = self.clf.predict_proba(X_new)
		return np.apply_along_axis(self._get_profane_prob, 1, y_pred)

	def _get_profane_prob(self, prob):
  		return prob[1]

if __name__ == '__main__':
	baxtableep = Baxtableep()
	texts = ['Offensive Text', 'Non-offensive Text']
	print(baxtableep.predict_probs(texts))
	print(baxtableep.predict(texts))