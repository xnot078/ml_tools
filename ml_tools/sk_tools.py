# %%
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from sklearn.decomposition import FastICA, PCA
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.offline as pyo


class AnomalyExtractor(BaseEstimator, TransformerMixin):
	"""
	find anomaly samples and extract them.

	args:
		method: Literal['PCA', 'ICA']; choose a method to caculate origin-rebuilt SSE. default id PCA.
		drop_ratio: tops origin-rebuilt-SSE samples to drop in percent.
		thresh: if provide, drop samples with origin-rebuilt-SSE larger than thresh instead of use drop_ratio.
		return_label_only: if True, return anomaly mask only.
	retrun:
		(if return_label_only == False) samples without anomaly samples.
		(if return_label_only == True) bool array of anomaly samples, True means anomaly.

	"""
	def __init__(self, method:Literal['PCA', 'ICA'] = 'PCA', drop_ratio:float = .01, thresh:float = None, return_label_only = False, **kwargs):
		self.kwargs = kwargs
		self.method = method
		self.drop_ratio = drop_ratio
		self.thresh = thresh
		self.return_label_only = return_label_only
		assert self.method in ['PCA', 'ICA']; ValueError("method must be 'PCA' or 'ICA' either.")
	def fit(self, X, y = None):
		return self
	def transform(self, X):
		if self.method == 'PCA':
			estimator = PCA(**self.kwargs)
		else:
			estimator = FastICA(**self.kwargs)
		X_de = estimator.fit_transform(X)
		X_inverse = estimator.inverse_transform(X_de)
		SSE = np.sum((X - X_inverse)**2, axis = 1)
		SSE_scaled = (SSE - min(SSE)) / (max(SSE) - min(SSE))

		# 如果thresh有提供，>= thresh的視為離群; 反之，SSE_scaled 前drop_ratio%的視為離群值
		if isinstance(self.thresh, float) and self.thresh <=1 and self.thresh > 0:
			mask = SSE_scaled >= self.thresh
		else:
			self.thresh = np.quantile(SSE_scaled, 1-self.drop_ratio)
			mask = SSE_scaled >= self.thresh
		# 如果只回傳離群值
		if self.return_label_only:
			return mask

		return X[~mask]


def to_2d(X, return_pipe = False):
	"""
	如果要做X_test的轉換，記得要return_pipe = True
	"""
	pipe = Pipeline(steps = [('imputer', SimpleImputer(strategy = 'median')),
					   		 ('scaler', MinMaxScaler()),
					   		 ('pca', PCA(n_components = 2)),])
	if return_pipe:
		return pipe.fit_transform(X), pipe
	return pipe.fit_transform(X)


def cluster_plot(X_2d, labels, ax = None, use_plotly = False):
	if not use_plotly:
		markers = pd.DataFrame(X_2d)
		markers['label'] = labels
		if ax is None:
			fig, ax = plt.subplots()
		for i, (k, g) in enumerate(markers.groupby('label')):
			color = plt.cm.tab10(i)
			ax.plot(g[0], g[1], '.', color = color, label = k)

	if use_plotly:
		fig_data = go.Scatter(x = X_2d[:, 0], y = X_2d[:, 1], mode = 'markers', marker = {'color': labels})
		fig = go.Figure(fig_data)
		pyo.plot(fig, auto_open = False)

# %%
if __name__ == '__main__':
	from sklearn.datasets import make_circles, make_blobs
	import pandas as pd
	import matplotlib.pyplot as plt

	X0 = make_blobs(centers = [[0,0]], n_features=2, n_samples = 1000, cluster_std=.25)[0]
	X1 = make_circles()[0]
	X = np.vstack([X0, X1])

	X_2d = to_2d(X)

	df = pd.DataFrame(X)
	ae = AnomalyExtractor(method = 'ICA', drop_ratio = .15, n_components = 2, return_label_only = True)
	label_ae = ae.transform(df)
	df['label_ae'] = label_ae
	print(sum(label_ae))
	print(df.groupby(label_ae).mean())

	cluster_plot(X_2d, label_ae)

