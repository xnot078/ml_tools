# %%
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import FastICA, PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer


from typing import Literal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.offline as pyo
import os

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

    def __init__(self, method: Literal['PCA', 'ICA'] = 'PCA', drop_ratio: float = .01, thresh: float = None, return_label_only=False, **kwargs):
        self.kwargs = kwargs
        self.method = method
        self.drop_ratio = drop_ratio
        self.thresh = thresh
        self.return_label_only = return_label_only
        assert self.method in ['PCA', 'ICA']
        ValueError("method must be 'PCA' or 'ICA' either.")

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.method == 'PCA':
            estimator = PCA(**self.kwargs)
        else:
            estimator = FastICA(**self.kwargs)
        X_de = estimator.fit_transform(X)
        X_inverse = estimator.inverse_transform(X_de)
        SSE = np.sum((X - X_inverse)**2, axis=1)
        SSE_scaled = (SSE - min(SSE)) / (max(SSE) - min(SSE))
        self.scores = SSE_scaled

        # 如果thresh有提供，>= thresh的視為離群; 反之，SSE_scaled 前drop_ratio%的視為離群值
        if isinstance(self.thresh, float) and self.thresh <= 1 and self.thresh > 0:
            mask = SSE_scaled >= self.thresh
        else:
            self.thresh = np.quantile(SSE_scaled, 1 - self.drop_ratio)
            mask = SSE_scaled >= self.thresh
        # 如果只回傳離群值
        if self.return_label_only:
            return mask

        return X[~mask]


def to_2d(X, return_pipe=False):
    """
    如果要做X_test的轉換，記得要return_pipe = True
    """
    pipe = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),
                           ('scaler', MinMaxScaler()),
                           ('pca', PCA(n_components=2)), ])
    if return_pipe:
        return pipe.fit_transform(X), pipe
    return pipe.fit_transform(X)


def cluster_plot(X_2d, labels, ax=None, use_plotly=False):
    if not use_plotly:
        markers = pd.DataFrame(X_2d)
        markers['label'] = labels
        if ax is None:
            fig, ax = plt.subplots()
        for i, (k, g) in enumerate(markers.groupby('label')):
            color = plt.cm.tab10(i)
            ax.plot(g[0], g[1], '.', color=color, label=k)

    if use_plotly:
        fig_data = go.Scatter(
            x=X_2d[:, 0], y=X_2d[:, 1], mode='markers', marker={'color': labels})
        fig = go.Figure(fig_data)
        pyo.plot(fig, auto_open=False)

def cluster_plot_featDist(X, labels):
    """
    plot kde of each feature in one subplot. the left-top is 2D scatter plot of X which is hue by labels.
    """
    num_subplots = len(X.columns) + 1
    num_rows = int(np.ceil(num_subplots / 3))
    fig, axs = plt.subplots(num_rows, 3, figsize=(4 * 3, 4 * num_rows))
    # 各cluster的樣本數
    cluster_feq = pd.Series(labels).value_counts()
    p1 = axs[0, 0].barh(cluster_feq.index, cluster_feq.values)
    axs[0, 0].bar_label(p1)
    axs[0, 0].grid()
    # 投影成2維之後的2d散佈
    X_2d = to_2d(X)
    cluster_plot(X_2d, labels, ax=axs[0, 1])
    axs[0, 1].grid()
    # 各群組的各個特徵分布
    X_des = X.copy()
    X_des['label'] = labels
    X_des = X_des.fillna(0)
    for i, feat in enumerate(X.columns, start=2):
        r, c = i // 3, i % 3
        sns.kdeplot(data=X_des[[feat, 'label']],
                    x=feat,
                    hue='label',
                    fill=True,
                    linewidth=2,
                    alpha=.2,
                    palette='coolwarm',
                    ax=axs[r, c],
                    warn_singular=False)
        axs[r, c].grid()
    return fig


# %%
def cluster_feat_radar(X, labels, file_name=None, aggfunc='mean', title=''):
    """
    plot radar of each feature which is aggregated by aggfunc. 
    """
    # 缺失值先補0 (不然平均時nan會跳過)
    X_des = X.copy().fillna(0)
    X_des['label'] = labels
    # 為clusters計算agg
    cluster_agg = X_des.groupby('label').agg(aggfunc)
    cluster_agg_std = (cluster_agg - cluster_agg.min()) / \
        (cluster_agg.max() - cluster_agg.min())
    cluster_agg_std = cluster_agg_std.fillna(0)

    holder = []
    showlegend = True
    for c in cluster_agg.T:
        vals = cluster_agg.loc[c]
        angle = cluster_agg_std.loc[c]
        holder.append(
            go.Scatterpolar(
                r=angle,
                theta=vals.index,
                name=f'cluster {c}',
                text=vals,
                line={"dash": "solid", "width": 2},
                # marker = {'size': 16, 'color': Cmap_Marker.get(landing)},
                fill="toself",
                opacity=.9,
                showlegend=True,
                hovertemplate="%{theta}: %{text}"
            )
        )
        # showlegend = False
    fig = go.Figure(holder)
    fig.update_layout(title={'text': title, 'x': .5, 'font': {'size': 16}})

    if file_name is not None:
        pyo.plot(fig, filename=file_name, auto_open=False)
        static_name = os.path.splitext(file_name)[0]
        fig.write_image(static_name + '.png')
    pyo.plot(fig, filename='temp_radar.html', auto_open=False)
    fig.write_image('temp_radar.png')


# %%
if __name__ == '__main__':
    from sklearn.datasets import make_circles, make_blobs

    X0 = make_blobs(centers=[[0, 0]], n_features=2,
                    n_samples=1000, cluster_std=.25)[0]
    X1 = make_circles()[0]
    X = np.vstack([X0, X1])

    X_2d = to_2d(X)

    df = pd.DataFrame(X)
    ae = AnomalyExtractor(method='ICA', drop_ratio=.15,
                          n_components=2, return_label_only=True)
    label_ae = ae.transform(df)
    df['label_ae'] = label_ae
    print(sum(label_ae))
    print(df.groupby(label_ae).mean())

    cluster_plot(X_2d, label_ae)
