# sklearn2json

將sklearn的estimator儲存成json格式。

目前能處理:
sklearn
pipeline (含steps)
DataFrameMapper

**範例: single, not nested estimator**
```python
import sklearn2json as s2j
from sklearn.linear_model import LinearRegression
# create a model and train it.
X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
y = np.dot(X, np.array([1, 2])) + 3
reg = LinearRegression()
reg.fit(X, y)
# save
save_to = "./test" # check floder ./test exists first.
save_estimator(reg, path = save_to, estimator_name = 'reg')
# read
reg_ = open_estimator(os.path.join(save_to, "reg"))
# check
print((reg_.predict(X) == reg.predict(X)).all())
```

**範例: [sklearn_pandas.DataFrameMapper](https://github.com/scikit-learn-contrib/sklearn-pandas)**
```python
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
import pandas as pd

tax = pd.DataFrame([[3000, ['469921', '952312']],
                     [50000, ['773099']],
                     [50000, ['561113']],
                     [45000, ['561113']],
                     [1000000, ['454999']],
                     [100000, ['484311', '465311', '951199']],
                     [100000, ['563111']],
                     [50000, ['464999', '484311', '959999']],
                     [3000, ['473312']],
                     [3000, ['474912', '474213', '434013']]],
                     columns = ['tx_capital', 'tx_ind_item'])
mapper = DataFrameMapper([
                            (["tx_capital"], StandardScaler()),
                            ("tx_ind_item", MultiLabelBinarizer())
                          ])
mapper.fit(tax)
# save
save_to = "./test"
save_estimator(mapper, path = save_to, estimator_name = 'mapper')
# read
mapper_ = open_estimator(os.path.join(save_to, "mapper"))
# check
(mapper.transform(tax.head(100)) == mapper_.transform(tax.head(100))).all()
```

**範例: [sklearn.pipeline.Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)**
```python
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
X, y = make_classification(random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    random_state=0)
pipe = Pipeline([("scaler", StandardScaler()), ("svc", SVC(C=1.0))])
# The pipeline can be used as any other estimator
# and avoids leaking the test set into the train set
pipe.fit(X_train, y_train)
# save
save_to = r"D:\新安\Dev\iuse_prop\file\train\pp"
save_estimator(pipe, path = save_to, estimator_name = 'pipe')
# read
pipe_ = open_estimator(os.path.join(save_to, "pipe"))
# check
(pipe.predict(X_test) == pipe_.predict(X_test)).all()
```
