

import numpy as np
import os, json, glob, re, datetime
import importlib

class Np_Encoder(json.JSONEncoder):
    """
    要轉存成json，numpy型別必須先轉檔。
    同理，要使用sklearn時，部分需要用到numpy型別。
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(Np_Encoder, self).default(obj)

    @staticmethod
    def inverse( obj):
        if isinstance(obj, int):
            return np.int64(obj)
        if isinstance(obj, float):
            return np.float64(obj)
        if isinstance(obj, list):
            return np.array(obj)
        return obj

def model_type(model, return_meta = False):
    """
    現在能處理三種estimators: DataFrameMapper, pipeline, 一般transformer/predictor
    """
    module_name = model.__class__.__module__
    qualname = model.__class__.__qualname__
    if qualname == 'DataFrameMapper':
        # type = mapper:
        # 此時meta會是columns name，會套用在 features, built_features上
        meta = [i[0] for i in model.__dict__['features']]
        type = 'mapper'
    elif re.match("(?=.*pipeline).*", module_name.lower()):
        meta = [i[0] for i in model.__dict__['steps']]
        type = 'pipeline'
    else:
        meta, type = '', ''
    if return_meta:
        return type, meta
    return type

# ================================================================== #
# save 相關
# ================================================================== #
def creat_estimator_folder(save_to, estimator_name, replace = True):
    """
    建立資料夾: dir = save_to/estimator_name
    如果replace = True，建立新資料夾覆蓋掉舊的；反之則建立一個新的資料夾: dir_建立時間
    """
    path = os.path.join(save_to, estimator_name)
    if estimator_name not in os.listdir(save_to):
        os.mkdir(path)
        return path
    if estimator_name in os.listdir(save_to) and not replace:
        path += "_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        os.mkdir(path)
        return path
    return path

def save_model(model,
               save_to: str = r"D:\新安\Dev\iuse_prop\file\train\pp",
               filename: str = "model.json",):
    """
    單純的儲存"一個model"到指定位置。
    """
    module_name = model.__class__.__module__
    qualname = model.__class__.__qualname__
    parameters = model.__dict__
    path = os.path.join(save_to, filename)
    with open(path, "w") as f:
        to_save = dict(module = module_name,
                       qualname = qualname,
                       parameters = parameters)
        f.write(json.dumps(to_save, cls=Np_Encoder))

def save_setting(model,
                 save_to: str = r"D:\新安\Dev\iuse_prop\file\train\pp",
                 filename: str = "setting.json",
                 do_not_save = False):
    def _is_jsonable(x):
        try:
            json.dumps(x, cls = Np_Encoder)
            return True
        except:
            return False

    module_name = model.__class__.__module__
    qualname = model.__class__.__qualname__

    type, meta = model_type(model, return_meta = True)
    setting = {'meta': meta,
               'type': type,
               'module': module_name,
               'qualname': qualname,
               'unjsonable': [],
               'parameters': dict()}
    for k, v in model.__dict__.items():
        if _is_jsonable(v):
            setting["parameters"][k] = v
        else:
            setting["unjsonable"].append(k)
    if do_not_save:
        return setting
    else:
        with open(os.path.join(save_to, filename), 'w') as f:
            f.write(json.dumps(setting))

def save_estimator(model,
                   path: str = r"D:\新安\Dev\iuse_prop\file\train\pp",
                   estimator_name: str = 'estimator',
                   replace = True):
    """
    新建一個資料夾，把這個estimator裡的model都存成.json。
    如果像pipeline中的pipeline，就再開啟一次save_estimator
    """
    # 建立資料夾(如果estimator_name = ''，就不建立新資料夾)
    if estimator_name:
        path = creat_estimator_folder(path, estimator_name, replace)
    # 判定 estimator的類別，
    # 記下 estimator中的model
    # 如果是有sub_estimators的情況，儲存setting
    type = model_type(model)
    if type == 'mapper':
        save_setting(model, save_to = path, filename ="setting.json", do_not_save = False)
        models = [m[1] for m in model.built_features] # DataFrameMapper 的存在.built_features
    elif type == 'pipeline':
        save_setting(model, save_to = path, filename ="setting.json", do_not_save = False)
        models = [m[1] for m in model.steps] # DataFrameMapper 的存在.steps
    else:
        models = [model] # 沒有sub_estimators的情況
    # 儲存
    for m_id, m in enumerate(models):
        type = model_type(m)
        if type in ['mapper', 'pipeline']:
            # 若有sub_estimators，建立資料夾，把sub_estimator + setting放進去
            m_path = os.path.join(path, f"model_{m_id}")
            creat_estimator_folder(path, f"model_{m_id}", replace=True)
            save_estimator(m, m_path, estimator_name = '')
        else:
            save_model(m, save_to = path, filename = f"model_{m_id}.json")

# ================================================================== #
# read 相關
# ================================================================== #
def create_obj(module, qualname):
    module = importlib.import_module(module)
    return getattr(module, qualname)

def find_models(path: str):
    res = []
    for mp in glob.glob(os.path.join(path, "model*")):
        if os.path.isdir(mp):
             res.append(find_models(mp))
        if os.path.splitext(mp)[-1] == '.json':
            res.append(mp)
    return res

def open_model(path: str):
    """
    單純的load已訓練的estimator的json，並建立一個對應的estimator。
    path指向一個.json
    """
    if not((f := os.path.splitext(path)[-1]) == '.json'):
        f = 'floder (or empty)' if f == '' else f
        raise ValueError(f"need a .json file. not a {f}.")
    with open(path, 'r') as f:
        model_setting = json.load(f)
    obj = create_obj(model_setting["module"], model_setting["qualname"])
    parameters = {k: Np_Encoder.inverse(v) for k, v in model_setting["parameters"].items()}
    # init specific model
    model_load = obj()
    model_load.__dict__ = parameters
    return model_load

def open_setting(path: str):
    """
    回傳一個dict:
    {obj: 目標estimator(還沒實例化),
     type: mapper, pipeline, 其他,
     unjsonable: 不能寫進json的__dict__屬性，通常包含estimator,
     parameters: 實例化的時候要update進__dict__的屬性
     }
    """
    with open(path, 'r') as f:
        setting = json.load(f)
    # print(f"WARNING: {setting['unjsonable']} should be add to __dict__.")
    module = importlib.import_module(setting["module"])
    obj = getattr(module, setting["qualname"])
    return {"obj": obj,
            "meta": setting["meta"],
            "type": setting["type"],
            "unjsonable": setting["unjsonable"],
            "parameters": setting["parameters"]}


def open_estimator(path: str = r"D:\新安\Dev\iuse_prop\file\train\pp\estimator"):
    """
    讀取一個儲存好的estimator
    """
    if "setting.json" in os.listdir(path):
        setting = open_setting(os.path.join(path, "setting.json"))
        meta = setting["meta"]
        model_paths = glob.glob(os.path.join(path, "model*"))
        models = [open_model(mp) if mp.endswith(".json") else open_estimator(mp) for mp in model_paths]
        if setting["type"] == 'pipeline':
            steps = [(alias, step) for alias, step in zip(meta, models)]
            obj = setting["obj"](steps = steps)
            obj.__dict__.update(setting["parameters"])
        if setting["type"] == 'mapper':
            features = [(input, m) for input, m in zip(meta, models)]
            built_features = [(input, m, {}) for input, m in zip(meta, models)]
            obj = setting["obj"](features = features)
            obj.__dict__.update(setting["parameters"])
            obj.__dict__["built_features"] = built_features
        return obj
    else:
        model_paths = glob.glob(os.path.join(path, "model*"))
        models = [open_model(mp) for mp in model_paths]
        if len(models) > 1:
            print("WARNING: found mutiple model in path without setting.json.")
            return models
        return models[0]

if __name__ == '__main__':
    # ======================================================================== #
    # common, single, not nested estimator
    # ======================================================================== #
    from sklearn.linear_model import LinearRegression
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.dot(X, np.array([1, 2])) + 3
    reg = LinearRegression()
    reg.fit(X, y)
    # save
    save_to = r"D:\新安\Dev\iuse_prop\file\train\pp"
    save_estimator(reg, path = save_to, estimator_name = 'reg')
    # read
    reg_ = open_estimator(os.path.join(save_to, "reg"))
    # 測試結果是否相同
    (reg_.predict(X) == reg.predict(X)).all()

    # ======================================================================== #
    # DataFrameMapper
    # ======================================================================== #
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
    save_to = r"D:\新安\Dev\iuse_prop\file\train\pp"
    save_estimator(mapper, path = save_to, estimator_name = 'mapper')
    # read
    mapper_ = open_estimator(os.path.join(save_to, "mapper"))
    # 測試結果是否相同
    (mapper.transform(tax.head(100)) == mapper_.transform(tax.head(100))).all()

    # ======================================================================== #
    # Pipeline
    # ======================================================================== #
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
    # 測試結果是否相同
    (pipe.predict(X_test) == pipe_.predict(X_test)).all()


    # ======================================================================== #
    # DataFrameMapper + Pipeline
    # ======================================================================== #
    from sklearn.preprocessing import StandardScaler, MaxAbsScaler, MultiLabelBinarizer
    # 含有pipeline的mapper
    mapper_complicate = DataFrameMapper([
                                (["tx_capital"], [StandardScaler(), MaxAbsScaler()]),
                                ("tx_ind_item", MultiLabelBinarizer())
                              ])
    mapper_complicate.fit(tax)
    # save
    save_to = r"D:\新安\Dev\iuse_prop\file\train\pp"
    save_estimator(mapper_complicate, path = save_to, estimator_name = 'mapper_complicate')
    # read
    mapper_ = open_estimator(os.path.join(save_to, "mapper_complicate"))
    # 測試結果是否相同
    (mapper_.transform(tax.head(100)) == mapper_complicate.transform(tax.head(100))).all()
