# Data Url

url = r".\data.csv"

# Importing Libraries  

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import ElasticNetCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import neighbors
from sklearn.svm import SVR
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from warnings import filterwarnings
filterwarnings('ignore')  

# Exploring Data

df = pd.read_csv(url)
df.dtypes
df.head()
df.groupby("model").count()

# Editing Units
df.rename(columns={"yıl":"yil"})
df["fiyat"]= df["fiyat"]*1000
df["KM"] = df["KM"]*1000

# Data Visualization (Fiyat)

sns.boxplot(x=df["fiyat"])
plt.show()
df.describe()

Q1 = df["fiyat"].quantile(0.25)    # Değişkeni küçükten büyüğe sıralıyoruz ve %25. veriye ulaşıyoruz.
Q3 = df["fiyat"].quantile(0.75)    # Değişkeni küçükten büyüğe sıralıyoruz ve %75. veriye ulaşıyoruz.
IQR = Q3-Q1     # Deftere not alınan formülü uyguluyoruz.

alt_sinir = Q1-1.5*IQR
ust_sinir = Q3+1.5*IQR

aykiri_min_fiyat = df["fiyat"] < alt_sinir  
aykiri_maks_fiyat = df["fiyat"] > ust_sinir

df["fiyat"][aykiri_maks_fiyat] = alt_sinir
df["fiyat"][aykiri_min_fiyat] = ust_sinir

# Eşik Değer Belirlemesi (KM)

Q1 = df["KM"].quantile(0.25)    # Değişkeni küçükten büyüğe sıralıyoruz ve %25. veriye ulaşıyoruz.
Q3 = df["KM"].quantile(0.75)    # Değişkeni küçükten büyüğe sıralıyoruz ve %75. veriye ulaşıyoruz.
IQR = Q3-Q1     # Deftere not alınan formülü uyguluyoruz.

alt_sinir = Q1-1.5*IQR
ust_sinir = Q3+1.5*IQR

aykiri_min_fiyat = df["KM"] < alt_sinir  
aykiri_maks_fiyat = df["KM"] > ust_sinir

df["KM"][aykiri_maks_fiyat] = alt_sinir
df["KM"][aykiri_min_fiyat] = ust_sinir

# Dropping unnecessary features

df=df.drop(["ilan_tarihi","ilçe"],axis=1)


# Editing Classes 

df["yeni_il"] = LabelEncoder().fit_transform(df["il"])
df["yeni_model"] = LabelEncoder().fit_transform(df["model"])
df[["il","yeni_il"]].head(10)
df[["model","yeni_model"]].groupby("model").mean()
# Dropping Non-Numeric Features

df=df.drop(["model","il"],axis=1)
df=df.astype("int64")

# column name editing for for lightgbm
df.columns = ["".join (c if c.isalnum() else "_" for c in str(x)) for x in df.columns]
# Train Data and Test Data
y = df["fiyat"]
X = df.drop("fiyat",axis=1)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20)

# Models List

models = [
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
    KNeighborsRegressor,
    SVR,
    MLPRegressor,
    DecisionTreeRegressor,
    RandomForestRegressor,
    GradientBoostingRegressor,
    XGBRegressor,
    ]

# Modelling Function

def modelling(alg):
            
    # Modelling

    model = alg().fit(X_train,y_train)
    y_pred = model.predict(X_test)
    RMSE = np.sqrt(mean_squared_error(y_pred,y_test))
    model_ismi = alg.__name__
    print(f"Model Ismi: {model_ismi} ve hatasi: {RMSE}")


for i in models:
    modelling(i)

# En iyi modeli seçin ve devam edin

gbm_model = GradientBoostingRegressor().fit(X_train,y_train)
y_pred = gbm_model.predict(X_test)
np.sqrt(mean_squared_error(y_pred,y_test))

# Model Tuning

gbm_params = {
    "learning_rate":[0.001,0.1,0.01],
    "max_depth":[3,5,8],
    "n_estimators":[100,200,500],
    "subsample":[1,0.5,0.8],
    "loss":["ls","lad","quantile"]}

gbm_cv_model = GridSearchCV(gbm_model, gbm_params, n_jobs=-1,verbose=1,cv=5).fit(X_train,y_train)
gbm_cv_model.best_params_["learning_rate"]

# Final Model

gbm_tuned = GradientBoostingRegressor(
    learning_rate=gbm_cv_model.best_params_["learning_rate"],
    loss=gbm_cv_model.best_params_["loss"],
    max_depth=gbm_cv_model.best_params_["max_depth"],
    n_estimators=gbm_cv_model.best_params_["n_estimators"],
    subsample=gbm_cv_model.best_params_["subsample"]
)

gbm_tuned
y_pred = gbm_tuned.fit(X_train,y_train).predict(X_test)
np.sqrt(mean_squared_error(y_pred,y_test))
mean_absolute_error(y_pred,y_test)

# Tahmin Edilmek İstenen [Araç, Yıl, KM, Kasa Tipi, İl Numarası]
gbm_tuned.predict([[2017,50000,6,20]])

# Değişken Önem Düzeyleri

Importance = pd.DataFrame({"Importance":gbm_tuned.feature_importances_*100},index=X_train.columns)
Importance.sort_values(
    by="Importance",
    axis=0,
    ascending=True).plot(kind="barh",color="r")

plt.xlabel("Variable Importance")
plt.gca().legend_=None
plt.show()

