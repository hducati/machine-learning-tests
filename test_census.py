import pandas as pd

base = pd.read_csv('files/census.csv')

previsores = base.iloc[:, 0:14].values
classe = base.iloc[:, 14].values

from sklearn.preprocessing import LabelEncoder
previsores_labelencoder = LabelEncoder()
previsores[:, 1] = previsores_labelencoder.fit_transform(previsores[:, 1])
previsores[:, 3] = previsores_labelencoder.fit_transform(previsores[:, 3])

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)
