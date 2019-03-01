import pandas as pd

base = pd.read_csv('risco-credito2.csv')
previsores = base.iloc[:, 0:4].values
classe = base.iloc[:, 4].values

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
previsores[:, 0] = encoder.fit_transform(previsores[:, 0])
previsores[:, 1] = encoder.fit_transform(previsores[:, 1])
previsores[:, 2] = encoder.fit_transform(previsores[:, 2])
previsores[:, 3] = encoder.fit_transform(previsores[:, 3])

from sklearn.linear_model import LogisticRegression

classificador = LogisticRegression()
classificador.fit(previsores, classe)
# param b0
print(classificador.intercept_)
# return coef from each attr
print(classificador.coef_)

resultado = classificador.predict([[0,0,1,2], [3,0,0,0]])
resultado2 = classificador.predict_proba([[0,0,1,2], [3,0,0,0]])
print(resultado)
# prob from each class
# what it means?
# 0.18 * 100 - alto
# 0.81 * 100 - baixo
# 0.90 * 100 - alto
# 0.093 * 100 - baixo
print(resultado2)
