import pandas as pd

base = pd.read_csv('files/credit-data.csv')
base.loc[base.age < 0, 'age'] = 40.92

previsores = base.iloc[:, 1:4].values
classe = base.iloc[:, 4].values

from sklearn.impute import SimpleImputer

imputer = SimpleImputer()
imputer = imputer.fit(previsores[:, 1:4])
previsores[:, 1:4] = imputer.transform(previsores[:, 1:4])

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

from sklearn.model_selection import train_test_split

previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.25, random_state=0)

from sklearn.ensemble import RandomForestClassifier

# n_estimators=numbers of forests
classificador = RandomForestClassifier(n_estimators=40, criterion='entropy', random_state=0)
classificador.fit(previsores_treinamento, classe_treinamento)
previsoes = classificador.predict(previsores_teste)

from sklearn.metrics import confusion_matrix, accuracy_score

# comparando a integridade dos valores, retorna a porcentagem de acertos
precisao = accuracy_score(classe_teste, previsoes)
# verifica a quantidade de acertos
matriz = confusion_matrix(classe_teste, previsoes)


import collections
# base line(zeroR) classifier counter
# if it is below classe_teste]/total of entries          use base line classifier
# else dont use
collections.Counter(classe_teste)