import pandas as pd

base = pd.read_csv("files/census.csv")
previsores = base.iloc[:, 0:14].values
classe = base.iloc[:, 14].values

# transformacao de string para numerico
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

label_previsores = LabelEncoder()
# labels = label_previsores.fit_transform(previsores[:, 1])
previsores[:, 1] = label_previsores.fit_transform(previsores[:, 1])
previsores[:, 3] = label_previsores.fit_transform(previsores[:, 3])
previsores[:, 5] = label_previsores.fit_transform(previsores[:, 5])
previsores[:, 6] = label_previsores.fit_transform(previsores[:, 6])
previsores[:, 7] = label_previsores.fit_transform(previsores[:, 7])
previsores[:, 8] = label_previsores.fit_transform(previsores[:, 8])
previsores[:, 9] = label_previsores.fit_transform(previsores[:, 9])
previsores[:, 13] = label_previsores.fit_transform(previsores[:, 13])

onehotencoder = OneHotEncoder(categorical_features=[1, 3, 5, 6, 7, 8, 9, 13])
previsores = onehotencoder.fit_transform(previsores).toarray()

labelencoder_classe = LabelEncoder()
classe = labelencoder_classe.fit_transform(classe)

# aplicando o escalonamento para melhor performance
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

from sklearn.model_selection import train_test_split

previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.15, random_state=0)

from sklearn.naive_bayes import GaussianNB

classificador = GaussianNB()
classificador.fit(previsores_treinamento, classe_treinamento)
previsoes = classificador.predict(previsores_teste)

from sklearn.metrics import confusion_matrix, accuracy_score

precisao = accuracy_score(classe_teste, previsoes)
matriz = confusion_matrix(classe_teste, previsoes)