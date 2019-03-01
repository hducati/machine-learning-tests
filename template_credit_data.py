import pandas as pd

base = pd.read_csv('files/credit-data.csv')
base.loc[base.age < 0] = 40.92

previsores = base.iloc[:, 1:4].values
classe = base.iloc[:, 4].values

from sklearn.impute import SimpleImputer

imputer = SimpleImputer()
imputer = imputer.fit(previsores[:, 1:4])
previsores = imputer.transform(previsores[:, 1:4])

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

from sklearn.model_selection import train_test_split

previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.25, random_state=0)
