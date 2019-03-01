import pandas as pd
import keras
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense


def main():
    data = pd.read_csv('files/credit-data.csv')
    previsores = data.iloc[:, 1:4].values
    classe = data.iloc[:, 4].values
    
    imputer = SimpleImputer()
    imputer = imputer.fit(previsores[:, 1:4])
    previsores[:, 1:4] = imputer.transform(previsores[:, 1:4])
    scaler = StandardScaler()
    previsores = scaler.fit_transform(scaler)
    
    previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(
            previsores, classe, test_size=0.25, random_state=1)
    
    classifier = Sequential()
    # units= quantos neuronios vão existir na camada oculta
    classifier.add(Dense(units=2, activation='relu', input_dim=3))
    classifier.add(Dense(2, 'relu'))
    classifier.add(Dense(1, 'sigmoid'))
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # batch_size = atualiza os pesos a cada X registros
    classifier.fit(previsores_treinamento, classe_treinamento, batch_size=10, nb_epoch=100)
    prevision = classifier.predict(previsores_teste)
    
    accuracy = accuracy_score(classe_teste, prevision)
    matrix = confusion_matrix(classe_teste, prevision)    
    print(accuracy)
    print(matrix)
