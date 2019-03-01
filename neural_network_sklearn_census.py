import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder 
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

def main():
    data = pd.read_csv('files/census.csv')
    previsores = data.iloc[:, 0:14].values
    classe = data.iloc[:, 14].values
    previsores_list = [1, 3, 5, 6, 7, 8, 9, 13]
    
    encoder = LabelEncoder()
    for x in previsores_list:
        previsores[:, x] = encoder.fit_transform(previsores[:, x])
    # classe[:, 14] = encoder.fit_transform(classe[:, 14])
    
    imputer = SimpleImputer()
    imputer = imputer.fit(previsores[:, 0:14])
    previsores[:, 0:14] = imputer.transform(previsores[:, 0:14])
    
    scaler = StandardScaler()
    previsores = scaler.fit_transform(previsores)
    
    previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(
            previsores, classe, test_size=0.25, random_state=1)
    classifier = MLPClassifier(max_iter=1000,
                               verbose=True,
                               tol=0.0000010)
    classifier.fit(previsores_treinamento, classe_treinamento)
    prevision = classifier.predict(previsores_teste)
    
    accuracy = accuracy_score(classe_teste, prevision)
    matrix = confusion_matrix(classe_teste, prevision)
    print(accuracy)
    print(matrix)
    
    
if __name__ == '__main__':
    main()