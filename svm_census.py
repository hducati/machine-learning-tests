import pandas as pd

def main():
    base = pd.read_csv('files/census.csv')
    previsores = base.iloc[:, 0:14].values
    classe = base.iloc[:, 14].values
    
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
    encoder= LabelEncoder()
    previsores_list = [1, 3, 5, 6, 7, 8, 9, 13]
    for x in previsores_list:
        previsores[:, x] = encoder.fit_transform(previsores[:, x])
        
    onehotencoder = OneHotEncoder(categorical_features=previsores_list)
    previsores = onehotencoder.fit_transform(previsores).toarray()
    
    scaler = StandardScaler()
    previsores = scaler.fit_transform(previsores)
    
    from sklearn.model_selection import train_test_split
    previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.25, random_state=1)
    
    from sklearn.svm import SVC
    classificador = SVC(kernel='linear', random_state=1)
    classificador.fit(previsores_treinamento, classe_treinamento)
    previsoes = classificador.predict(previsores_teste)
    
    from sklearn.metrics import accuracy_score, confusion_matrix
    precisao = accuracy_score(classe_teste, previsoes)
    matriz = confusion_matrix(classe_teste, previsoes)
    print('Accuracy: ' + str(precisao))
    print(matriz)
    
if __name__ == '__main__':
    main()