import pandas as pd

def main ():
    base = pd.read_csv('files/census.csv')
    previsores = base.iloc[:, 0:14].values
    classe = base.iloc[:, 14].values
    
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    previsores_encoder = LabelEncoder()
    previsores_list = [1, 3, 5, 6, 7, 8, 9, 13]
    for x in previsores_list:
        previsores[:, x] = previsores_encoder.fit_transform(previsores[:, x])
    previsores_onehotencoder = OneHotEncoder(categorical_features=[1,3,5,6,7,8,9,13])
    previsores = previsores_onehotencoder.fit_transform(previsores).toarray()
    # treating missing values
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer()
    imputer = imputer.fit(previsores[:, 0:14])
    previsores[:, 0:14] = imputer.transform(previsores[:, 0:14])
    
    # scaling values
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    previsores = scaler.fit_transform(previsores)rc
    
    # training and test data
    from sklearn.model_selection import train_test_split
    previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.25, random_state=0)
    
    # applying logistic regression
    from sklearn.linear_model import LogisticRegression
    classificador = LogisticRegression(random_state=0)
    classificador.fit(previsores_treinamento, classe_treinamento)
    # predict values
    previsoes = classificador.predict(previsores_teste)
    
    # check accuracy
    from sklearn.metrics import accuracy_score, confusion_matrix
    precisao = accuracy_score(classe_teste, previsoes)
    matriz = confusion_matrix(classe_teste, previsoes)
    print(precisao)
    print(matriz)
    
if __name__ == '__main__':
    main()