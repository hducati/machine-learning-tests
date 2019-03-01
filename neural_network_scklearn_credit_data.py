import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


def main():
    data = pd.read_csv('credit-data.csv')
    data.loc[data.age < 0, 'age'] = 40.92
    previsores = data.iloc[:, 1:4].values
    classe = data.iloc[:, 4].values
    
    imputer = SimpleImputer()
    imputer = imputer.fit(previsores[:, 1:4])
    previsores[:, 1:4] = imputer.transform(previsores[:, 1:4])
    scaler = StandardScaler()
    previsores = scaler.fit_transform(previsores)
    previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(
            previsores, classe, test_size=0.25, random_state=1)
    # max_iters = maxima interacao
    # tol = tolerancia
    # solver = algoritmo utilizado para otimizacao dos pesos
    # hidden_layer_sizes = qt de camadas ocultas
    # activation = funcao de ativacao para a camada oculta
    network = MLPClassifier(verbose=True,
                            max_iter=1000,
                            tol=0.0000000010)
    network.fit(previsores_treinamento, classe_treinamento)
    prevision = network.predict(previsores_teste)
    
    accuracy = accuracy_score(classe_teste, prevision)
    matrix = confusion_matrix(classe_teste, prevision)
    print(accuracy)
    print(matrix)


if __name__ == '__main__':
    main()