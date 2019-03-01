import pandas as pd

def main():
    """
        Quando há a inconsistência dos dados,
        temos quatro tipos de soluções:
            
            -Apagar a coluna:
                base.drop('age', 1, inplace=True) 
            -Apagar apenas os registros com problemas:
                base.drop(base[base.age < 0].index, inplace=True)
            -Preencher os valores manualmente
            -Preencher os valores com a média
            
            Sintaxes usadas:
            inplace=True roda o comando na própria base da dados.
            .index=Apaga os valores do índice também.
            .mean=média dos valores
            .loc=localizar um determinado valor, sendo ele baseado em uma condição ou não
            .iloc=(params=quantidade de linhas, colunas)
            .fit=preencher os valores faltantes a partir da strategy definida(default=mean)
            .transform=transformar os valores gerados a partir do .fit para a variável em questão
            
        Tratamento de colunas com dados faltantes:
            from sklearn.impute import SimpleImputer
            
            imputer = SimpleImputer()
            imputer = imputer.fit(previsores[:, 0:3])
            previsores[:, 0:3] = imputer.transform(previsores[:, 0:3])
            
            Explicação:
                imputer = imputer.fit(Variavel[qt.linhas, quais colunas])
                Variavel[qt.linhas, quais colunas] = imputer.transform(Variavel[qt.linhas, colunas])
            
    """
    base = pd.read_csv('credit-data.csv')
    # localizando os clientes com idade negativa
    base.loc[base['age'] < 0]
    base.drop(base[base.age < 0].index, inplace=True)
    # média dos dados
    base.mean()
    # média de um dado específico
    base['age'].mean()
    # média sem os dados com problemas
    base['age'][base.age > 0].mean()
    # atribuindo a todos os valores inconsistentes o valor da média(40.92)
    base.loc[base.age < 0, 'age'] = 40.92
    # localiza os valores nulos
    pd.isnull(base['age'])
    base.loc[pd.isnull(base['age'])]
    
    # divindo o dataframe em duas variáveis, previsores e classe.
    previsores = base.iloc[:, 1:4].values
    classe = base.iloc[:, 4].values
    
    from sklearn.impute import SimpleImputer
    # aqui será realizado o tratamento dos dados faltantes
    imputer = SimpleImputer()
    imputer = imputer.fit(previsores[:, 0:3])
    previsores[:, 0:3] = imputer.transform(previsores[:, 0:3])
    
    # escalonamento dos dados
    # padronização
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    previsores = scaler.fit_transform(previsores)
    
if __name__ == "__main__":
    main()