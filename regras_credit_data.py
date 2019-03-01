import Orange

base = Orange.data.Table('credit-data.csv')
base.domain

# n=porcentagem a ser dividida
base_dividida = Orange.evaluation.testing.sample(base, n=0.25)
base_teste = base_dividida[0]
base_treinamento = base_dividida[1]

len(base_treinamento)
len(base_teste)

cn2_learner = Orange.classification.rules.CN2Learner()
# where the 'learn' happens
classificador = cn2_learner(base_treinamento)

for regras in classificador.rule_list:
    print(regras)

resultado = Orange.evaluation.testing.TestOnTestData(base_treinamento, base_teste, [classificador])
# test accuracy
print(Orange.evaluation.CA(resultado))
