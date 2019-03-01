import Orange

base = Orange.data.Table('census.csv')
base.domain

base_divida = Orange.evaluation.testing.sample(base, n=0.25)
base_teste = base_divida[0]
base_treinamento = base_divida[1]

cn2_learner = Orange.classification.rules.CN2Learner()
classificador = cn2_learner(base_treinamento)

for regras in classificador.rule_list:
    print(regras)
    
resultado = Orange.evaluation.testing.TestOnTestData(base_treinamento, base_teste, [classificador])
print(Orange.evaluation.CA(resultado))
