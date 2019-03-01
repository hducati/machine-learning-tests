import Orange

base = Orange.data.Table('credit-data.csv')
base.domain

base_divida = Orange.evaluation.testing.sample(base, n=0.25)
base_teste = base_divida[0]
base_treinamento = base_divida[1]

# base line classifier
classificador = Orange.classification.MajorityLearner()
resultado = Orange.evaluation.testing.TestOnTestData(base_treinamento, base_teste, [classificador])
print(Orange.evaluation.CA(resultado))

from collections import Counter

print(Counter(str(d.get_class()) for d in base_teste))