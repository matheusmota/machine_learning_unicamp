# Exercício 2

**Data de entrega: 5/10, as 7:00 (da manha)**.

Leia os dados do arquivo [data1.csv](http://www.ic.unicamp.br/~wainer/cursos/2s2016/ml/data1.csv) A classe de cada dado é o valor da última coluna (0 ou 1).

Treine um SVM com kernel RBF nos dados do arquivos.

A validação externa deve ser 5-fold estratificado.

Para cada conjunto de treino da validação externa faça um 3-fold para escolher os melhores hiperparametros para C (cost) e gamma.

Faça um grid search de para o C nos valores 2^-5, 2^-2, 2^0, 2^2, e 2^5 e gamma nos valores 2^-15, 2^-10, 2^-5, 2^0, e 2^5

1. Qual a accuracia media (na validação de fora).
2. Quais os valores de C e gamma a serem usados no classificador final (fazer o 3-fold no conjunto todo).

**NÃO** use funções prontas que ja fazer o grid search como *GridSearchCV* do sklearn ou o *tune* do pacote *e1070* do R. Neste exercicio eu quero que voces façam os loops explicitamente.

Gere um pdf com o código (R ou Python) e as respostas as perguntas. O exercicio deverá ser submetido via [Moodle](https://www.ggte.unicamp.br/ea/).

## Detalhes R

K-fold estratificado em R é feito pela função [createFolds](https://www.rdocumentation.org/packages/caret/versions/6.0-71/topics/createDataPartition) do pacotecaret

a função [svm](https://www.rdocumentation.org/packages/e1071/versions/1.6-7/topics/svm) do pacote *e1070* implementa o SVM. O default é o SVM RBF (kernel="radial")

## Detalhes em Python

Stratified k-fold em sklear é a função [StratifiedKFold](http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedKFold.html), >p> SVM é implementado por [SVC](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
