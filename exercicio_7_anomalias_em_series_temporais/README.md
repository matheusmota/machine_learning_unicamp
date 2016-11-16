# Exercício 7

**Data de entrega: 21/11, as 7:00 (da manha).**.

Este exercicio é sobre anomalias e series temporais, e é aberto. Eu vou propor um problema para voces mas eu nao sei a solucao, eu gostaria que voces explorassem suas ideias e intuicoes.

Nao há resposta certa ou errada, so ha alternativas que dao e nao dao mais ou menos certo e nao dar certo NAO é um problema - é a realidade de uma pesquisa/exploracao.

Queremos detectar regioes numa serie temporal que sao anomalas, obviamente sem uma definicao do que é anomalo!. As series estao abaixo.

![ts1](http://www.ic.unicamp.br/~wainer/cursos/2s2016/ml/high.png)
Serie temporal 1 [dados aqui](http://www.ic.unicamp.br/~wainer/cursos/2s2016/ml/serie1.csv)
![ts2](http://www.ic.unicamp.br/~wainer/cursos/2s2016/ml/mid.png)
Serie temporal 2 [dados aqui](http://www.ic.unicamp.br/~wainer/cursos/2s2016/ml/serie2.csv)
![ts3](http://www.ic.unicamp.br/~wainer/cursos/2s2016/ml/low.png)
Serie temporal 3 [dados aqui](http://www.ic.unicamp.br/~wainer/cursos/2s2016/ml/serie3.csv)
![ts4](http://www.ic.unicamp.br/~wainer/cursos/2s2016/ml/spikes.png)
Serie temporal 4 [dados aqui](http://www.ic.unicamp.br/~wainer/cursos/2s2016/ml/serie4.csv)
![ts5](http://www.ic.unicamp.br/~wainer/cursos/2s2016/ml/extra.png)
Serie temporal 5 [dados aqui](http://www.ic.unicamp.br/~wainer/cursos/2s2016/ml/serie5.csv)

Queremos detectar anomalias locais, ou seja, o trecho da serie temporal é anomala em relacao aos outros trechos, e nao o valor em particular que é anomalo (detectar valores anomalos funcionaria para a serie 1 mas nao para as outras).

A ideia central é definir um trecho da serie como N pontos seguidos, e representar cada trecho. Eu acho que apenas a media e desvio padrao dos valores no trecho, ou media, maximo e minimo sera suficiente. Assim, cada trecho sera representado por 2 (media e desvio padrao) ou 3 (media maximo minimo) dimensões.

Duas questoes a serem tratadas sao qual o valor de N e quanto de interssecao entre os trechos.

Com a representacao de cada trecho, o problema é encotrar o/os trechos anomalos. E aqui as varias alternativas discutidas em classe sao aplicaveis: distancia ao k-esimo vizinho, dados em subespacos diferentes, dados em regiao de baixa densidade, etc.

Discuta a sua solução para o problema, e mostre o resultado do seu algoritimo nas series temporais de 1 a 4. Pense numa forma de mostrar os trechos anonalos no plot das series temporais. Finalmente rode o seu algoritimo na serie temporal 5 (onde nao é nada claro o que é anomalo e nao anaomalo) e mostre e discuta os resultados. 
