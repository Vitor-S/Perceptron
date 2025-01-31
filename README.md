
# Objetivo

Este repositório visa documentar o meu aprendizado em **deep learning** e como primeiro passo, vou descrever o meu entendimento sobre o que é um perceptron e como ele aprende, detalhando matematicamente o processo.

# Perceptron

Um perceptron é um dos modelos mais simples de rede neural artificial, desenvolvido na década de 1950 por Frank Rosenblatt. Ele é a unidade básica de um sistema de aprendizado de máquina e serve como bloco de construção para redes neurais mais complexas. 

O perceptron é um classificador binário, o que significa que ele pode separar dados em duas classes. Ele faz isso encontrando um hiperplano que divide o espaço de características em duas regiões e por isso o perceptron só consegue resolver problemas que são linearmente separáveis. Isso significa que, dado um conjunto de dados, o perceptron somente conseguirá classificar com sucesso estes dados casos eles possam ser separados completamente por uma linha reta (ou um hiperplano em gráficos com mais de duas dimensões).

Apesar de extremamente limitado, o perceptron é muito útil para introduzir os conceitos de **Machine Learning**.

# Como uma funciona uma rede neural ?

O perceptron é um modelo matemático que visa simular o funcionamento de um neurônio biológico, recebendo uma ou várias entradas, pesando seus valores e decidindo sua ativação.

![Exemplo de rede neural](https://i.imgur.com/Ejcxjag.png)

Frequentemente se soma um valor de ***viés*** (**bias** ou simplesmente **b**) à ponderação. Não entrarei em detalhes aqui sobre o motivo da introdução desse valor de viés, basta saber que o valor do viés ajuda a rede neural a se adaptar a situações onde a saída desejada não pode ser explicada apenas pelas entradas e pesos.

Por exemplo, se **x = 0** (y = 2 * 0 + 2), é impossível achar o valor **y = 2** sem a introdução do viés.

Para facilitar a didática implementarei o perceptron mais simples o possível:

![Rede neural C° para F°](https://i.imgur.com/m6AMVrd.png)

Este é um modelo para um perceptron que recebe uma temperatura em graus Celsius e converte para fahrenheit.

Após uma pesquisa rápida na web você pode constatar que, para converter uma temperatura de Celsius para Fahrenheit, basta multiplicá-la por **1.8** e somar com **32** (F = C * 1.8 + 32), ou seja, os valores perfeitos para que nosso perceptron possa converter com precisão valores em Celsius para valores em Fahrenheit é **w = 1.8** e **b = 32**

Vamos supor então que queremos converter **-10 C°** em fahrenheit, sendo assim:

![-10 C° para F°](https://i.imgur.com/zZXvkCs.png)

    F = -10 * 1.8 + 32
    F = -18 + 32
    F = 14

O aprendizado de uma rede neural consiste na exatamente na descoberta destes valores, os valores que vão permitir que a rede sempre ***"acerte a conta"***, neste caso: **w = 1.8** e **b = 32**. Então esta é a principal pergunta: **Como uma rede neural aprende os valores perfeitos ?**

# Como uma rede neural aprende ?

Primeiro, vamos observar o que acontece quando uma rede neural erra. Para isso, vamos inicializar **b** com o valor ideal (32) e w com um valor próximo ao ideal, nesse caso **1.6**.

![-10 C° para F° com w = 1.6](https://i.imgur.com/Wjc2tHD.png)

Nesse caso é possível observar que a rede errou a previsão em **2 unidades**. Para calcular o quanto uma rede neural errou em sua previsão pode usar a seguinte equação:

    erro = valor predito - valor esperado

Para evitar valores negativos é comum elevar o resultado ao quadrado:

    erro = (valor predito - valor esperado)²

Isso evita que, por exemplo, quando o valor predito for menor que o valor esperado o erro não seja menor do que zero:

    erro = (valor predito - valor esperado)²
    erro = (12 - 14)²
    erro = (-2)²
    erro = 4

Neste caso:

    erro = (16 - 14)²
    erro = 2²
    erro = 4

Colocando os valores em um gráfico, temos:

![Erro(w)](https://i.imgur.com/AsWtRf5.png)

Analisando o gráfico podemos observar que nosso objetivo é: **Achar o valor de w que faça o erro ser igual a 0**. Mas como faremos isso sem precisar calcular cada ponto da função ?

Vamos supor que a função tenha o seguinte formato:

![Erro(w) 2](https://i.imgur.com/SOF79ll.png)

Se tivéssemos ideia de para onde a função aumenta (esquerda do ponto (1.6, 2)) e para onde a função diminui (direita do ponto (1.6, 2)), mesmo que não soubéssemos o quanto temos que "caminhar" para que o erro seja igual a 0, saberemos para onde devemos caminhar, a partir disso, poderíamos dar pequenos passos até que o erro chegue a 0. Por exemplo, sabendo que temos que caminhar para a direita (sentido positivo), podemos dar um pequeno passo, digamos de 0.05 para a direita, logo, **w** que antes era 1.6 agora será 1.65 o que é mais próximo do valor ideal **1.8**, e o erro que antes era de **4 unidades** agora será **2,25** 

    erro = valor predito - valor esperado
    erro = ((-10 * 1.65 + 32) - 14)².
    erro = (15.5 - 14)²
    erro = 1,5²
    erro = 2.25

Mas agora temos outro problema: **Como saberemos em qual direção devemos caminhar sem ter que calcular todos os pontos do gráfico ?**

# Derivadas

Não nos aprofundaremos em derivadas aqui, basta entender que a derivada de uma função é uma forma de calcular a **variação instantânea em um ponto desta função**. Ou seja, dada uma função, a derivada é a expressão que nos ajuda a calcular a inclinação em um ponto específico. Por exemplo, para a função ***f(x) = 2x² - 2x + 0.5*** a derivada é ***f'(x) = 4x -2***. O que significa que, para qualquer ponto na função ***f(x)*** a inclinação neste ponto é calculado com ***f'(x)*** (o apóstrofo ao lado de f indica que **f'** é a função derivada de **f**).

![Exemplo derivadas](https://i.imgur.com/6DbzSIM.png)

Na imagem podemos observar que, quando **x = 1.5** a inclinação é 4, o que é verdade já que a reta vermelha (reta que tangencia o ponto x = 1.5) é positiva. Já quando **x = 0.5** temos a inclinação sendo 0, o que também se demonstra verdadeiro observando que a reta que tangencia o ponto onde x = 0.5 (reta verde) está completamente na horizontal, ou seja, sem inclinação.

Caso queira, é possível observar também o cálculo da reta tangente neste gráfico: https://www.desmos.com/calculator/9qjyfam2nw

Bom, então só precisamos descobrir qual a derivada da função erro, para assim, podermos obter a inclinação em um determinado ponto e poder dar um passo na direção que diminui o erro.

Não explicarei detalhadamente como calcular a derivada da função erro aqui, temos entendido o conceito de derivada e sabendo que a derivada da função de erro é **e'(w) = 2 * (valor predito - valor esperado) * x**, ou seja, a inclinação no ponto onde **x = 1.6** será:

    e'(w) = 2 * (16 - 14) * (-10)
    e'(w) = 2 * 2 * (-10)
    e'(w) = 4 * (-10)
    e'(w) = -40

# Como atualizar o valor de W ?

Quando o valor da inclinação é **negativo**, precisamos **aumentar o valor de w** para diminuir o erro, já para valores de inclinação **positivos**, precisamos **diminuir o valor de w** para diminuir o erro. Agora que sabemos a inclinação da função podemos atualizar o valor de w da seguinte forma.

    w = w - dw

Onde **dw** é a derivada do erro em função de w.

Se fizermos isso teremos: 

    w = 1.6 - (-40)
    w = 1.6 + 40
    w = 41.6

Funcionou, conseguimos aumentar o valor de w, porém, demos um passo muito grande. Isso acontece, pois o valor da inclinação é muito alto, para evitar que isso aconteça precisa pegar apenas uma fração do valor da inclinação, podemos fazer isso multiplicando dw por um número bem pequeno, normalmente chamado de taxa de aprendizado (learning rate). Esse valor é comumente representado pela letra grega eta (equivalente à letra H apesar de parecer a letra n). Vamos considerar essa taxa e aprendizado sendo **0.002**.

    w = w - (dw * eta)
    w = 1.6 - (-40 * 0.002)
    w = 1.6 - (-0.08)
    w = 1.6 + 0.08
    w = 1.68

E é assim que uma rede neural aprende, calculando predições com valores *"incorretos"*, usando estes para calcular a inclinação da função e dando pequenos passos em direção ao valor ideal.

Lembrando que, este é um exemplo extremamente limitado com foco na didática. Não estão sendo implementados aqui conceitos mais complexos como **funções de ativação**, **funções de custo**, **normalização de dados**, **gradiente descendente** ou até mesmo uma **"rede neural completa"** (uma rede de perceptrons interligados).