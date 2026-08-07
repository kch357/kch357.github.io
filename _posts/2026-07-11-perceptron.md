---
title: "[AI] 1. 활성화 함수(Activation Function)가 단조 증가해야 하는 이유"
author: kch357
date: 2026-07-11 15:13:00 +0800
categories: [AI]
tags: [AI, Perceptron, Neural Network, Activation Function]
render_with_liquid: false
math: true
---
간단하게 퍼셉트론(Perceptron)과 관련된 용어에 대해 알아보자.

![Desktop View](/assets/img/2026-07-11-perceptron/perceptron.png){: height="589"}

위 그림은 입력으로 2개의 신호를 받은 `퍼셉트론(Perceptron)`이다.<br>
그림의 원을 `뉴런(Neuron)` 혹은 `노드(node)`라고 부른다. $x_1$, $x_2$는 `activation`이라고 부르고 $w_1$, $w_2$는 `가중치(weight)`라고 한다.

출력값 $y$는 $h(w_1 x_1 + w_2 x_2 + b)$이다. 이 때 $b$를 `bias`라고 하고 $h(x)$를  `활성화 함수(Activation Function)`라고 한다.

![Desktop View](/assets/img/2026-07-11-perceptron/XOR.PNG){: height="589"}

위 그림은 **비선형 영역**인 **XOR연산**을 Single-layer Perceptron으로는 분리할 수 없음을 보여준다.

이 때 나는 궁금증이 생겼다. 다음과 같이 활성화 함수를 설정하면 해결할 수 있지 않을까?

$$
w_1, = w_2 = 0.5, b = 0
$$

$$
h(x) = \begin{cases} 1 & \text{if } x = 0 \text{ or } x = 1 \\ 0 & \text{if } x = 0.5 \end{cases}
$$

$x_1, x_2$는 0 또는 1의 값을 가질 것이므로 이렇게 활성화함수를 설정하면 하나의 층으로만 이루어진 Perceptron으로도 $h(w_1 x_1 + w_2 x_2 + b)$의 값은 XOR연산과 동일하게 1 또는 0의 값을 출력할 것이다.
 
하지만 우리가 사용되는 **신경망(Neural Network)**의 활성화 함수는 이렇게 설정되지 않는다.

### 활성화 함수(Activation Function)

활성화 함수는 단조증가 해야한다. <br>
입력($w_1x_1 + w_2x_2 + b$)이 커질수록, 그 뉴런이 출력(신호를 보낼 확률)도 커지거나 일정해야 한다는 의미를 가지고 있다.

`퍼셉트론`은 기본적으로 $w_1x_1 + w_2x_2 + b > 0$ 인 영역과 그렇지 않은 영역을 나눈다. 단조 증가인 `활성화 함수`를 사용하면, 이 부등식의 영역이 유지된다. <br>
함수가 감소 구간을 포함하면, 특정 입력값 범위에서는 '가중치 합'의 의미가 사라진다.

또한 딥러닝은 `경사하강법(Gradient Descent)`을 사용하기 때문에 학습을 하기 위해서는 함수가 미분 가능해야 하고, 가중치를 어느 방향으로 수정할지 계산할 수 있다.<br>
게다가 기울기가 0이 되거나 방향이 바뀌면 학습이 불안정해지므로 단조증가여야 한다.
