# Math_AI_prepration
## 1. Loss function:

#### Classificaiton problem:

Cross entropy or Hinge loss

Entropy: 

![](https://latex.codecogs.com/gif.latex?-\sum&space;_{i}{p_{i}}log{_2}({p_i}))

Hinge loss is faster to train in gradient descent than cross entropy - real time desicion making

If accuracy is more important, use cross entropy

#### Regression:
MSE, MAE or Huber loss

MAE (L1 loss), is more robust to outliers than MSE
Huber loss, even more robust to outliers
