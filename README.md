# Math_AI_prepration

## 1. Loss function:
___
### Classificaiton problem:

Cross entropy or Hinge loss

#### Entropy 

![](https://latex.codecogs.com/gif.latex?-\sum&space;_{i}{p_{i}}log{_2}({p_i}))

tell you how many bits you need to tranmit the messeges

#### cross-entropy

![](https://latex.codecogs.com/gif.latex?H({p},{q})=-\sum&space;_{i}{p_{i}}log{_2}({q_i}))

p: actual probability

q: predicted probability

![](https://people.richland.edu/james/lecture/m116/logs/log2.gif)

when p = q, the cross entropy has its minimum. It penalize the prediction that is highly confident but inaccurate.

For example, if one example belong to class 1, then the actually probability of class one is 1, class two is 0,
then the cross entropy will be -1log(1)+-0log(0) 
log(1) is zero and log(0) is -Inf, therefore, the cross entropy is 0


but on the other hand, if the classifier predict it is clas two while the true label is class one
the cross entropy will be Inf

    def cross_entropy(X,y):
      """
      X is the output from fully connect layer (shape with n_example, n_classes)
      y is labels (n_example, 1)
      """
      m = y.shape[0]
      p = softmax(X)
      log_likelihood = -np.log(p[range(m), y])
      loss = np.sum(log_likelihood)/m

      return loss

For unbalanced classes, can use Focal Loss:
![](https://latex.codecogs.com/gif.latex?FL(p{_t})&space;=&space;-\alpha_{t}(1-p{_t}){^{_{}}\gamma}&space;log(p{_t}))

#### Hinge loss

Hinge loss all penalize those that are correct but not confident

![](https://i.stack.imgur.com/Ifeze.png)

![](https://latex.codecogs.com/gif.latex?\sum&space;max(0,&space;1&space;-&space;{y_i}*h{_\theta}\left&space;(&space;x{_i}&space;\right&space;)))

Hinge loss is faster to train in gradient descent than cross entropy - real time desicion making

If accuracy is more important, use cross entropy

#### Kullback-Leibler divergence

In simplified terms, it is a measure of the distrance of two distributions. One appication of KL divergence is the famous generative adversarial network (GAN). 

![](https://i2.wp.com/syncedreview.com/wp-content/uploads/2017/07/fig_4.png?resize=771%2C301&ssl=1)

 [Example:](https://www.youtube.com/watch?v=LJwtEaP2xKA) For two coded method, P and Q, the KL divergence is the lower bound of the increase of average length to transmit a language that was coded in P but now need to be coded in Q. When P is the same as Q, then obviously, this increase is 0.

### Regression:
MSE, MAE or Huber loss

MAE (L1 loss), is more robust to outliers than MSE

Huber loss, even more robust to outliers


#### History of CNN

1. LeNet: 
Use conV net (shared weight)

2. AlexNet:
ReLU, dropout, Data augmentation

3. VGG-16/19
Stage-wise training - requried Xavier/MSRA initation

4. GoogleNet
Multiple branches, shortcuts, bottleneck

Use Batch Norm

5. ImageNet
Deep residual learning
1x1 conv
has lower time complexity than VGG-16/19




