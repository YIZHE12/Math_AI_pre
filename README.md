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






#### Hinge loss

Hinge loss all penalize those that are correct but not confident

![](https://i.stack.imgur.com/Ifeze.png)

![](https://latex.codecogs.com/gif.latex?\sum&space;max(0,&space;1&space;-&space;{y_i}*h{_\theta}\left&space;(&space;x{_i}&space;\right&space;)))

Hinge loss is faster to train in gradient descent than cross entropy - real time desicion making

If accuracy is more important, use cross entropy

### Regression:
MSE, MAE or Huber loss

MAE (L1 loss), is more robust to outliers than MSE

Huber loss, even more robust to outliers



Proposed paper to read about loss function

Focal loss

Due to the added exp weight based on class probability, it can be unstable. Atherefore, it needs to use sigmoid instead of ReLu, also it need to add alpha, and using prior for model initialization to damping down the effect of the exp term 

It is the opposite as Non-max suppresion, which remove all bounding box with low probability. But Non-max suppresion is more useful in YOLO, becasue the box size is small. But in the proposed RetinaNet, it has pyramid feature extraction, therefore it doesn't need that to remove false positive. It is problem is false negative. 

Anchor box: box with different shape for different classes

It compared the most three mehtods:

inbalanced classes weighfting, non-max suppresion, and hinge loss

It has several clever design
The skip connection and 1x1 conv (network in network) (bottleneck design) in 
