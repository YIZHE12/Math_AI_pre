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



### Proposed paper to read about loss function
___
#### Focal loss
This paper proposed a new loss funciton for dense objection detection. Its aim is to increase the one stage detector's accuracy so that it can match with the two stage detector methods while maintaing the advantage in speed. The new cost function is a dynamic scale cross entropy with modulation based on prediction confident. It emphasize the loss for low probability and reduce the influence of high confident prediction in the total loss, forcing the network to learn form the weak prediciton. The implementation is straight forward, by adding a modulating facotor ![](https://latex.codecogs.com/gif.latex?{_{(1-p{_t})}}^{\gamma&space;}), in the cross entropy equation before the summation. If ![](https://latex.codecogs.com/gif.latex?p_{t}) > 0.5, then this term will make its loss contribution smaller, and vice versa. They also proposed to keep ![](https://latex.codecogs.com/gif.latex?\alpha), which is the weighting factor for balanced cross entropy. So the final focal loss function is ![](https://latex.codecogs.com/gif.latex?FL(p{_t})&space;=&space;-\alpha_{t}(1-p{_t}){^{_{}}\gamma}&space;log(p{_t}))

Their proposed improvement is mainly based on the new loss function but not the archetecture themselves. Their RetianNet is based on two well known and well function articheture, the ResNet and FPN. The impact of this articles is that the proposed loss funciton can also be used in any other classification task. I tested the loss function on time series classification with highly inbalanced classes with an instance improvement. However, when I tested on image segmentation where the inbalanced class problem is less significant, the improvement is small. 

The authors also mentioned a few other types of methods for inbalanced classes: Hinge loss, weighted loss based on class distribution, Non-max suppresion and fixed background forground ratio. 

As Non-max suppresion and hinge loss both discard completely of data over a certain threshold but the focal loss still keep this informatin for later training. 


Due to the added exp weight based on class probability, it can be unstable. Atherefore, it needs to use sigmoid instead of ReLu, also it need to add alpha, and using prior for model initialization to damping down the effect of the exp term. However, because alpha is on top of the exp term, the impact and the range of an ideal alpha is small. I will suggest to add a linear term to damping down the effect, in other words, used <img src= 'CodeCogsEqn (3).gif'>

instead.

It is the opposite as Non-max suppresion, which remove all bounding box with low probability. But Non-max suppresion is more useful in YOLO, becasue the box size is small. But in the proposed RetinaNet, it has pyramid feature extraction, therefore it doesn't need that to remove false positive. It is problem is false negative. 

Anchor box: box with different shape for different classes

It compared the most three mehtods:

inbalanced classes weighfting, non-max suppresion, and hinge loss

It has several clever design
The skip connection and 1x1 conv (network in network) (bottleneck design) in 

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




