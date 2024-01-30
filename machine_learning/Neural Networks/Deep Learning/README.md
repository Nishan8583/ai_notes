# Deep Learning
- Several hidden layers (10-5).
- Choosing activation function must be careful, sigmoid won't be a good possible solution.
    - `loss function`: measures how close a given neural network is to its ideal state. for regression MSE.
    - for classification negetive log likelihood. Maximum likelihood estimation. 

## Rectified Linear (ReLu) Activation function
- Activates a node only if the input is above certain threshold.
- f(x) = max(0,x)
- Gradient is either 0 or a constant, can solve vanishing gradient issue.
    - `Vanishing Gradient` : training function relies on activation function , each of neural network wegiths receives an update propotional to current weight in each iteration of training, sometimes gradient becomes vanishingly small, so may prevent weight from changing its value or may stop the neural network completely.
    - Formulae `repeat until convergence {wj = wj-α*(dL(w)/dwi)}`. α is learning rate, when `dL(w)/dwi` is small or close to zero, weight won't change.
- `Softmax function` is usally used for output layer. Generalization of logistic/sigmoid function. for multiclass classificaiton like handwritten digits classification. 10 digits, 10 possible outcomes, 10 neurons in output layer, probability in every single class.
- Gradient descent has one issue, it works fine for `convex loss` function i.e. only one minumum value, but not in cases where there might be local and global minimum loss function. Optimization get stuck in local minum. `Normalization` (min-max Normalization) helps here, faster conversion and more accuracy.
- `stohastic gradient descent` computes the gradient on every training sample, instead of the whole training sample. faster but less accurate, different minimum on consequetives run.

## Hyper parameters
1. `Learning rate`: Defines the pace of learning. α > 1 is to high, α < 0.01 is too low.
2. `Momentum`: Helps learning algorithm get out of spots in search space where otherwise it would get stuck.
![Local_global.png](./Local_global.png)
- Value between 0 and  1, that increases the size of steps taken towards the minimum by trying to jump from a local minima.
- if this is large, keep learning rate smaller.
- If this is large, means convergence is large.
3. `Regularization`: Dropout, remove some neurons in hidden layer, with some probability. helps avoid overfitting.
