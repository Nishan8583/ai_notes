# Convolutional Neural Networks (CNN)
- Deep learning is cool, but as number of neurons increase, so does the number of weights.
- This creates a lot of connections, `combinatorial explosion`.
- Training becomes very slow.
- Soln -> `Convolutional Neural Networks`.
    - Nerons are not connected to every other neurons.
    - They have assumptions, Ex: inputs are images so we can encode certain properties into the architecture.
    - Under the hood, uses standard Neural network, but at beginning it transforms data in order to achieve best accuracy.
    - Takes the image as an input, subjects it to combinations of weights and biases, `extracts features` and outputs the results ([Medium](https://medium.com/analytics-vidhya/understanding-convolution-operations-in-cnn-1914045816d4))
    - Reduces feature dimension.
    - For example, for smile detection, we only need the smile portion, lets say the lips (the grin) in the face.

## Steps in CNN
1. Convolutional operation:
- Most of the points and images here are obtained from [this medium article](https://medium.com/analytics-vidhya/understanding-convolution-operations-in-cnn-1914045816d4). 
- From wikepiedia
<!--StartFragment-->

In mathematics convolution is a mathematical operation on two functions that produces a third function expressing how the shape of one is modified by the other

<!--EndFragment-->
- ITs for image.
<!--StartFragment-->

The shape of a kernel is heavily dependent on the input shape of the image and architecture of the entire network, mostly the size of kernels is **(MxM)** i.e a square matrix.

<!--EndFragment-->
- **Stride** defines by what step does to kernel move, for example stride of 1 makes kernel slide by one row/column at a time and stride of 2 moves kernel by 2 rows/columns.
![Stride.gif](./Stride.gif)
- Convolition operation
![operation.png](./operation.png)
<!--StartFragment-->

Sliding window protocol:

1.  The kernel gets into position at the top-left corner of the input matrix.
2.  Then it starts moving left to right, calculating the dot product and saving it to a new matrix until it has reached the last column.
3.  Next, kernel resets its position at first column but now it slides one row to the bottom. Thus following the fashion left-right and top-bottom.
4.  Steps 2 and 3 are repeated till the entire input has been processed.

<!--EndFragment-->
