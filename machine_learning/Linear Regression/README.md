# Linear Regression
- Data grows linearly.
![linear_regression.png](./linear_regression.png)
- we get the b values, with which we can predict other values
![mean_squared.png](./mean_squared.png)
!![3220f4dc2b0660c9b752b82a7efa097d.png](./3220f4dc2b0660c9b752b82a7efa097d.png "3220f4dc2b0660c9b752b82a7efa097d.png")![88be3c23b7472051504c43b549d01bbd.png](./88be3c23b7472051504c43b549d01bbd.png)
![4f9434f3aebef6190dce8d7aea3c5f6d.png](./4f9434f3aebef6190dce8d7aea3c5f6d.png)
- for every data row, caculate with b0 and b1, basically get prediction and substract with actual value and get MSE squared.
![gradiant-descent.png](./gradiant-descent.png)
- Get partial derivate of c(b), try to get low cost function of c(b)
- alpha symbol is the learning rate, the step we take to change b, if large, we take larger steps.
- Initially, random values of b are taken, and then adjusted.
![learning_rate.png](./learning_rate.png)
- R squred error = (1-RSS)/TSS, higher is better. 0.4 shows no linear relation
- RSS residual sum of sqaures sum of all E(H(x)-y)**2
- TSS Total sum of squares E(y-Mean)**2