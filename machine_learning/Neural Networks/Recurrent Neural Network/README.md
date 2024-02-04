# Recurrent Neural Network (RNN)
- Extremely important in Natural Language Processing.
- Google translate relies heavily on it.
- Can use RNN to make time series analysis, determine stock prices.
- `Turing Test`: a computer passes turing test if a human is unable to distinguish the computer from a human in a blind test.
- RNN can pass turing test, ex: a well trained RNN model can speak english.
- `LEARN LANGUAGE MODELS`.
- Can understand connection between data that are far away.
    - Ex: "I am from nepal, ..… jpt test ..… I can speak ". It is able to gues the last word `nepali`.
- In combination with CNN, we can also get a description from an image.

### The Basics
- In Feed Forward Neural Network, training samples are not related to each other. p(t) is not related to p(t-1) or p(t-2) …
- Ex: Tigers, elepants, cows have nothing to do wth each other.
- In RNN training samples are correlated with eac other, p(t) depends on p(t-1), p(t-2) …
![RNN-representation.png](./RNN-representation.png)
- Every single neuron points back to itself as well
![formulae.png](./formulae.png)
- several paramters are shared across layers.
- How to train RNN ? we unroll in time to end up with a feedforward network in that particular time.
![unroll.png](./unroll.png)
Image pulled from [this article](https://medium.com/towards-datascience/recurrent-neural-networks-101-1c1eea9d1776).
- `Vanising gradiet problem/exploding gradient`: Backgrpopagation through time, gradients are passed from future time to current time, as weights are updated with value < 1, multiple times cause many layers, `very very deep`, it becomes tooo close to zero, `edge weight wont update furhter from its layers`, or tooo large, so the model stops learning.
- There are many local optima, but we want to global optima., meta heuristic algorithms.
- Soln for exploding gradient: truncated BPTT, small backpropagation through k time steps, adjust learning rate.
- Soln for vanihsing gradient problem: initialize weghts properly (xavier-inilization), use proper activation functions like RELU, use other architectures like LSTM or GRUs.