# Sine-Classifier
A simple neural network to classify points generated using a sine function.<br>
The model is trained using **gradient descent algorithm with backpropagation for calculating gradients**.<br>
Graphics are generated using [WinBGI](https://github.com/jatin-47/BFS-Visual/tree/SourceCode) graphics library in C++ language.


## Generating Data:
<img src="https://github.com/jatin-47/Sine-Classifier/blob/main/Images/Generated_points.png" align="right" style="display:inline;" width="420" >

Random points are generated for train and test set and are saved in ``training_set.txt`` and ``test_set.txt``. <br>

* Train data:
  - **Label 1 (RED)  :** Points on and above the sine graph.
  - **Label 0 (GREEN):** Points below the sine graph.

* Test data:
  - Shown in  **YELLOW**.

<br>

## Neural Network:
<img src="https://github.com/jatin-47/Sine-Classifier/blob/main/Images/NN.png" align="right" style="display:inline;" width="480" >

### Architecture :-

  - **Input Layer:** 2 nodes with two coordinates of a point.
  - **Hidden Layer1:** 20 nodes with sigmoid activation function.
  - **Hidden Layer2:** 20 nodes with sigmoid activation function.
  - **Output Layer:** 1 node with sigmoid activation function.

&nbsp; &nbsp; <img src="https://latex.codecogs.com/gif.latex?\sigma&space;(z)=&space;\frac{1}{1&space;&plus;&space;e^{-z}}" title="\sigma (z)= \frac{1}{1 + e^{-z}}" width="120" />

### Cost function :-
Mean Squared Error (MSE)

### Gradient Descent with Backpropagation :-

Some good resources:- 
- [Neural networks playlist by 3Blue1Brown](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
- [Backpropagation Example](https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/)
- [The Vanishing Gradient Problem](https://towardsdatascience.com/the-vanishing-gradient-problem-69bf08b15484)

## Results:
<img src="https://github.com/jatin-47/Sine-Classifier/blob/main/Images/Classified_points_NN.png" align="right" style="display:inline;" width="420" >

* Test data:
  - Points in **PINK** are classified into **Label 1 (RED)**.
  - Points in **BLUE** are classified into **Label 0 (GREEN)**.

&nbsp; &nbsp; <img src="https://github.com/jatin-47/Sine-Classifier/blob/main/Images/Classified_points_NN_result.png" width="350" >
<br>

<hr>

# Regression Approach (Classification using Logistic Regression):

Apart from this neural network classification which is more general, I tried classification using regression also.<br>
I assumed a general sine curve : **y = Amplitude * sin(freq * x + Phase) + Offset** <br>
And then based on these 4 parameters, tried to fit this graph according to the points given.<br>

## Results:
<img src="https://github.com/jatin-47/Sine-Classifier/blob/main/Images/Classified_points_reg.png" align="right" style="display:inline;" width="420" >

* Graph in **WHITE** is the fitted graph with the predicted 4 parameters.

* Test data:
  - Points in **PINK** are classified into **Label 1 (RED)**.
  - Points in **BLUE** are classified into **Label 0 (GREEN)**.

&nbsp; &nbsp; <img src="https://github.com/jatin-47/Sine-Classifier/blob/main/Images/Classified_points_reg_result.png" width="350" >
<br>










