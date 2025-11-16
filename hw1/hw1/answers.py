r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers

part1_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part1_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============
# Part 2 answers

part2_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**Your answer:**

The ideal pattern to see in a residual plot is the one where all the points are very close to the 0 horizontal line.
That means that the residuals are very small all over the data, which indicates good model predictions.

Based on the residual plots we got, we can say that the model is quite accurate.
In all of the plots we can see that the majority of the dots are around the 0 horizontal line, although there are differences between the plots.
In the plot for the top-5 features, the dots are scattered widely around the 0 line. 
Also, we can see that there are many outliers which have inaccurate predictions.
In comparison, the final plot after CV, is much more compact around the 0 line, and has fewer outliers.
So, from the above observations we can conclude that the final model is quite fit, and is fitter than the beginning model.

"""

part3_q2 = r"""
**Your answer:**

We will explain the effect of adding non-linear features to our data, based of the given points:

1. **YES** , the model is still a linear regression model. Altough the model might not be linear in the variables, it is linear in the parameters. 
That means that it will find a linear combination in the new vector space.
We can look at it this way: by adding non-linear features to our data we will get new dimentions to the vector space where data lays. 
The added indices will be a skewed version of the original indices. 
In the new vector space, the regression will be linear and we will get a linear hyperplane.

The reason why it might not be a linear regression, is that we are using a non linear combinations of the data. 
Therefore we might get non linear hyperplanes, if we look at it from the perspective of the original vector space. 
As we said before, these hyperplanes will be linear in the new vector space. 


2. Theoretically, we **CAN** fit any non-linear function of the original features with this approach.
If we know the function of the original features, then we can add one index with the value of the function.
That way we would(probably) get very good predictions. 
**Note**: That contradicts the meaning of machine learning because we know the distribution beforehand... 
But it means that theoretically we could fit any non-linear function.

**Generally speaking**, we do not know the function beforehand and therefore we are limited to a linear combination of the original and added features.
So in most cases we **could not** fit any non-linear function of the original features.
For example, if we tried to fit an exponential with polynomial features, we would fail. 
That is because we cannot express an exponential function with a linear combination of polynomials.

3. As stated above, adding non-linear features will define a hyperplane in a vector space different than the original one. 
In the original vector space it might not have the shape or characteristics of a hyperplane.
So, people might say, it is not a hyperplane because of that. But because we are now looking at the new vector space, it is a hyperplane. 

"""

part3_q3 = r"""
**Your answer:**

1. Let's calculate the expected value $\mathbb{E}_{x,y}[|y-x|]$.
First, $x$ and $y$, both $\sim \text{Uniform}(0,1)$ and therfore $1=f_x(x)=f_y(y)=f_{x,y}(x,y)$

"""

# ==============

# ==============
