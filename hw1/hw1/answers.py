r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers

part1_q1 = r"""
1. False.  
A split of the data into two disjoint subsets does not always constitute a useful train–test split.  
For example, in a cat–dog classification task, if all dog images are placed in the training set and all cat images in the test set, the model will only learn what dogs look like and will fail on cats.  
If the training set is not representative of the full distribution, the evaluation is misleading and the split is not useful.

2. False.  
The test set must not be used during cross-validation. CV is performed only on the training data for tuning hyperparameters.  
Using the test set during CV leaks information into training, causing overfitting to the test set and producing a biased, unreliable evaluation.

3. True.  
In each fold of cross-validation, the model is trained on part of the data and evaluated on a held-out fold, simulating unseen data.  
Since every sample is used as validation exactly once, the average performance across folds provides a reliable estimate of generalization.

4. False.  
Injecting noise into the labels does not test robustness—it corrupts the ground truth and only checks whether the model can memorize wrong targets.  
Adding noise to the input data itself preserves the labels and *does* test robustness by checking whether the model handles realistic variations without failing.

"""

part1_q2 = r"""
No, your friend's approach is not justified.
The test set should be used only once for the final, unbiased evaluation of the model, and must not take part in the hyperparameter selection process.
By training models with different values of $\lambda$ and choosing the one that performs best on the test set, information from the test set leaks into the training procedure, causing the model to overfit to the test set and producing an overly optimistic estimate of performance.
The correct procedure is to tune $\lambda$ using a validation set (or cross-validation) created from the training data, and only after selecting the best $\lambda$ should the final evaluation be performed on the test set.
"""

# ==============
# Part 2 answers

part2_q1 = r"""
If we allow $\Delta < 0$, the SVM loss becomes meaningless.  
A negative margin allows bad predictions to avoid being penalized, because the hinge term can become negative and then is clipped to zero.

For example, consider the sample $s_i = (x_i, y_i)$ and take:
$w_j^T x_i = 0.5$, $w_{y_i}^T x_i = 0.4$, and $\Delta = -0.2$.

Then the margin term is:
$
\Delta + w_j^T x_i - w_{y_i}^T x_i
= -0.2 + 0.5 - 0.4
= -0.1 < 0.
$

Therefore,

$\max(0,\,-0.1) = 0.$

so the model receives **no loss even though the wrong class has a higher score than the correct class**.

Thus, $\Delta < 0$ breaks the meaning of the SVM loss and encourages incorrect behavior.
"""

part2_q2 = r"""
Based on the weight visualization, we see that the linear model is essentially learning a template for each digit. 
Since the model is linear, each class score is just a dot-product $w_j^T x_i$, meaning it compares the input image 
with the learned weight image for that class. Large positive weights highlight pixels that support the class, 
while negative weights penalize pixels that contradict the template. For example, the weight map for the digit “0” 
shows a bright circular outline, indicating that images with an oval shape in the center produce a high score for class 0.

Because the model is linear, it cannot capture complex shapes or spatial relationships; it can only look for 
global correlations between pixel intensities and the weight template. This explains many of the classification 
errors. Digits that share similar overall pixel patterns-such as 4 vs 9, 7 vs 1, or 5 vs 6-produce similar 
dot-products with the weight vectors, causing confusion. If a digit is written in an unusual style or slightly 
distorted, its pixel pattern may resemble the wrong class template more than the correct one, leading to mistakes.

In summary, the model behaves like a template matcher: it multiplies the image with each class weight image and 
chooses the template with the highest similarity. Therefore, errors occur mainly when two digits have similar 
global shapes or when the handwriting does not match the learned template well.
"""

part2_q3 = r"""
1.
The learning rate we chose appears to be **good**.
The training loss decreases smoothly over time, **without sudden spikes**, and it **converges nicely** toward a minimum. This indicates that the step sizes taken during gradient descent are appropriate.

If the learning rate were too small, the updates would make only tiny progress each step, so within the same number of epochs the loss would decrease very slowly and might not converge to a good minimum.

If the learning rate were too large, each update would make big jumps in parameter space, causing the loss to change dramatically, bounce around, or even diverge. The plot would show large oscillations or increases in loss instead of a smooth downward trend.


2.
The model is **slightly overfitted** to the training set.
During the epochs, the training accuracy is consistently higher than the validation accuracy, but the gap between them is relatively small.
This indicates that the model is slightly overfitted: it has learned patterns that fit the training data better than the unseen validation data, but the difference is small enough to show that it still generalizes reasonably well.

The fact that the training accuracy is higher is normal, but overfitting happens when the gap becomes large
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

1. The expected value is $\mathbb{E}_{x,y}[|y-x|] = \frac{1}{3}$.

First, $x$ and $y$ both $\sim \text{Uniform}(0,1)$ and therefore $1=f_x(x)=f_y(y)=f_{x,y}(x,y)$.

Hence, 

$\mathbb{E}_{x,y}[|y-x|] = \int_{0}^1\int_{0}^{1}f_{x,y}(x,y)|x-y|\,dx\,dy = \int_{0}^1\int_{0}^{1}|x-y|\,dx\,dy =
\int_{0}^1(\int_{0}^{y}|x-y|\,dx + (\int_{y}^{1}|x-y|\,dx))\,dy =
\int_{0}^1(\int_{0}^{y}(y-x)\,dx + (\int_{y}^{1}(x-y)\,dx))\,dy =
\int_{0}^1(\bigg[\, yx - \frac{1}{2}x^2 \,\bigg]_{0}^{y} + \bigg[\, \frac{1}{2}x^2 - yx \,\bigg]_{0}^{y})\,dy =
\int_{0}^1((y^2-\frac{1}{2})+(\frac{1}{2} - y - \frac{1}{2}y^2 + y^2))\,dy = 
\int_{0}^1(\frac{1}{2}-y+y^2)\,dy = 
\bigg[\, \frac{1}{2}y - \frac{1}{2}y^2 + \frac{1}{3}y^3 \,\bigg]_{0}^{1} =
\frac{1}{3}
$



2. The expected value is $\mathbb{E}_{x}[|\hat{x}-x|] = \frac{1}{2} - \hat{x} + \hat{x}^2$.

First, as in section 1, $f_{x}(x)=1$.

Second, we will treat $\hat{x}$ as a known value.

Therefore:

$\mathbb{E}_{x}[|\hat{x}-x|] =
\int_{0}^{1}f_{x}(x)|\hat{x}-x|\,dx = 
\int_{0}^{1}1\cdot|\hat{x}-x|\,dx = 
\int_{0}^{\hat{x}}(\hat{x}-x)\,dx + \int_{\hat{x}}^{1}(x-\hat{x})\,dx =
\bigg[\, \hat{x}x - \frac{1}{2}x^2 \,\bigg]_{0}^{\hat{x}} + \bigg[\, \frac{1}{2}x^2 - \hat{x}x \,\bigg]_{\hat{x}}^{1} =
\hat{x}^2 - \frac{1}{2}\hat{x}^2 + \frac{1}{2} - \hat{x} - \frac{1}{2}\hat{x}^2 + \hat{x}^2 = 
\frac{1}{2} - \hat{x} + \hat{x}^2
$

3. Dropping the scalar value of the polynomial will not give the correct expected value.

**But**, beacuse we will be using this expected value as a part of a loss function, this value will not be relevant anymore.
After removing the scalar value off the polynomial, the minimizing operation will be the same. 
That is because adding\\multiplying a function by a scalar keeps the max and min points.

Moreover, this will simplify our function. 
Therefore we would like to use this "trick" for easier calculations. 

"""

# ==============

# ==============
