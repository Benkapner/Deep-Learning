r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""


# ==============
# Part 1 answers

part1_q1 = r"""
1) False. The test set is used to estimate the generalization error (out-of-sample error) of the model.
in-sample error (training error) is the error rate of a model on the training data.

2) False. If I have a dataset with 1000 samples and I split it into train set of 100 samples and test set of 900 samples.
In this case, the small train set could possibly not be a good representative of the whole dataset and could lead to overfitting

3) True. The test set is used to evaluate the final model's performance after all the model selection and hyperparameter tuning is done. 

4) 4. True. We can use splits to approximate the generalization error. 

"""

part1_q2 = r"""
The approach is not good, as the model selection process should be performed always without using the test set.
Using the test set to select the regularization hyperparameter $\lambda$ could result in overfitting to the test set, which in turn could lead to poor performance on new, unseen data.

Instead, the friend should use a separate validation set or perform cross-validation to select the best value of $\lambda$.
This approach ensures that the model is not overfitting to the test set and provieds a more realistic estimate to the model's performance on new unseen data.
"""

# ==============
# Part 2 answers

part2_q1 = r"""
The selection of $\Delta > 0$ in the SVM loss function is arbitrary because the regularization term controls the model's complexity, which affects the margin $\Delta$.
The regularization term encourages a simpler decision boundary, reducing the number of possible boundaries.
Different values of $\Delta$ may be suitable for different datasets or applications, but the regularization term ensures that the model can generalize well to unseen data.


"""

part2_q2 = r"""
The linear model learns a linear decision boundary and attempts to distinguish between different classes. The model in our case is learning the weights of the data which are scanned handwritten digits represented as pixels with varied intensity (representing how light or dark the pixel is). As the model is learning the weights for each pixel, it assigns weights according to relationships and may put more weight to pixels that make up the border of a digit or the center (for example). This can lead to potential classification errors when images are rotated, or if a digit was written at an angle. This can also cause classification errors between digits that share similarities such as 3 and 8 or 5 and 6.


"""

part2_q3 = r"""
1) We can see that we converge quickly and that the validation and training loss decrease in the same direction which implies that our learning rate is *good*. When the learning rate is set to a high value, the model converges quickly, but it may only reach a local minimum, which may not have the best performance, and the optimization may miss the mininum. A low learning rate takes longer to converge but it may produce better results.

2) Based on the metrics, we can see that as we approach concergence the validation accuracy is lower than training accuracy so there may be some overfitting (slight overfitting).



"""

# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
An ideal residual plot should show randomness, constant variance (homoscedasticity), a mean of zero, and independence. Randomness indicates that the model captures the relationship between variables without violating assumptions. Constant variance suggests that the spread of residuals is consistent across predicted values. A mean of zero indicates unbiased predictions. Independence means that residuals at one point do not provide information about residuals at other points.

Based on the provided residual plots, we can make several observations regarding the fitness of the trained model. Firstly, the plots indicate that the model performs relatively well in predicting mid and low-priced houses, as the residuals exhibit a good fit in those ranges. However, there seems to be less accuracy when predicting high-priced houses. This discrepancy may be attributed to a limited representation of high prices in the available data, which affects the randomness of the residuals. Despite this limitation, a positive aspect is that the residuals are centered around a zero mean, suggesting that the model does not suffer from a significant problem of over- or underestimation.


"""

part3_q2 = r"""
Adding non-linear features to the data would result in a model that is no longer a pure linear regression model.
Linear regression assumes a linear relationship between the predictors and the response variable.
By introducing non-linear features, the relationship between the predictors and the response becomes non-linear, making it a form of polynomial regression or non-linear regression.
So, adding non-linear features changes the nature of the regression model.

Yes, by adding non-linear features, we can capture and fit non-linear functions of the original features.
For example, if we have a feature 'x', we can create additional features like 'x^2', 'x^3', or even more complex non-linear transformations like 'sqrt(x)' or 'log(x)'.
These non-linear features allow the model to capture and represent non-linear relationships between the predictors and the response.

"""

part3_q2_ = r"""
In the case of a linear classification model, adding non-linear features would lead to a decision boundary that is no longer a simple hyperplane.
The decision boundary would become a more complex shape, potentially curved or nonlinear, due to the presence of the non-linear features.
This is because the non-linear features introduce additional dimensions and complexity to the model, allowing it to separate classes that may not be linearly separable in the original feature space.
We can think of polynomial features as an additional layer to our model which no longer keeps it a linear model.
"""

part3_q3 = r"""
1) This is beneficial for cross-validation as it allows us to explore a wider range of hyperparameter values with a smaller number of samples.
This is particularly useful for regularization parameters like $\lambda$, where the optimal value may lie on a logarithmic scale. 

2) The outer loop iterates over the combinations of degree and $\lambda$, resulting in $d \times l$ iterations.
Within each iteration of the outer loop, the model is fitted $k_{\text{folds}}$ times, where $k_{\text{folds}}$ is the number of folds used for cross-validation.
Therefore, the total number of times the model is fitted is $d \times l \times k_{\text{folds}}$.
There are $d$ degrees and $l$ $\lambda$ values, and $k_{\text{folds}}$ is set to 3, the model will be fitted $d \times l \times 3$ times in total.
degree_range has 3 values and the lambda_range has 20 values, the model will be fitted $3 \times 20 \times 3 = 180$ times in total.
Following cross-validation, the model undergoes a final fine-tuned fit on the complete training set, utilizing the selected hyperparameters.


"""

# ==============

# ==============
