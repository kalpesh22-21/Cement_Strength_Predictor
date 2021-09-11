# Problem Statement
In this preoject we tested various regression models to predict the concrete compressive strength which is a highly nonlinear function of age and ingredients.
 
## Project Overview: 
* Collected data of Cement Strength, values of ingredients, cement composition and age.
* Checked Linear Independency in the features using co-relation.
* Feature Engineering - Used Literature review to generate a  new feature water-cement ratio & Cement composition ratio.
* Feature Selection using mlxtend.feature_selection on RandomForest Regressor. 
* Used XGB, RandomForest, Logistic, KNN regressors.
* Optimized hyperparameters is GridSearchCV & fine tuned the model.
* Checked the Confidence interval using Resampling of the data. 

## Code and Resources Used 
**Python Version:** 3.7  
**Packages:** mlxtend.pandas, numpy, sklearn, matplotlib,XGB. 

## EDA
* We cannot see high correlation values among any attributes thus we cannot directly drop any attributes for feature selection.
* The maximum positive corelation value we see between the attributes is of 0.38 i.e. between (Superplastic & Ash) and maximum negative corelation value we see between the attributes is of -0.66 i.e. between (Superplastic & Water)
* Cement & Ash are also showing relatively high negative co-relation of -0.4 and also Fineagg & water are also showing relatively high negative co-relation of -0.45.
* In co-relation of attributes vs the strength (target value) we can see cement being the most important feature or attribute have a co-realtion of 0.5 followed by Superplastic and Age. Which can be explained scientifically, as the amount of cement increases the strength of the mixture also increases, as it is allowed to set for longer time it the strength of the mixture increases.
* No attribute have very less co-relation with the strength thus we will not drop any attributes.

### Correlation Matrix
<img target="_blank" src="https://github.com/kalpesh22-21/Cement_Strength_Predictor/blob/main/Correlation%20Heatmap.png" width=700>

### Multivariate Scatter Plot
<img target="_blank" src="https://github.com/kalpesh22-21/Cement_Strength_Predictor/blob/main/Feature%20Co-relation.png" width=700>

### Accuracy for Feature Selection
<img target="_blank" src="https://github.com/kalpesh22-21/Cement_Strength_Predictor/blob/main/Features%20Accuracy.png" width=270>

( 0, 1, 2, 3 denotes number of features droped from the training data)
We can see that the model with 8 ,9 and 10 features have very close average CV score.
Models with 8 and 10 Features are also been trained and tested but the model with 9 Features have provided with the best accuracies and R2 scores.
The list for the selected features are:

['cement', 'slag', 'ash', 'water', 'superplastic', 'fineagg', 'age', 'Cement_agg_ratio', 'Water_Cement_ratio']

## Model performance
The XGB (Xtreme Gradient Boosting ) model outperformed the other approaches on the test.

This model delivered an accuracy between 89.95% to 93.77% in the production with a confidence level of 95%
<img target="_blank" src="https://github.com/kalpesh22-21/Cement_Strength_Predictor/blob/main/Accuracy.png" width=270>
