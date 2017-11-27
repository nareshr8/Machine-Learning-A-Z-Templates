# Polynomial Linear Regression

# Importing the dataset
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[2:3]

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
#library(caTools)
#set.seed(123)
#split = sample.split(dataset$Level, SplitRatio = 0.8)
#training_set = subset(dataset, split == TRUE)
#test_set = subset(dataset, split == FALSE)

# Feature Scaling
# training_set = scale(training_set)
# test_set = scale(test_set)

# Fitting to Linear Regression
linear_regression = lm(formula = Salary ~ . ,data = dataset)

# Fitting into Polynomial Linear Regression
dataset$Level2 = dataset$Level^2
dataset$Level3 = dataset$Level^3
dataset$Level4 = dataset$Level^4
polynomial_regression = lm(formula = Salary ~ . ,data = dataset)


# Visualise the Linear Regression
library(ggplot2)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = 'red') +
  geom_line(aes(x = dataset$Level, y = predict(linear_regression, newdata = dataset)),
             colour = 'blue') +
  geom_line(aes(x = dataset$Level, y = predict(polynomial_regression, newdata = dataset)),
            colour = 'green') +
  ggtitle('Linear Regression Prediction') +
  xlab('Years') +
  ylab('Salary')


# Predict the values using Linear Regression
y_pred = predict(linear_regression, data.frame(Level = 6.5))


# Predict the values using Polynomial Regression
y_pred = predict(polynomial_regression, data.frame(Level = 6.5,
                                                   Level2 = 6.5^2,
                                                   Level3 = 6.5^3,
                                                   Level4 = 6.5^4))