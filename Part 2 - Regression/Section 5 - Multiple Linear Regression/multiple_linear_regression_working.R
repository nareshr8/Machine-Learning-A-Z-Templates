# Multiple Linear Regression

# Importing the dataset
dataset = read.csv('50_Startups.csv')

# Encoding categorical data
dataset$State = factor(dataset$State,
                       levels = c('New York', 'California', 'Florida'),
                       labels = c(1, 2, 3))

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Profit, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Fit the model to training set
# regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State , data = training_set)
regressor = lm(formula = Profit ~ . , data = training_set)
# To check the P value : summary(regressor)

# Predict the test set results
y_pred = predict(regressor , newdata = test_set)

# Building the Optimal model using Backward Elimination
regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State , data = dataset)
summary(regressor)

regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend , data = training_set)
summary(regressor)

regressor = lm(formula = Profit ~ R.D.Spend + Marketing.Spend , data = training_set)
summary(regressor)

regressor = lm(formula = Profit ~ R.D.Spend , data = training_set)
summary(regressor)
