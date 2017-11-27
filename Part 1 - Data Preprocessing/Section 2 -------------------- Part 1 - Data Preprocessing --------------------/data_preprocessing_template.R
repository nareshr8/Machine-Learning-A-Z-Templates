#Import Dataset
dataset = read.csv('Data.csv')

#Deal with missing values
dataset$Age = ifelse(is.na(dataset$Age),
                     ave(dataset$Age, FUN = function(x) mean(x, na.rm = TRUE)),
                     dataset$Age)

#Deal with Categorical Data
dataset$Salary = ifelse(is.na(dataset$Salary),
                     ave(dataset$Salary, FUN = function(x) mean(x, na.rm = TRUE)),
                     dataset$Salary)
dataset$Country = factor(dataset$Country, c('France','Spain','Germany'),c(0,1,2))
dataset$Purchased = factor(dataset$Purchased, c('Yes','No'),c(0,1))

#Split the data set to training and test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Purchased,SplitRatio = 0.8)
training_set = subset(dataset,split==TRUE)
test_set = subset(dataset,split==FALSE)

#Feature Scaling
training_set[,2:3]=scale(training_set[,2:3])
test_set[,2:3]=scale(test_set[,2:3])