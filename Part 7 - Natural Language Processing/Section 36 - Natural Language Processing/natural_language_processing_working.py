# Natural Language Processing

# Import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import the dataset, Specify to ignore Quotes
dataset = pd.read_csv("Restaurant_Reviews.tsv",delimiter="\t", quoting = 3)

# Cleaning the Text
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
#nltk.download('stopwords')
# Stemming using Porter Stemmer
ps = PorterStemmer()

corpus = []

for i in range(0,1000):
    review = re.sub('[^a-zA-Z]', ' ' ,dataset['Review'][i])
    review = review.lower()
    review =review.split()
    
    # For faster performance use Set of words instead of List
    review = [ps.stem(word) for word in review if not word in 
              set(stopwords.words('english'))]
    
    # Join the List to form the String of words
    review = ' '.join(review)
    
    # Append them to the list
    corpus.append(review)

# Create Bag Of Words
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:,-1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting classifier to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train,y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

