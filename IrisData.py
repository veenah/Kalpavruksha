
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# Load the data
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
data = pd.read_csv('IrisData.txt', sep = ',', names = names)

# Save some plots to the Graphs Directory
# Box & Whisker plots
data.plot(kind = 'box', subplots = True, layout = (2,2))
plt.savefig('Graphs/box_plot.png')
# Histograms
data.hist()
plt.savefig('Graphs/histogram.png')
# Scatter plot matrix
scatter_matrix(data)
plt.savefig('Graphs/scatter_matrix.png')

# Create the validation data
# 10-fold cross-validation
array = data.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X,Y, test_size = validation_size, random_state = seed)

scoring = 'accuracy'

"""  k-fold cross-validation
     In k-fold cross-validation we randomly partition a sample into k equally sized sub-samples.
     A simple sub-sample is retained as validation data. The remaining k - 1 samples are
     used as training data. We repeat the process k-times, using each sub-sample as validation
     data exactly once. The k results are averaged to produce a single estimation.
"""

# Using different algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB())),
models.append(('SVM', SVC()))

"""   
    Logistic Regression: 
        Regression analysis conducted when the dependent variable is dichotomous.
        Describes data, explaining the relationship between one dependent binary variable
        and one or more nominal, ordinal, interval, or ratio-level independent variables.
    Linear Discriminant Analysis:
        Dimensionality reduction technique. Projects a sample onto a lower-dimensional space
        with good class-separability in order to avoid over-fitting.
        Projects an n-dimensional sample onto a smaller subspace k where k <= n - 1 while 
        maintaining class-discriminatory information. Reduces computational cost.
    K Nearest Neighbors Classifier:
        To classify x find the closest K neighbors among the training points, x', and assign 
        to x the label of x'.
    Decision Tree Classifier:
        A decision tree is a hierarchical structure consisting of nodes and directed edges.
        Root, internal, and leaf/terminal nodes. Each terminal node is assigned a class 
        label. The root and internal nodes contain attribute test conditions to separate
        points with different characteristics.
    Gaussian Naive Bayes:
        Classification algorithm for binary and multi-class classification problems.
        Recall Bayes thm: P(h|d) = (P(d|h) * P(h)) / P(d). Assume a Gaussian distribution.
        Calculate the mean and std. deviation of each input variable. Probabilities of new
        inputs are calculated using the Gaussian Probability Density Function.
    Support Vector Clustering
        Data points are mapped from data space to a high-dimensional feature space using a 
        kernel function. We look for the smallest sphere enclosing the image of the data
        (Support Vector Domain Description algo.). The sphere, when mapped back to data space,
        forms a set of contours enclosing the data points. Interpret the contours as cluster
        boundaries, and points enclosed by each contour as associated by SVC to the same cluster.
"""

results = []
names = []

# Evaluate each model
for name, model in models:
    k_fold = model_selection.KFold(n_splits = 10, random_state = seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv = k_fold, scoring = scoring)
    results.append(cv_results)
    names.append(name)
    print("%s: %f (%f)" % (name, cv_results.mean(), cv_results.std()))

# Graph the model-comparison
fig = plt.figure()
fig.suptitle('Model Comparison for Iris Flower Data')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.savefig('Graphs/Model_Comparison.png')


# SVM was the most accurate model tested 
# Make predictions on the validation data
svm = SVC()
svm.fit(X_train, Y_train)
predictions = svm.predict(X_validation)
print('\n\n')
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from time import sleep

#Load iris.csv into a pandas dataframe 
iris = pd.read_csv("iris.csv")


#Using 2-D Scatter plot I can't make sense out of data so I tried to use 3-D plot,but it also did'nt turn to give nice result for visualization,
#and humans cant visualize 4-D plots ,so I decided to use Pair Plot
#Pairwise scatter plot :Pair Plot  

sns.set_style("whitegrid")
sns.pairplot(iris,hue='species',height=3)
#plt.show()


#Predicting the flower based on visualization of Pair Plots

petal_length = float(input("Enter the petal length:"))
petal_width  = float(input("Enter the petal width :"))

if petal_length <= 2.2 and petal_width <= 1.0 : 
	print("The given properties are of Setosa flower")
elif (petal_length >= 2.5 and petal_length <= 5.2) and (petal_width >= 0.8 and petal_width <= 1.8):
	print("The given properties are of Versicolor flower")
elif (petal_length >= 5.3 and petal_length <= 7.0) and (petal_width >= 1.8 and petal_width <= 2.5):
	print("The given properties are of Virginica flower")
else:
	print("Properties dont match any of the flowers")

print("Displaying Pair Plot ....")
sleep(5)
plt.show()

