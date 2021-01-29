import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap # for plot
from sklearn.impute import SimpleImputer # for missing data
from sklearn.compose import ColumnTransformer # to transform categorical to binary
from sklearn.preprocessing import OneHotEncoder # to transform categorical to binary
from sklearn.model_selection import train_test_split # for splitting data
from sklearn.preprocessing import StandardScaler # for feature scaling

# classifier section
from sklearn.linear_model import LogisticRegression # for Logistic Regression
from sklearn.neighbors import KNeighborsClassifier # for KNN

# matrix
from sklearn.metrics import confusion_matrix, accuracy_score # for making confusion matrix

def knn():

    # import data
    dataset = pd.read_csv('Social_Network_Ads.csv')
    print(dataset.head(5))

    # independent variables
    independent = dataset.iloc[:, :-1].values

    # dependent variable
    dependent = dataset.iloc[:, -1].values

    # print(independent, dependent)

    # splitting dataset into 4 parts 300 customers go to train
    x_train, x_test, y_train, y_test = train_test_split(independent, dependent, train_size=0.75, random_state=1)

    # feature scaling
    # this is not necessary to do, but for better prediction we do
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.fit_transform(x_test)

    # classifiler
    classifier = KNeighborsClassifier(n_neighbors=5, p=2)
    classifier.fit(x_train, y_train)
    
    # prediction 
    print('The prediction value is %.2f' %classifier.predict(sc.transform([[30, 80000]])))

    # confusion matrix
    cm = confusion_matrix(y_true = y_test, y_pred = classifier.predict(x_test))
    print(cm)

    print('\nThe correctness is %.2f percent' %accuracy_score(y_true = y_test, y_pred = classifier.predict(x_test)))

    # taking several min to plot
    # because a lot of calculations behind the code

    # Train set visualization
    # x_set, y_set = sc.inverse_transform(x_train), y_train
    # X1, X2 = np.meshgrid(np.arange(start=x_set[:, 0].min() - 10, stop=x_set[:, 0].max() + 10, step=1),
    #                      np.arange(start=x_set[:, 1].min() - 1000, stop=x_set[:, 1].max() + 1000, step=1))
    # plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
    #              alpha=0.75, cmap=ListedColormap(('red', 'green')))
    # plt.xlim(X1.min(), X1.max())
    # plt.ylim(X2.min(), X2.max())
    # for i, j in enumerate(np.unique(y_set)):
    #     plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1], c=ListedColormap(('red', 'green'))(i), label=j)
    # plt.title('K-NN (Training set)')
    # plt.xlabel('Age')
    # plt.ylabel('Estimated Salary')
    # plt.legend()
    # plt.show()
    #
    # # Visualising the Test set results
    # X_set, y_set = sc.inverse_transform(x_test), y_test
    # X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 10, stop=X_set[:, 0].max() + 10, step=1),
    #                      np.arange(start=X_set[:, 1].min() - 1000, stop=X_set[:, 1].max() + 1000, step=1))
    # plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
    #              alpha=0.75, cmap=ListedColormap(('red', 'green')))
    # plt.xlim(X1.min(), X1.max())
    # plt.ylim(X2.min(), X2.max())
    # for i, j in enumerate(np.unique(y_set)):
    #     plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c=ListedColormap(('red', 'green'))(i), label=j)
    # plt.title('K-NN (Test set)')
    # plt.xlabel('Age')
    # plt.ylabel('Estimated Salary')
    # plt.legend()
    # plt.show()


if __name__ == '__main__':
    knn()

