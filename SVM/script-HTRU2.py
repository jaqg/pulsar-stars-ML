import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn import svm, metrics
import sklearn.linear_model
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import sklearn.neighbors
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler, NearMiss
from collections import Counter
from tabulate import tabulate
import warnings

# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------
# matplotlib.use("pgf")
plt.style.use('mine.mplstyle')

# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------
def section(string):
    print("\n" + "+" + "-"*78 + "+")
    print("|" + string.center(78) + "|")
    print("+" + "-"*78 + "+" + "\n")

def save_plot(filename, formats):
    for name in filename:
        for format in formats:
            plt.savefig('figures/'+name+'.'+format,
                        transparent='True', bbox_inches='tight')

def oversampling(X, Y):
    over_sampler = RandomOverSampler(random_state=100)
    x_os, y_os = over_sampler.fit_resample(X, Y)
    return x_os, y_os

def maxColumn(A):    
    return list(map(max, zip(*A)))

def read_data(filename):
    data = pd.read_csv("./datasets/"+filename)
    return data

class model:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def split_data(self, cut):
        X = self.X
        Y = self.Y
        x_tr, x_te, y_tr, y_te = train_test_split(X, Y, test_size=cut)
        self.x_train = x_tr
        self.y_train = y_tr
        self.x_test = x_te
        self.y_test = y_te

    def build_PCA(self, n):
        pca = PCA(n_components=n)
        pca.fit(self.x_train)
        self.x_pca = pca.transform(self.x_train)

    def plot_PCA(self, filename, format):
        # plt.scatter(self.x_pca[:, 0], self.x_pca[:, 1], c=self.y_train,
        #             cmap=plt.cm.prism, edgecolor='k', alpha=0.7)
        plt.scatter(self.x_pca[:, 0], self.x_pca[:, 1], c=self.y_train)
        save_plot(filename, format)
        plt.close("all")

    def evaluate_SVM(self, kernel, class_weight=None):
        if class_weight:
            self.classifier = svm.SVC(kernel=kernel, class_weight=class_weight)
        else:
            self.classifier = svm.SVC(kernel=kernel)
        # Train the model using the training sets
        self.classifier.fit(self.x_train, self.y_train)
        # Predict the response for test dataset
        self.y_pred = self.classifier.predict(self.x_test)
        # Model Accuracy: how often is the classifier correct?
        self.acc = metrics.accuracy_score(self.y_test, self.y_pred)
        # print("Accuracy: {:.4f}".format(self.acc))
        # Model Precision: what percentage of positive tuples are labeled as such?
        self.prec = metrics.precision_score(self.y_test, self.y_pred)
        # print("Precision: {:.4f}".format(self.prec))
        # Model Recall: what percentage of positive tuples are labelled as such?
        self.rec = metrics.recall_score(self.y_test, self.y_pred)
        # print("Recall: {:.4f}".format(self.rec))

    def SVM_predict(self, X):
        Y = self.classifier.predict(X)
        return Y


# Silence warning
# warnings.filterwarnings(action="ignore")

# -----------------------------------------------------------------------------
# Load the data
# -----------------------------------------------------------------------------
data = read_data("HTRU_2.csv")
problem_data = read_data("problem.csv")

# -----------------------------------------------------------------------------
# Info of the dataset
# -----------------------------------------------------------------------------
section("Information of the dataset")
print("Size of the dataset:", data.shape)
# Count number of positives and negatives
num_data = data.shape[0]
num_pos = data["target_class"].value_counts()[1]
num_neg = data["target_class"].value_counts()[0]
print("Number of positive samples:", num_pos,
      "({:.2f}% of the total)".format(num_pos/num_data*100))
print("Number of negative samples:", num_neg,
      "({:.2f}% of the total)".format(num_neg/num_data*100))
columns = data.columns

# Visualize the data
print("\nSaving the pairplot...")
sns.pairplot(data, hue="target_class",palette="bright")
plot_formats = ['pgf', 'pdf', 'jpg']
save_plot('pairplot', ['jpg'])
plt.close("all")

# -----------------------------------------------------------------------------
# Preprocessing of the data
# -----------------------------------------------------------------------------
section("Preprocessing of the data")

# IMBALANCED DATA
print("- Imbalanced data -\n".center(80))
X = data.iloc[:, :-1]
Y = data.iloc[:, -1]
imbalanced = model(X, Y)

# Split the data
print("- Splitting the data into training and test sets...")
cut = 0.2
imbalanced.split_data(cut)
print("Size of the training set ({:.0f}%):".format(100*(1 - cut)),
      imbalanced.x_train.shape,
      f"; {Counter(imbalanced.y_train)}")
print("Size of the test set ({:.0f}%):".format(100*cut),
      imbalanced.x_test.shape,
      f"; {Counter(imbalanced.y_test)}")

# Build and Plot PCA
n_components = 2
print("\n- Building and plotting the PCA...\n")
# imbalanced.build_PCA(n_components)
# imbalanced.plot_PCA('pca-imbalanced', plot_formats)

# Balance the dataset
print("- Balanced data -".center(80))
print("\n- Balancing the dataset with...")

# -- Oversampling --
print("\n + Oversampling")
x_os, y_os = oversampling(imbalanced.x_train, imbalanced.y_train)
print("Size of the over-sampled training set:", x_os.shape)
print(f"Statistics: {Counter(y_os)}")

# Create the new class for the balanced data
balanced_os = model(X, Y)
balanced_os.x_train = x_os
balanced_os.y_train = y_os
balanced_os.x_test = imbalanced.x_test
balanced_os.y_test = imbalanced.y_test

# Build and Plot PCA
# balanced_os.build_PCA(n_components)
# balanced_os.plot_PCA('pca-balanced-os', plot_formats)

# -- SMOTE --
print("\n + SMOTE")
over_sampler = SMOTE(k_neighbors=2)
x_smote, y_smote = over_sampler.fit_resample(imbalanced.x_train,
                                             imbalanced.y_train)
print("Size of the SMOTE over-sampled training set:", x_smote.shape)
print(f"Statistics: {Counter(y_smote)}")

# Create the new class for the SMOTE balanced data
balanced_smote = model(X, Y)
balanced_smote.x_train = x_smote
balanced_smote.y_train = y_smote
balanced_smote.x_test = imbalanced.x_test
balanced_smote.y_test = imbalanced.y_test

# Build and Plot PCA
# balanced_smote.build_PCA(n_components)
# balanced_smote.plot_PCA('pca-balanced-smote', plot_formats)

# -- UNDERSAMPLE --
print("\n + UNDERSAMPLE")
under_sampler = RandomUnderSampler(random_state=100)
x_us, y_us = under_sampler.fit_resample(imbalanced.x_train,
                                        imbalanced.y_train)
print("Size of the under-sampled training set:", x_us.shape)
print(f"Statistics: {Counter(y_us)}")

# Create the new class for the undersampled balanced data
balanced_us = model(X, Y)
balanced_us.x_train = x_us
balanced_us.y_train = y_us
balanced_us.x_test = imbalanced.x_test
balanced_us.y_test = imbalanced.y_test

# Build and Plot PCA
# balanced_us.build_PCA(n_components)
# balanced_us.plot_PCA('pca-balanced-us', plot_formats)

# -- NEARMISS --
print("\n + NEARMISS")
under_sampler = NearMiss()
x_nearmiss, y_nearmiss = under_sampler.fit_resample(imbalanced.x_train,
                                                    imbalanced.y_train)
print("Size of the under-sampled training set:", x_nearmiss.shape)
print(f"Statistics: {Counter(y_nearmiss)}")

# Create the new class for the NearMiss balanced data
balanced_nearmiss = model(X, Y)
balanced_nearmiss.x_train = x_nearmiss
balanced_nearmiss.y_train = y_nearmiss
balanced_nearmiss.x_test = imbalanced.x_test
balanced_nearmiss.y_test = imbalanced.y_test

# Build and Plot PCA
# balanced_nearmiss.build_PCA(n_components)
# balanced_nearmiss.plot_PCA('pca-balanced-nearmiss', plot_formats)

# -- CLASSWEIGHT --
print("\n + CLASSWEIGHT")
n = Counter(imbalanced.y_train)
ratio = int(n[0]/n[1])
class_weight={0:1, 1:ratio}
print(f"n: {n}, ratio: {ratio}")
print(f"Class weight: {class_weight}")

balanced_cw = model(X, Y)
balanced_cw.x_train = imbalanced.x_train
balanced_cw.y_train = imbalanced.y_train
balanced_cw.x_test = imbalanced.x_test
balanced_cw.y_test = imbalanced.y_test

# Build and Plot PCA
# balanced_cw.build_PCA(n_components)
# balanced_cw.plot_PCA('pca-balanced-cw', plot_formats)

# -----------------------------------------------------------------------------
# Testing the classifier
# -----------------------------------------------------------------------------
section("Testing the classifier")
print("- Trainig & evaluating the SVM on...")

# Evaluate SVM on the raw data
print(" + Imbalanced data")
imbalanced.evaluate_SVM('rbf')

# Evaluate SVM with the oversampled balanced data
print(" + Oversampled balanced data")
balanced_os.evaluate_SVM('rbf')

# Evaluate SVM with the SMOTE balanced data
print(" + SMOTE balanced data")
balanced_smote.evaluate_SVM('rbf')

# Evaluate SVM on the undersampled balanced data
print(" + Undersampled balanced data")
balanced_us.evaluate_SVM('rbf')

# Evaluate SVM with the NearMiss balanced data
print(" + NearMiss balanced data")
balanced_nearmiss.evaluate_SVM('rbf')

# Evaluate SVM with the classs weight
print(" + Class-weighted imbalanced data")
balanced_cw.evaluate_SVM('rbf', class_weight=class_weight)

# -----------------------------------------------------------------------------
# Results
# -----------------------------------------------------------------------------
# Create a table
table_header = ["Classifier", "Accuracy", "Precision", "Recall"]
method_classes = ["imbalanced", "balanced_os", "balanced_smote", "balanced_us",
                  "balanced_nearmiss", "balanced_cw"]
table = [
        ["Imbalanced", imbalanced.acc, imbalanced.prec, imbalanced.rec],
        ["Oversampled", balanced_os.acc, balanced_os.prec, balanced_os.rec],
        ["SMOTE", balanced_smote.acc, balanced_smote.prec, balanced_smote.rec],
        ["Undersampled", balanced_us.acc, balanced_us.prec, balanced_us.rec],
        ["NearMiss", balanced_nearmiss.acc, balanced_nearmiss.prec,
         balanced_nearmiss.rec],
        ["Class weight", balanced_cw.acc, balanced_cw.prec, balanced_cw.rec],
        ]
print("\nResults:")
print(tabulate(table, headers=table_header, tablefmt="grid",
               floatfmt=".4f"))

# Store the results in a separate array by removing 1st column of table
results = []
for row in table:
    results.append(row[1:])

# Choose best method based on best accuracy
# best_acc = maxColumn(table[1:])[1]
# best_method = table[table[:][1].index(best_acc)][0]
# print("The best balanced method is", best_method)

# Choose best method based mean of 3 measures
mean_res = np.sum(results[1:], axis=1)/len(results[0])
best_method_index = mean_res.argmax()+1
best_method = table[best_method_index][0]
best_method_class = method_classes[best_method_index]
print("\nThe best balanced method is", best_method)

# -----------------------------------------------------------------------------
# Classify problem data
# -----------------------------------------------------------------------------
section("Classify problem data")

# Prediction using the best method
y_problem = balanced_os.SVM_predict(problem_data)

# Store the results
problem_data["target_class"] = y_problem
 
# Print the results
print(f"Problem data classified with {best_method}:\n")
print(problem_data)
 
# Count & print number of positives
# problem_positives = problem_data["target_class"].value_counts()[1]
# print("\nNumber of positives: {} \
# (out of {})".format(problem_positives, problem_data["target_class"].shape[0]))
