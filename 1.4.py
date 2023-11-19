import numpy as np
import matplotlib.pyplot as plt

from sklearn import tree
from sklearn.metrics import accuracy_score

def evaluatePerformance():
    '''
    Evaluate the performance of decision trees,
    averaged over 100 trials of 10-fold cross-validation

    Return:
      a matrix giving the performance that will contain the following entries:
      stats[0,0] = mean accuracy of decision tree
      stats[0,1] = std deviation of decision tree accuracy
      stats[1,0] = mean accuracy of decision stump
      stats[1,1] = std deviation of decision stump accuracy
      stats[2,0] = mean accuracy of 3-level decision tree
      stats[2,1] = std deviation of 3-level decision tree accuracy

    ** Note that your implementation must follow this API**
    '''

    # Load Data
    filename = '/content/SPECTF.train'
    data = np.loadtxt(filename, delimiter=',')
    X = data[:, 1:]
    y = np.array([data[:, 0]]).T
    n, d = X.shape

    # shuffle the data
    idx = np.arange(n)
    np.random.seed(13)
    np.random.shuffle(idx)
    X = X[idx]
    y = y[idx]

    # Initialize variables for collecting results
    decision_tree_accuracies = []
    decision_stump_accuracies = []
    dt3_accuracies = []

    # Additional decision trees of varying limited depths
    dt_depth_accuracies = {2: [], 4: [], 6: []}

    # Initialize learning curve variables
    learning_curve_means = np.zeros((10, 4))
    learning_curve_stddevs = np.zeros((10, 4))

    # 100 trials of 10-fold cross-validation
    for _ in range(100):
        for i in range(10):
            # Split the data into train and test folds
            start = i * (n // 10)
            end = (i + 1) * (n // 10)

            Xtest = X[start:end, :]
            ytest = y[start:end, :]

            for j in range(1, 11):
                # Use j% of the training data
                train_size = int(j * 0.1 * (n - n // 10))
                Xtrain = X[:train_size, :]
                ytrain = y[:train_size, :]

                # Train the decision tree
                clf = tree.DecisionTreeClassifier()
                clf = clf.fit(Xtrain, ytrain)

                # Output predictions on the test fold
                y_pred = clf.predict(Xtest)

                # Compute accuracy for the decision tree
                acc_decision_tree = accuracy_score(ytest, y_pred)
                decision_tree_accuracies.append(acc_decision_tree)

                learning_curve_means[j-1, 0] += acc_decision_tree
                learning_curve_stddevs[j-1, 0] += acc_decision_tree ** 2

                # Train the decision stump
                clf_stump = tree.DecisionTreeClassifier(max_depth=1)
                clf_stump = clf_stump.fit(Xtrain, ytrain)

                # Output predictions on the test fold
                y_pred_stump = clf_stump.predict(Xtest)

                # Compute accuracy for decision stump
                acc_decision_stump = accuracy_score(ytest, y_pred_stump)
                decision_stump_accuracies.append(acc_decision_stump)

                learning_curve_means[j-1, 1] += acc_decision_stump
                learning_curve_stddevs[j-1, 1] += acc_decision_stump ** 2

                # Train the 3-level decision tree
                clf_dt3 = tree.DecisionTreeClassifier(max_depth=3)
                clf_dt3 = clf_dt3.fit(Xtrain, ytrain)

                # Output predictions on the test fold
                y_pred_dt3 = clf_dt3.predict(Xtest)

                # Compute accuracy for 3-level decision tree
                acc_dt3 = accuracy_score(ytest, y_pred_dt3)
                dt3_accuracies.append(acc_dt3)

                learning_curve_means[j-1, 2] += acc_dt3
                learning_curve_stddevs[j-1, 2] += acc_dt3 ** 2

                # Train additional decision trees of varying limited depths
                for depth in [2, 4, 6, 8, 10]:
                    if depth not in dt_depth_accuracies:
                      dt_depth_accuracies[depth] = []
                    clf_depth = tree.DecisionTreeClassifier(max_depth=depth)
                    clf_depth = clf_depth.fit(Xtrain, ytrain)

                    # Output predictions on the test fold
                    y_pred_depth = clf_depth.predict(Xtest)

                    # Compute accuracy for the additional decision trees
                    acc_depth = accuracy_score(ytest, y_pred_depth)
                    dt_depth_accuracies[depth].append(acc_depth)

                    learning_curve_means[j-1, 3] += acc_depth
                    learning_curve_stddevs[j-1, 3] += acc_depth ** 2

    # Compute mean and standard deviation for each classifier
    meanDecisionTreeAccuracy = np.mean(decision_tree_accuracies) if decision_tree_accuracies else np.nan
    stddevDecisionTreeAccuracy = np.std(decision_tree_accuracies) if decision_tree_accuracies else np.nan

    meanDecisionStumpAccuracy = np.mean(decision_stump_accuracies) if decision_stump_accuracies else np.nan
    stddevDecisionStumpAccuracy = np.std(decision_stump_accuracies) if decision_stump_accuracies else np.nan

    meanDT3Accuracy = np.mean(dt3_accuracies) if dt3_accuracies else np.nan
    stddevDT3Accuracy = np.std(dt3_accuracies) if dt3_accuracies else np.nan

    # Compute mean and standard deviation for the learning curve
    learning_curve_means /= 100
    learning_curve_stddevs = np.sqrt(learning_curve_stddevs / 100 - learning_curve_means ** 2)

    # Make certain that the return value matches the API specification
    stats = np.zeros((3, 2))
    stats[0, 0] = meanDecisionTreeAccuracy
    stats[0, 1] = stddevDecisionTreeAccuracy
    stats[1, 0] = meanDecisionStumpAccuracy
    stats[1, 1] = stddevDecisionStumpAccuracy
    stats[2, 0] = meanDT3Accuracy
    stats[2, 1] = stddevDT3Accuracy

    # Plot the learning curve
    x_vals = np.arange(0.1, 1.1, 0.1)
    plt.errorbar(x_vals, learning_curve_means[:, 0], yerr=learning_curve_stddevs[:, 0], label='Decision Tree')
    plt.errorbar(x_vals, learning_curve_means[:, 1], yerr=learning_curve_stddevs[:, 1], label='Decision Stump')
    plt.errorbar(x_vals, learning_curve_means[:, 2], yerr=learning_curve_stddevs[:, 2], label='3-level Decision Tree')
    plt.errorbar(x_vals, learning_curve_means[:, 3], yerr=learning_curve_stddevs[:, 3], label='Depth-limited Decision Tree')

    plt.xlabel('Percentage of Training Data')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    return stats

# Do not modify from HERE...
if __name__ == "__main__":
    stats = evaluatePerformance()
    print("Decision Tree Accuracy = ", stats[0, 0], " (", stats[0, 1], ")")
    print("Decision Stump Accuracy = ", stats[1, 0], " (", stats[1, 1], ")")
    print("3-level Decision Tree = ", stats[2, 0], " (", stats[2, 1], ")")
# ...to HERE.
