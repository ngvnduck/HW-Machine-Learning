import numpy as np
from sklearn import tree
from sklearn.metrics import accuracy_score

def evaluatePerformance():
    # Load Data
    filename = '/content/SPECTF.dat'
    data = np.loadtxt(filename, delimiter=',')
    X = data[:, 1:]
    y = np.array([data[:, 0]]).T
    n, d = X.shape

    # Shuffle the data
    idx = np.arange(n)
    np.random.seed(13)
    np.random.shuffle(idx)
    X = X[idx]
    y = y[idx]

    # Number of trials and folds
    num_trials = 100
    num_folds = 10

    # Initialize arrays to store accuracy values
    decision_tree_accuracies = np.zeros(num_trials)
    decision_stump_accuracies = np.zeros(num_trials)
    dt3_accuracies = np.zeros(num_trials)

    for t in range(num_trials):
        # Shuffle the data at the start of each trial
        np.random.shuffle(idx)
        X = X[idx]
        y = y[idx]

        # num_folds = 10 => fold_size = 26
        fold_size = n // num_folds

        # Initialize arrays to store accuracy values for each fold
        decision_tree_fold_accuracies = np.zeros(num_folds)
        decision_stump_fold_accuracies = np.zeros(num_folds)
        dt3_fold_accuracies = np.zeros(num_folds)

        for fold in range(num_folds):
            # Split data into training and testing sets
            test_indices = range(fold * fold_size, (fold + 1) * fold_size)
            train_indices = [i for i in range(n) if i not in test_indices]

            Xtrain, Xtest = X[train_indices, :], X[test_indices, :]
            ytrain, ytest = y[train_indices, :], y[test_indices, :]

            # Train the decision tree
            clf = tree.DecisionTreeClassifier()
            clf = clf.fit(Xtrain, ytrain)

            # Output predictions on the test data
            y_pred = clf.predict(Xtest)

            # Compute accuracy for decision tree
            decision_tree_fold_accuracies[fold] = accuracy_score(ytest, y_pred)

            # Train the decision stump
            clf_stump = tree.DecisionTreeClassifier(max_depth=1)
            clf_stump = clf_stump.fit(Xtrain, ytrain)

            # Output predictions on the test data for decision stump
            y_pred_stump = clf_stump.predict(Xtest)

            # Compute accuracy for decision stump
            decision_stump_fold_accuracies[fold] = accuracy_score(ytest, y_pred_stump)

            # Train the 3-level decision tree
            clf_dt3 = tree.DecisionTreeClassifier(max_depth=3)
            clf_dt3 = clf_dt3.fit(Xtrain, ytrain)

            # Output predictions on the test data for 3-level decision tree
            y_pred_dt3 = clf_dt3.predict(Xtest)

            # Compute accuracy for 3-level decision tree
            dt3_fold_accuracies[fold] = accuracy_score(ytest, y_pred_dt3)

        # Average accuracy over folds for each trial
        # np.mean() => Calculate avarage number of numpyarray
        decision_tree_accuracies[t] = np.mean(decision_tree_fold_accuracies)
        decision_stump_accuracies[t] = np.mean(decision_stump_fold_accuracies)
        dt3_accuracies[t] = np.mean(dt3_fold_accuracies)

    # Compute mean and standard deviation over all trials
    mean_decision_tree_accuracy = np.mean(decision_tree_accuracies)
    std_decision_tree_accuracy = np.std(decision_tree_accuracies)

    mean_decision_stump_accuracy = np.mean(decision_stump_accuracies)
    std_decision_stump_accuracy = np.std(decision_stump_accuracies)

    mean_dt3_accuracy = np.mean(dt3_accuracies)
    std_dt3_accuracy = np.std(dt3_accuracies)

    # Return the statistics
    stats = np.array([[mean_decision_tree_accuracy, std_decision_tree_accuracy],
                      [mean_decision_stump_accuracy, std_decision_stump_accuracy],
                      [mean_dt3_accuracy, std_dt3_accuracy]])

    return stats

if __name__ == "__main__":
    stats = evaluatePerformance()
    print("Decision Tree Accuracy = {:.4f} ({:.4f})".format(stats[0, 0], stats[0, 1]))
    print("Decision Stump Accuracy = {:.4f} ({:.4f})".format(stats[1, 0], stats[1, 1]))
    print("3-level Decision Tree Accuracy = {:.4f} ({:.4f})".format(stats[2, 0], stats[2, 1]))
