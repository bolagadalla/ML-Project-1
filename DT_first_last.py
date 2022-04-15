

from collections import Counter
import random
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split

class Node:
    '''
    Node class that would hold data for the Decision tree. Data like: `features`, `threshold`, `left` for left nodes, `right` for right nodes, `value` and the value at which it splits
    '''
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
    
    def is_leaf(self):
        return self.value is not None
    

class DecisionTreeModel:
    '''
    Decision Tree Class which would build a model.

    ## Input:
    `max_depth` is the max height of the tree.
    `criterion` is the type of criterion to measure the `information gain`.
    `min_samples_split` how many splits are we going to perform
    `impurity_stopping_threshold` for graduate students
    '''
    def __init__(self, max_depth=100, criterion = 'gini', min_samples_split=2, impurity_stopping_threshold = 1):
        self.max_depth = max_depth
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.impurity_stopping_threshold = impurity_stopping_threshold
        self.root = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        # Will change y to numpy array later on the code
        self._fit(X.to_numpy(), y) 
        # print("Done fitting")

    def predict(self, X: pd.DataFrame):
        '''
        Takes in an `X` Dataframe and returns an array of prediction for each point.

        ## Input:
        `X` **Dataframe** type.

        ## Return:
        `np.ndarray` which will hold the predictions
        '''
        # Returns an array of predictions
        return self._predict(X.to_numpy())
        
    def _fit(self, X: np.ndarray, y: pd.Series):
        # Check if this series has any none integer value (categorical values)
        y = self._change_if_categorical(y).to_numpy() # changes it into numpy array
        self.root = self._build_tree(X, y)
        
    def _predict(self, X: np.ndarray):
        # Triverse the tree and return the values into an array
        predictions = [self._traverse_tree(x, self.root) for x in X]
        return np.array(predictions)
        
    def _is_finished(self, depth):
        # TODO: for graduate students only, add another stopping criteria
        # modify the signature of the method if needed
        if (depth >= self.max_depth
            or self.n_class_labels == 1
            or self.n_samples < self.min_samples_split):
            return True
        # end TODO
        return False
    
    def _is_homogenous_enough(self):
        # TODO: for graduate students only
        result = False
        # end TODO
        return result
                              
    def _build_tree(self, X: np.ndarray, y: pd.Series, depth=0):
        '''
        Builds a Decision Tree by splitting the date at the point where its the best information gain using the inputted `criterion`.
        '''

        self.n_samples, self.n_features = X.shape
        self.n_class_labels = len(np.unique(y))
        
        # stopping criteria
        if self._is_finished(depth):
            most_common_Label = np.argmax(np.bincount(y))
            return Node(value=most_common_Label)

        # get best split
        rnd_feats = np.random.choice(self.n_features, self.n_features, replace=False)
        best_feat, best_thresh = self._best_split(X, y, rnd_feats)

        # grow children recursively
        left_idx, right_idx = self._create_split(X[:, best_feat], best_thresh)
        left_child = self._build_tree(X[left_idx, :], y[left_idx], depth + 1)
        right_child = self._build_tree(X[right_idx, :], y[right_idx], depth + 1)
        return Node(best_feat, best_thresh, left_child, right_child)

    def _change_if_categorical(self, y: pd.Series):
        '''
        Gets all the labesl of `y` and loops throught them and gives them an Ordinal Encoding from 0-n where `n` is the length of unique labels.
        '''
        if self._is_categorical(y):
            # If so, get the unique classes of that feature
            classes_label = np.unique(y)
            # Iterate through all the unique classes and apply to each value to check
            for i in range(0, len(classes_label)):
                # If its the current class label, if so we give it the index of the class label as a numerical value
                y = y.apply(lambda x: i if x == classes_label[i] else None)
        return y
    
    def _is_categorical(self, y: pd.Series) -> bool:
        """
        Loops through the `y` series and if it hits any value that its not an integer then it will
        return `True` as in that series is categorical, otherwise it will return `False` once it 
        loops through all the values of `y` and all values were `int`.

        **Assuming the values are not `int` that in `str` form** Otherwise it will consider them as categorical.
        """
        for _, val in y.iteritems():
            if type(val) != int:
                return True
        return False

    def _gini(self, y:np.ndarray):
        """
        Calculates the `gini` metric.
        It already changed values to numeric values if it was categorical input using Ordinal Encoding.
        """
        # The y will always be represented as numerical value since we changed it in `_build_tree()`
        proportions = np.bincount(y) / len(y)
        gini = np.sum([p * (1 - p) for p in proportions if p > 0])
        return gini
    
    def _entropy(self, y: pd.Series):
        """
        Calculates the `entropy` of the split.
        It checks if the series is categorical or not, if it is then it will change the values of
        the series into numerical values from 0-n where `n` is the number of classes that feature has.
        """               
        # The y will always be represented as numerical value since we changed it in `_build_tree()`
        proportions = np.bincount(y) / len(y)
        entropy = -np.sum([p * np.log2(p) for p in proportions if p > 0])

        return entropy
        
    def _create_split(self, X: np.ndarray, thresh):
        # Creates a split and assign the values where X is less then or equal tot the threshold to the left, and to the right the rest.
        left_idx = np.argwhere(X <= thresh).flatten()
        right_idx = np.argwhere(X > thresh).flatten()
        # Returns the left and right indexes of the input `X`
        return left_idx, right_idx

    def _call_entropy_or_gini(self, y: pd.Series):
        """
        It will return either `self._entropy(y)` or `self._gini(y)` based on what the `criterion` is.
        """
        return self._entropy(y) if self.criterion == 'entropy' else self._gini(y)

    def _information_gain(self, X: np.ndarray, y: np.ndarray, thresh):
        # It will get the either entropy of gini based on the criterion 
        parent_loss = self._call_entropy_or_gini(y)

        # Gets all the indexes of the data that would be dealt with on the left side of the tree and the right side of the tree
        left_idx, right_idx = self._create_split(X, thresh)
        n, n_left, n_right = len(y), len(left_idx), len(right_idx)

        if n_left == 0 or n_right == 0: 
            return 0
        
        child_loss = (n_left / n) * self._call_entropy_or_gini(y[left_idx]) + (n_right / n) * self._call_entropy_or_gini(y[right_idx])

        return parent_loss - child_loss
       
    def _best_split(self, X: np.ndarray, y: np.ndarray, features):
        '''
        It will loop through all the features one by one. And at each feature it will get all the 
        column of that feature and creates an array of all the unique values in that colnmn. Then, 
        looping through each unique value and tries to calculate the information gain score if we 
        split it at that point and storing it if the `score` was higher then the previous value score.

        ## Returns:
        split feature and split threshold as tuple
        '''
        split = {'score':- 1, 'feat': None, 'thresh': None}
        for feat in features:
            X_feat = X[:, feat]
            thresholds = np.unique(X_feat)
            for thresh in thresholds:
                score = self._information_gain(X_feat, y, thresh)

                if score > split['score']:
                    split['score'] = score
                    split['feat'] = feat
                    split['thresh'] = thresh

        return split['feat'], split['thresh']
    
    def _traverse_tree(self, x, node):
        '''
        It will triverse through the tree starting at the `root` node. Then it will check the current
        value of `x` to see if its less then or equal to the node threshold. If it is, it will start
        to triverse the left node. Otherwise, it will triverse the right node. It will do this until
        it reaches a leaf node which will return the value of that last node.
        '''
        if node.is_leaf():
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)


class RandomForestModel(object):

    def __init__(self, n_estimators: int, max_depth=100, criterion = 'gini', min_samples_split=2, impurity_stopping_threshold = 1):
        '''
        It creates an array of Decision Trees passing in the required parameters that was also passed into this class. 
        It will create `n_estimators` amount of trees, and for each tree when its *fitted* the input data is shuffled.

        ## Input:
        `n_estimators`: int - The amount of trees ot create 
        `max_depth` = 100 - The max depth of each tree.
        `criterion` = 'gini' - This is the criterion which will be used to split the data.
        '''
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.impurity_stopping_threshold = impurity_stopping_threshold
        # The trees in the forest
        self.forest = []

    def fit(self, X: pd.DataFrame, y: pd.Series):
        for i in range(0, self.n_estimators):
            # Shuffles the data
            idxs = np.random.choice(len(y), replace=True, size=len(y)) 
            # Creates a tree
            tree = DecisionTreeModel(max_depth=self.max_depth, criterion=self.criterion, min_samples_split=self.min_samples_split, impurity_stopping_threshold=self.impurity_stopping_threshold)
            # Fit the tree while passing in the shuffled indexes
            tree.fit(X.iloc[idxs], y.iloc[idxs])
            # Append the tree to the array of `forests`
            self.forest.append(tree)

    def _common_result(self, values:list):
        '''
        It will look at each column of the `values` (since it will be a 2D array where each value of the array is the prediction of a tree in the forest)
        and pick the most common, the prediction that is most predicted. Then return all the results as a 1D array.
        '''
        return np.array([Counter(col).most_common(1)[0][0] for col in zip(*values)])

    def predict(self, X: pd.DataFrame):
        '''
        Will loop through each tree in the `forest` and call the `predict(X)` method on them, and return the result as its own array into another array.
        This will be a result of 2D array where each value of the array is a prediction of a tree.
        '''
        tree_values = []
        for tree in self.forest:
            tree_values.append(tree.predict(X))
        return self._common_result(tree_values)
        
def accuracy_score(y_true, y_pred):
    '''Calcualtes the accuracy score of a model using its `y_pred` values.'''
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

def _count_occurence(y_pred):
    '''
    It counts the occurence of `0` and `1` in the predicted array and return it as a `splitted list`
    '''
    one, zero = 0, 0
    
    for i in y_pred:
        if i == 1:
            one += 1
        else:
            zero += 1
    
    return zero, one

def classification_report(y_test, y_pred):
    '''
    It will calculate precision, recall, f1-score, support, accuracy, macro avg, and weighted avg
    and then display it into a matrix form.
    '''
    # calculate precision, recall, f1-score
    top, bottom = confusion_matrix(y_test, y_pred)
    tp, fn = top
    fp, tn = bottom

    precision0 = (tn / (tn + fn))
    precision1 = (tp / (tp + fp))

    recall0 = (tn / (tn + fp))
    recall1 = (tp / (tp + fn))

    f1_score0 = 2 * (recall0 * precision0) / (recall0 + precision0)
    f1_score1 = 2 * (recall1 * precision1) / (recall1 + precision1)

    support0, support1 = _count_occurence(y_pred)

    result = f'''                    precision    recall    f1-score    support
    0               {"%.2f" % round(precision0, 2)}        {"%.2f" % round(recall0, 2)}        {"%.2f" % round(f1_score0, 2)}        {support0}
    1               {"%.2f" % round(precision1, 2)}        {"%.2f" % round(recall1, 2)}        {"%.2f" % round(f1_score1, 2)}        {support1}
    accuracy                                {"%.2f" % round(accuracy_score(y_test, y_pred), 2)}        {support0 + support1}
    macro avg       {"%.2f" % round((precision0 + precision1) / 2, 2)}        {"%.2f" % round((recall0 + recall1) / 2, 2)}        {"%.2f" % round((f1_score0 + f1_score1) / 2, 2)}        {support0 + support1}
    weighted avg    {"%.2f" % round(((precision0 * support0) + (precision1 * support1)) / (support0 + support1), 2)}        {"%.2f" % round(((recall0 * support0) + (recall1 * support1)) / (support0 + support1), 2)}        {"%.2f" % round(((f1_score0 * support0) + (f1_score1 * support1)) / (support0 + support1), 2)}        {support0 + support1}
    '''
    return result

def confusion_matrix(y_test: pd.Series, y_pred: np.ndarray):
    '''
    It creates a confusion matrix where it shows you the true positive, false negative, false positive, and true negative

    ## Returns
    `2D array` which will looks like this:

                      Predicted Value
                        +---------+
                        | tp | fn |
            Actual Value|----|----|
                        | fp | tn |
                        +---------+
    '''
    # return the 2x2 matrix
    tp, fn, fp, tn = 0, 0, 0, 0
    # loops through the prediction series
    for i, val in y_test.reset_index(drop=True).iteritems():
        # Compares the current value of the prediction to the y_test at this index
        p_val = y_pred[i]
        if val == 1 and p_val == 1:
            tp += 1
        elif val == 1 and p_val == 0:
            fn += 1
        elif val == 0 and p_val == 1:
            fp += 1
        else:
            tn += 1
    result = np.array([[tp, fn], [fp, tn]])
    return(result)

def _test():
    
    df = pd.read_csv('breast_cancer.csv')
    
    #X = df.drop(['diagnosis'], axis=1).to_numpy()
    #y = df['diagnosis'].apply(lambda x: 0 if x == 'M' else 1).to_numpy()

    X = df.drop(['diagnosis'], axis=1)
    y = df['diagnosis'].apply(lambda x: 0 if x == 'M' else 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    clf = DecisionTreeModel(max_depth=10)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("Accuracy:", acc)

if __name__ == "__main__":
    _test()
