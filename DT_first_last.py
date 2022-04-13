

import pandas as pd
import numpy as np
import seaborn as sns

from sklearn import datasets

from sklearn.model_selection import train_test_split

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
    
    def is_leaf(self):
        return self.value is not None
    

class DecisionTreeModel:

    def __init__(self, max_depth=100, criterion = 'gini', min_samples_split=2, impurity_stopping_threshold = 1):
        self.max_depth = max_depth
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.impurity_stopping_threshold = impurity_stopping_threshold
        self.root = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        # TODO
        # call the _fit method
        # 
        # end TODO
        print("Done fitting")

    def predict(self, X: pd.DataFrame):
        # TODO
        # call the predict method
        # return ...
        # end TODO
        pass
        
    def _fit(self, X: pd.DataFrame, y: pd.Series):
        self.root = self._build_tree(X, y)
        
    def _predict(self, X: pd.DataFrame):
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
                              
    def _build_tree(self, X: pd.DataFrame, y: pd.Series, depth=0):
        self.n_samples, self.n_features = X.shape
        self.n_class_labels = len(np.unique(y))

        # Check if this series has any none integer value (categorical values)
        y = self._change_if_categorical(y)

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

    def _gini(self, y:pd.Series):
        """
        Calculates the `gini` metric.
        It checks if the series is categorical or not, if it is then it will change the values of
        the series into numerical values from 0-n where `n` is the number of classes that feature has.
        """
        proportions = np.bincount(y) / len(y)
        gini = np.sum([p * (1 - p) for p in proportions if p > 0])
        return gini
    
    def _entropy(self, y: pd.Series):
        """
        Calculates the `entropy` of the split.
        It checks if the series is categorical or not, if it is then it will change the values of
        the series into numerical values from 0-n where `n` is the number of classes that feature has.
        """       
        # Check if this series has any none integer value (categorical values)
        if self._is_categorical(y):
            # If so, get the unique classes of that feature
            classes_label = np.unique(y)
            # Iterate through all the unique classes and apply to each value to check
            for i in range(0, len(classes_label)):
                # If its the current class label, if so we give it the index of the class label as a numerical value
                y = y.apply(lambda x: i if x == classes_label[i] else None)
        
        # change the data to be 0 to n unque values of y
        proportions = np.bincount(y) / len(y)
        entropy = -np.sum([p * np.log2(p) for p in proportions if p > 0])

        return entropy
        
    def _create_split(self, X: pd.DataFrame, thresh):
        left_idx = np.argwhere(X <= thresh).flatten()
        right_idx = np.argwhere(X > thresh).flatten()
        return left_idx, right_idx

    def _call_entropy_or_gini(self, y: pd.Series):
        """
        It will return either `self._entropy(y)` or `self._gini(y)` based on what the `criterion` is.
        """
        return self._entropy(y) if self.criterion == 'entropy' else self._gini(y)

    def _information_gain(self, X: pd.DataFrame, y: pd.Series, thresh):
        # It will get the either entropy of gini based on the criterion 
        parent_loss = self._call_entropy_or_gini(y)

        left_idx, right_idx = self._create_split(X, thresh)
        n, n_left, n_right = len(y), len(left_idx), len(right_idx)

        if n_left == 0 or n_right == 0: 
            return 0
        
        child_loss = (n_left / n) * self._call_entropy_or_gini(y[left_idx]) + (n_right / n) * self._call_entropy_or_gini(y[right_idx])

        return parent_loss - child_loss
       
    def _best_split(self, X: pd.DataFrame, y: pd.Series, features):
        '''TODO: add comments here

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
        '''TODO: add some comments here
        '''
        if node.is_leaf():
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)


class RandomForestModel(object):

    def __init__(self, n_estimators):
        # TODO:
        pass
        # end TODO

    def fit(self, X: pd.DataFrame, y: pd.Series):
        # TODO:
        pass
        # end TODO
        pass


    def predict(self, X: pd.DataFrame):
        # TODO:
        pass
        # end TODO

    

def accuracy_score(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

def classification_report(y_test, y_pred):
    # calculate precision, recall, f1-score
    # TODO:
    result = 'To be implemented'
    # end TODO
    return(result)

def confusion_matrix(y_test, y_pred):
    # return the 2x2 matrix
    # TODO:
    result = np.array([[0, 0], [0, 0]])
    # end TODO
    return(result)


def _test():
    
    df = pd.read_csv('breast_cancer.csv')
    
    #X = df.drop(['diagnosis'], axis=1).to_numpy()
    #y = df['diagnosis'].apply(lambda x: 0 if x == 'M' else 1).to_numpy()

    X = df.drop(['diagnosis'], axis=1)
    y = df['diagnosis'].apply(lambda x: 0 if x == 'M' else 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1
    )

    clf = DecisionTreeModel(max_depth=10)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("Accuracy:", acc)

if __name__ == "__main__":
    _test()
