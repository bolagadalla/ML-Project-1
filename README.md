<h1 style='color:yellow'><b>CSCI-381 ML Project 1</b></h1> 

Implementing a Decision Tree and Random Forest from scratch in python.

<br>
<h2 style='color:lightgreen'>Description</h2>
<br>

In the `DT_orig.py` file, it contains the code from the article [Implementing a Decision Tree From Scratch](https://towardsdatascience.com/implementing-a-decision-tree-from-scratch-f5358ff9c4bb) which is a simple implementation of a <b style='color:red'>Decision Tree</b>. However, in that article the author only implements the Decision Tree based on the criterion `entropy` and excluding `gini` implmentation. As well as the Tree only accepts numpy arrays and numerical values only. 

In this project I implemented the following:

1. The `gini` criterion
2. Uses `pandas.DataFrame` instead of numpy array
3. The target value can be `categorical` input.
4. Random Forest Ensemble