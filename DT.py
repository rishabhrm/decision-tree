import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Sidebar options for hyperparameters
st.sidebar.title("Decision Tree Hyperparameters")
criterion = st.sidebar.selectbox("Criterion", ("gini", "entropy"))
splitter = st.sidebar.selectbox("Splitter", ("best", "random"))
min_samples_split = st.sidebar.slider("Min Samples Split", 2, 10, 2)
min_samples_leaf = st.sidebar.slider("Min Samples Leaf", 1, 5, 1)
max_depth = st.sidebar.slider("Max Depth", 1, 10, 3)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Build Decision Tree model
clf = DecisionTreeClassifier(
    criterion=criterion,
    splitter=splitter,
    min_samples_split=min_samples_split,
    min_samples_leaf=min_samples_leaf,
    max_depth=max_depth,
    random_state=42
)
clf.fit(X_train, y_train)

# Predict and calculate accuracy
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Display decision tree
st.subheader("Decision Tree Visualization")
plt.figure(figsize=(15, 10))
plot_tree(clf, filled=True, feature_names=iris.feature_names, class_names=iris.target_names)
st.pyplot(plt.gcf())  # Streamlit renders matplotlib figure

# Display accuracy
st.subheader("Model Performance")
st.write(f"Accuracy: {accuracy * 100:.2f}%")

st.subheader("Inference on Hyperparameters")
st.write("""
- **Criterion**: The function to measure the quality of a split. `'gini'` is the Gini Impurity, and `'entropy'` is the Information Gain. 
   - Gini tries to minimize the chances of misclassification, while entropy measures the disorder of the data.
- **Splitter**: The strategy used to split at each node. `'best'` chooses the best split, while `'random'` chooses a random split.
   - `'best'` usually provides better results, but `'random'` can work faster for large datasets.
- **Min Samples Split**: The minimum number of samples required to split an internal node. Higher values make the model more conservative, avoiding overfitting but possibly underfitting.
   - Lower values allow the tree to grow deeper, which can lead to overfitting.
- **Min Samples Leaf**: The minimum number of samples required to be at a leaf node. Increasing this will smooth the model, making it less likely to overfit.
- **Max Depth**: The maximum depth of the tree. A deeper tree can model more complex patterns but may overfit, while a shallower tree generalizes better.
""")

st.subheader("Understanding Overfitting and Underfitting")
st.write("""
- **Underfitting**: When both training and validation accuracies are low, the model is too simple to capture the data's underlying structure (e.g., when the tree depth is too small).
- **Overfitting**: When training accuracy is high but validation accuracy drops, the model is too complex (e.g., when the tree depth is too large).
- **Balanced Model**: The best performance is when both accuracies are relatively high, indicating the model generalizes well to new data.
""")

# Display scatter plot with decision boundary
st.subheader("Scatter Plot with Decision Boundaries")

# Choose two features for scatter plot visualization
feature_x = st.selectbox("Select X-axis feature", iris.feature_names, index=0)
feature_y = st.selectbox("Select Y-axis feature", iris.feature_names, index=1)

# Get the indices of the selected features
feature_x_index = iris.feature_names.index(feature_x)
feature_y_index = iris.feature_names.index(feature_y)

# Reduce the data to the two selected features
X_train_2d = X_train[:, [feature_x_index, feature_y_index]]
X_test_2d = X_test[:, [feature_x_index, feature_y_index]]

# Train the classifier again with just the 2 selected features
clf_2d = DecisionTreeClassifier(
    criterion=criterion,
    splitter=splitter,
    min_samples_split=min_samples_split,
    min_samples_leaf=min_samples_leaf,
    max_depth=max_depth,
    random_state=42
)
clf_2d.fit(X_train_2d, y_train)

# Create a mesh grid to plot decision boundaries
x_min, x_max = X_train_2d[:, 0].min() - 1, X_train_2d[:, 0].max() + 1
y_min, y_max = X_train_2d[:, 1].min() - 1, X_train_2d[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))

# Predict over the mesh grid
Z = clf_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot decision boundary
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdYlBu)

# Scatter plot of the selected features from the training set
for i, color in zip(range(len(iris.target_names)), ['red', 'blue', 'green']):
    plt.scatter(X_train_2d[y_train == i, 0], X_train_2d[y_train == i, 1], 
                color=color, label=iris.target_names[i])

plt.xlabel(feature_x)
plt.ylabel(feature_y)
plt.legend(loc="upper right")
plt.title("Decision Boundary and Training Data Scatter Plot")
st.pyplot(plt.gcf())
