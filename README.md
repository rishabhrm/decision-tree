# Decision Tree Visualizer

## Project Description

The **Decision Tree Visualizer** is a web application built using Streamlit that allows users to explore the behavior of Decision Tree classifiers on the Iris dataset. Users can interactively modify hyperparameters such as the criterion, splitter, min samples split, min samples leaf, and max depth to see how these changes affect the model's performance, decision tree structure, and decision boundaries.

This application aims to help users understand concepts like overfitting, underfitting, and model performance visually and intuitively.

### Features
- **Interactive Hyperparameter Tuning**: Change hyperparameters in real-time and see the impact on the model.
- **Decision Tree Visualization**: Visualize the structure of the decision tree.
- **Model Performance Metrics**: Display accuracy scores based on user-defined parameters.
- **Scatter Plot with Decision Boundaries**: Visualize how the decision tree classifies the feature space.

## Concepts Covered

### Decision Trees
Decision Trees are a type of supervised learning algorithm used for classification and regression tasks. They work by recursively splitting the data into subsets based on the value of input features.

#### Key Hyperparameters:
- **Criterion**: Function to measure the quality of a split (Gini impurity or entropy).
- **Splitter**: Strategy for splitting at each node (best or random).
- **Min Samples Split**: Minimum number of samples required to split an internal node.
- **Min Samples Leaf**: Minimum number of samples required to be at a leaf node.
- **Max Depth**: Maximum depth of the tree.

### Overfitting and Underfitting
- **Overfitting**: When a model learns noise in the training data instead of the underlying pattern, resulting in high accuracy on training data but poor performance on unseen data.
- **Underfitting**: When a model is too simple to capture the underlying pattern of the data, leading to low accuracy on both training and validation data.


## Screenshots




Decision Tree Visualization
![image](https://github.com/user-attachments/assets/9240a87e-3e42-4a2f-86cd-3ed3aedc23c7)

Scatter Plot with Decision Boundaries
![image](https://github.com/user-attachments/assets/a1516beb-a93b-4ade-b025-1413dcbf5bb4)
