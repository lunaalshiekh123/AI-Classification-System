"""
Project: The Human AI Classifier
Description: Visualizing Machine Learning model performance (KNN, SVM, Decision Tree, Random Forest)
using a creative 'Human-like' plot where body parts represent key metrics.
"""

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Patch
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

# 1. Load Data
data = load_iris()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 2. Define Models for comparison
models = {
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}

# 3. Dynamic Coloring based on Metric Value
def get_color_for_metric(value, metric):
    if metric == 'precision':  # Head: Yellow -> Red
        return (value, 1-value, 0)
    elif metric == 'recall':    # Hands: Blue -> Black
        return (0, 0, 1-value)
    elif metric == 'f1':        # Legs: Orange -> Green
        return (1, 0.5*value, 0)
    elif metric == 'accuracy':  # Body: Purple -> Pink
        return (0.5*value, 0, 0.5+0.5*value)

def plot_colored_line(ax, x, y, color):
    points = np.array([x, y]).T.reshape(-1,1,2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, colors=[color]*len(segments), linewidths=6)
    ax.add_collection(lc)

# 4. Drawing the 'Human' Performance Character
def draw_human(ax, metrics, title):
    # Head = Precision
    head_color = get_color_for_metric(metrics['precision'], 'precision')
    head = plt.Circle((0.5, 0.9), 0.05, color=head_color)
    ax.add_patch(head)

    # Body = Accuracy
    plot_colored_line(ax, [0.5, 0.5], [0.8, 0.6], get_color_for_metric(metrics['accuracy'], 'accuracy'))

    # Arms = Recall
    plot_colored_line(ax, [0.5, 0.4], [0.75, 0.7], get_color_for_metric(metrics['recall'], 'recall'))
    plot_colored_line(ax, [0.5, 0.6], [0.75, 0.7], get_color_for_metric(metrics['recall'], 'recall'))

    # Legs = F1-score
    plot_colored_line(ax, [0.5, 0.45], [0.6, 0.5], get_color_for_metric(metrics['f1'], 'f1'))
    plot_colored_line(ax, [0.5, 0.55], [0.6, 0.5], get_color_for_metric(metrics['f1'], 'f1'))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title(title, fontsize=12, fontweight='bold')

# 5. Calculate Metrics
metrics_dict = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics_dict[name] = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='macro'),
        'recall': recall_score(y_test, y_pred, average='macro'),
        'f1': f1_score(y_test, y_pred, average='macro')
    }

# 6. Final Visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 12))
for ax, (name, metric) in zip(axes.flatten(), metrics_dict.items()):
    draw_human(ax, metric, name)

plt.suptitle("Human AI Performance Classifier", fontsize=16)
plt.show()
