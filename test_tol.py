from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Load dataset
X, y = load_iris(return_X_y=True)
clf = RandomForestClassifier(random_state=0)

# Create SequentialFeatureSelector with tol
sfs = SequentialFeatureSelector(clf, k_features=2, forward=True, tol=0.01)
sfs = sfs.fit(X, y)

# Print the selected features to verify
print("Selected features:", sfs.k_feature_names_)
print("SFS finished successfully with tol=0.01")
