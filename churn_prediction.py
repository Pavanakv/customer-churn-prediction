# Step 1: Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
            
# Step 2: Load your dataset
data = pd.read_csv('C:/Users/Lenovo/Desktop/churn_data.csv')

# Step 3: Data Preprocessing
# Remove rows with missing values
data = data.dropna()

# Convert categorical columns to numeric (dummy encoding)
data = pd.get_dummies(data, drop_first=True)

# Step 4: Split the data into features (X) and target (y)
X = data.drop('Churn_Yes', axis=1)  # Features (everything except 'Churn_Yes')
y = data['Churn_Yes']  # Target (what we want to predict)

# Step 5: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Create and train the RandomForest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 7: Make predictions on the test set
y_pred = model.predict(X_test)
# Step 8: Plot the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Step 9: Plot Feature Importance
importances = model.feature_importances_
features = X.columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=False)

# Plot the feature importance
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Feature Importance')
plt.show()



