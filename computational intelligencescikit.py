import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score as calc_f1_score, precision_score, recall_score, roc_auc_score
import matplotlib.pyplot as plt

# Assuming your data is stored in the 'df' variable
df = pd.read_csv('C:/Users/joeha/OneDrive/Documents/PythonScripts/garments_worker_productivity.csv', parse_dates=['date'])
df['wip'] = df['wip'].fillna(0)
df['date'] = df['date'].dt.dayofyear
df['quarter'] = df['quarter'].map({'Quarter1': 1, 'Quarter2': 2, 'Quarter3': 3, 'Quarter4': 4, 'Quarter5': 5})
df['day'] = df['day'].map({'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 'Friday': 5, 'Saturday': 6, 'Sunday': 7})
df['department'] = df['department'].map({'sewing': 1, 'finishing': 2, 'finishing ': 2})
numerical_features = ['date', 'quarter', 'team', 'smv', 'wip', 'over_time', 'incentive', 'idle_time', 'idle_men', 'no_of_style_change', 'no_of_workers', 'department']

# Normalize the numerical features
df[numerical_features] = (df[numerical_features] - df[numerical_features].min()) / (df[numerical_features].max() - df[numerical_features].min())

# Binary classification
df['actual_productivity'] = (df['actual_productivity'] > 0.75).astype(int)

# Prepare input and output
inputs = df[numerical_features].values
outputs = df['actual_productivity'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(inputs, outputs, test_size=0.2, random_state=42)

# Scale the input features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize the MLPClassifier with warm_start=True
clf = MLPClassifier(hidden_layer_sizes=(50, 25), max_iter=1, random_state=42, warm_start=True)
errors = []
# Train the model for 10,000 epochs
for epoch in range(3000):
    clf.partial_fit(X_train, y_train, classes=np.unique(y_train))
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    loss = clf.loss_
    errors.append(loss)
    # Calculate metrics
    f1 = calc_f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    roc_score = roc_auc_score(y_test, y_pred_proba)

    # Print the results (optional: you might want to print every n epochs)
    if epoch % 1000 == 0:
        print(f"Epoch: {epoch+1}, Accuracy: {clf.score(X_test, y_test):.4f}, F1 Score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, ROC AUC: {roc_score:.4f}")
   
    num =  len(errors)+1

    if epoch >2990:
     plt.plot(range(1, num), errors)
     plt.xlabel('Epoch')
     plt.ylabel('Error')
    
     plt.title('Error over Epochs')
     plt.show()

# Print the final error achieved
final_error = clf.loss_
print(f'Final training error: {final_error:.4f}')
