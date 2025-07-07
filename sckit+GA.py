import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from sklearn_genetic import GASearchCV
from sklearn_genetic.space import Categorical, Continuous

# === 1. Load and preprocess data ===
df = pd.read_csv('C:/Users/joeha/OneDrive/Documents/PythonScripts/ai/garments_worker_productivity.csv', 
                 parse_dates=['date'])
df['wip'] = df['wip'].fillna(0)
df['date'] = df['date'].dt.dayofyear
df['quarter'] = df['quarter'].map({'Quarter1': 1, 'Quarter2': 2,
                                   'Quarter3': 3, 'Quarter4': 4, 'Quarter5': 5})
df['day'] = df['day'].map({'Monday': 1, 'Tuesday': 2, 
                           'Wednesday': 3, 'Thursday': 4, 'Friday': 5, 'Saturday': 6, 'Sunday': 7})
df['department'] = df['department'].map({'sewing': 1, 
                                         'finishing': 2, 'finishing ': 2})

numerical_features = [
    'date', 'quarter', 'team', 'smv', 'wip',
    'over_time', 'incentive', 'idle_time', 'idle_men', 
    'no_of_style_change', 'no_of_workers', 'department'
]

# Normalize the numerical features
df[numerical_features] = (
    df[numerical_features] - df[numerical_features].min()
) / (df[numerical_features].max() - df[numerical_features].min())

# Binary classification
df['actual_productivity'] = (df['actual_productivity'] > 0.75).astype(int)

# Prepare input and output
inputs = df[numerical_features].values
outputs = df['actual_productivity'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    inputs, outputs, test_size=0.2, random_state=42
)

# Scale the input features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# === 2. Set up the genetic algorithm search space ===
param_grid = {
    'hidden_layer_sizes': Categorical([
        (25,), (50,), (100,), (50, 25), (100, 50)
    ]),
    'activation': Categorical(['relu', 'tanh', 'logistic']),
    'solver': Categorical(['adam', 'sgd']),
    'learning_rate_init': Continuous(0.0001, 0.1),
    'alpha': Continuous(0.0001, 0.1)
}

# === 3. Initialize the MLPClassifier ===
mlp = MLPClassifier(max_iter=3000, random_state=42)

# === 4. Run the genetic algorithm ===
evolved_search = GASearchCV(
    estimator=mlp,
    param_grid=param_grid,
    scoring='f1',
    population_size=20,
    generations=8,
    n_jobs=-1,
    cv=3,
    verbose=True,
    keep_top_k=4
)

print("Starting genetic algorithm hyperparameter search...")
evolved_search.fit(X_train, y_train)
print("Genetic algorithm search completed.\n")

# === 5. Evaluate the optimized model ===
best_model = evolved_search.best_estimator_
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
roc_score = roc_auc_score(y_test, y_pred_proba)

print("=== Optimized Model Performance ===")
print(f"F1 Score: {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"ROC AUC: {roc_score:.4f}")
print("\nBest Hyperparameters found:")
print(evolved_search.best_params_)

# === 6. Plot the convergence of the genetic algorithm ===
plt.figure(figsize=(8, 5))
plt.plot(evolved_search.history['fitness'], marker='o')
plt.title('Genetic Algorithm Fitness (F1 Score) Over Generations')
plt.xlabel('Generation')
plt.ylabel('Best F1 Score')
plt.grid(True)
plt.tight_layout()
plt.show()
