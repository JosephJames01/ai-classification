import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load and preprocess the dataset
df = pd.read_csv('C:/Users/joeha/OneDrive/Documents/PythonScripts/ai/garments_worker_productivity.csv', 
                 parse_dates=['date'])
df['wip'] = df['wip'].fillna(0)
df['date'] = df['date'].dt.dayofyear
df['quarter'] = df['quarter'].map({'Quarter1': 1, 'Quarter2': 2, 'Quarter3': 3, 'Quarter4': 4, 
                                   'Quarter5': 5})
df['day'] = df['day'].map({'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 'Friday': 5,
                            'Saturday': 6, 'Sunday': 7})
df['department'] = df['department'].map({'sewing': 1, 'finishing': 2, 
                                         'finishing ': 2})
numerical_features = ['date', 'quarter', 'team', 'smv', 'wip', 'over_time',
                       'incentive', 'idle_time', 'idle_men', 'no_of_style_change', 
                       'no_of_workers', 'department']
#normalise 
df[numerical_features] = (df[numerical_features] - df[numerical_features].min()) / (df[numerical_features].max() - df[numerical_features].min())
#binary classification
df['actual_productivity'] = (df['actual_productivity'] >= 0.75).astype(int)

# Preparing input and output
inputs = df[numerical_features].values
outputs = df['actual_productivity'].values.reshape(-1, 1)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Neural network structure
input_neurons = inputs.shape[1]
hidden_neurons = 20
output_neurons = 1

weights_0 = 2* np.random.rand(input_neurons, hidden_neurons) - 1
weights_1 = 2* np.random.rand(hidden_neurons, output_neurons) - 1


learning_rate = 0.002
epochs = 1000000
error_history = []
precision = 1 
recall = 1
tp=1
fp=1
fn = 1

for epoch in range(epochs):
    # Forward propagation
    hidden_layer_input = np.dot(inputs, weights_0) 
    hidden_layer_output = sigmoid(hidden_layer_input)
    output_layer_input = np.dot(hidden_layer_output, weights_1) 
    predicted_output = sigmoid(output_layer_input)
    error = outputs - predicted_output
    

    if epoch % 1000 == 0:
        loss = np.mean(abs(error))
        print(f'Epoch: {epoch}, Loss: {loss}')
        error_history.append(loss)

    # Backpropagation
    weights_1T = weights_1.T
    d_predicted_output = error * sigmoid_derivative(predicted_output)
    error_hidden_layer = d_predicted_output.dot(weights_1T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)
    
    # Updating Weights 
    weights_1 += hidden_layer_output.T.dot(d_predicted_output) * learning_rate
    weights_0 += inputs.T.dot(d_hidden_layer) * learning_rate
    
tp = np.sum((predicted_output >= 0.75) & (outputs >= 0.75))
fp = np.sum((predicted_output >= 0.75) & (outputs < 0.75 ))
fn = np.sum((predicted_output < 0.75) & (outputs >= 0.75))
tn = np.sum((predicted_output < 0.75) & (outputs < 0.75))
   
precision = tp/(tp+fp)
recall = tp/(tp +fn)
accuracy = (tp+ tn)/(tp+fp+fn+tn)   
    
f1_score = 2*(precision*recall)/(precision+ recall)  
print('f1_score:')
print(f1_score)
print('precision: ', (precision))
print('recall ', (recall))
print('accuracy', (accuracy) )

# Plotting the error over epochs
plt.plot([x * 1000 for x in range(len(error_history))], error_history)
plt.xlabel('Epoch (x1000)')
plt.ylabel('Loss')
plt.title('Loss over Epochs')
plt.show()
