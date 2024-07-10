# with open('model_pkl' , 'rb') as f:
#     lr = pickle.load(f)
# print("checvk")
# # check prediction

# lr.predict([[5000]]) # similar
# # return r2_square

import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Function to load the model from .pkl file
def load_model(model_filename):
    with open(model_filename, 'rb') as file:
        model = pickle.load(file)
    return model

# Example input data as key-value pairs (features)
input_data = [
    {'InvoiceId': 71111, 'CustomerName': 'Amit', 'NumberSites': 6, 'ServiceType': 'Internet','Month': 'May'},
    {'InvoiceId': 712111, 'CustomerName': 'kwe', 'NumberSites': 60, 'ServiceType': 'Voice','Month': 'May'},
    # Add more samples as needed
]

import pandas as pd
df = pd.DataFrame(input_data)
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
for col in categorical_cols:
    df[col] = label_encoder.fit_transform(df[col])
input_data = df.to_dict(orient='records')

# Load the pre-trained model from .pkl file
model_filename = 'Bill_model_pkl'
model = load_model(model_filename)

# Extract feature names (columns) from the input data (assuming they are the same for all samples)
features = list(input_data[0].keys())
print('features',features)
# Prepare input data as numpy array for prediction
input_values = np.array([[sample[feature] for feature in features] for sample in input_data])
print('input_values',input_values)

# Perform prediction using the loaded model
predictions = model.predict(input_values)

print('predictions',predictions)

# Create dictionary of sample_id -> prediction
predictions_dict = {i: predictions[i] for i in range(len(predictions))}

# Print the predictions as key-value pairs
for sample_id, prediction in predictions_dict.items():
    print(f"Sample {sample_id + 1}: Prediction = {prediction}")
