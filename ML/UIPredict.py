import pandas as pd
import pickle

# Sample input data for prediction
input_data = [
    {'InvoiceId': 71111, 'CustomerName':'Planet Fitness','NumberSites': 10, 'ServiceType': 'Internet'},
    # {'InvoiceId': 71111, 'NumberSites': 6, 'ServiceType': 'Internet', 'Month': 'May'},
    # Add more samples as needed
]

# Convert input data to DataFrame
df = pd.DataFrame(input_data)

# Load the preprocessor and the model
with open('preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

with open('Bill_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Transform the input data using the loaded preprocessor
input_feature_test_arr = preprocessor.transform(df)

# Make predictions using the loaded model
predictions = model.predict(input_feature_test_arr)
print('predictions:', predictions[0])

