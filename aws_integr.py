from flask import Flask, jsonify
import pickle
import librosa
import numpy as np
import boto3
import base64
import io
import soundfile as sf
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load the trained model
model_path = r"C:\Users\youca\Downloads\logistic_regression_model (1).pkl"
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Define the original classes used during training
original_classes = ['normal', 'wear', 'cracks', 'burnt_wheels']  # Replace with your actual class labels

# Re-create the LabelEncoder and fit it with the original classes
label_encoder = LabelEncoder()
label_encoder.fit(original_classes)

# Function to extract MFCC features from audio data
def extract_mfcc_features(audio_data, sr, n_mfcc=13):
    mfcc = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc.T, axis=0)  # Compute mean of MFCCs across time
    return mfcc_mean.reshape(1, -1)

# Function to fetch audio data from DynamoDB
def fetch_audio_data_from_dynamodb(table_name):
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table(table_name)
    
    # Scan the table to retrieve all items
    response = table.scan()
    items = response['Items']
    
    return items

# Function to send anomaly data back to DynamoDB
def send_anomaly_data_to_dynamodb(table_name, gps_latitude, gps_longitude, anomaly_type):
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table(table_name)
    
    # Prepare the item to be inserted
    item = {
        'gps_latitude': str(gps_latitude),   # DynamoDB stores numbers as strings
        'gps_longitude': str(gps_longitude),
        'anomaly_type': anomaly_type
    }
    
    # Insert the item into the table
    table.put_item(Item=item)

# API endpoint to process audio data from DynamoDB, predict anomalies, and store results back
@app.route('/process_and_store_dynamodb_audio', methods=['GET'])
def process_and_store_dynamodb_audio():
    print("Fetching data from DynamoDB...")
    audio_records = fetch_audio_data_from_dynamodb('Recored_data')
    print(f"Fetched {len(audio_records)} records.")
    
    anomalies = []

    for record in audio_records:
        gps_latitude = record['latitude']
        gps_longitude = record['longitude']
        audio_base64 = record['audio_file']  # Assuming audio data is stored as Base64 string

        # Decode the Base64 string back to binary audio data
        audio_blob = base64.b64decode(audio_base64)

        # Load audio data
        audio_data, sr = sf.read(io.BytesIO(audio_blob))

        # Extract MFCC features
        mfcc_features = extract_mfcc_features(audio_data, sr)

        # Predict using the trained model
        prediction = model.predict(mfcc_features)

        # Reverse transform the prediction to get the original class label
        original_label = label_encoder.inverse_transform(prediction)[0]

        # Check if an anomaly was detected
        if original_label in ['wear', 'cracks', 'burnt_wheels']:  # Adjust based on your actual labels
            anomaly_data = {
                "latitude": gps_latitude,
                "longitude": gps_longitude,
                "anomaly_type": original_label
            }
            anomalies.append(anomaly_data)

            # Send the anomaly data to DynamoDB
            send_anomaly_data_to_dynamodb('Final_data', gps_latitude, gps_longitude, original_label)

    print("Anomalies detected and sent to DynamoDB:", anomalies)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

