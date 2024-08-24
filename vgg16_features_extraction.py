import os
import numpy as np
import pandas as pd
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.vgg16 import preprocess_input

# Set the current working directory and path to the data
current_directory = os.getcwd()
data_path_1 = os.path.join(current_directory, "PCNA")

# Define the image size compatible with VGG16
image_size = (224, 224)

# Load the VGG16 model with pre-trained weights, excluding the fully connected layers
model = VGG16(weights="imagenet", include_top=False)

def extract_features(image_path):
    # Load, resize, and preprocess the image
    img = load_img(image_path, target_size=image_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Pass the image through the truncated VGG16 model
    features = model.predict(img_array)

    # Select features from layers 1 to 5
    features_list = []
    for i in range(1, 6):
        features_list.append(features[0, ..., i-1:i])

    # Concatenate features from layers
    all_features = np.concatenate(features_list, axis=1)

    # Return the flattened feature vector
    return all_features.flatten()

all_features_1 = []
all_image_names = []

# Loop through all files in the directory
for filename in os.listdir(data_path_1):
    image_path = os.path.join(data_path_1, filename)
    features = extract_features(image_path)
    all_features_1.append(features)
    all_image_names.append(os.path.basename(image_path))

all_features_array = np.array(all_features_1)

# Create a pandas DataFrame from the NumPy array and add a column with image names
df = pd.DataFrame(all_features_array)
df.insert(0, "image_name", all_image_names)

# Generate feature names
feature_name = []
count = 1
block_names = ["block1_conv1", "block1_conv2", "block2_conv1", "block2_conv2", "block3_conv1"]
data_types = ["PCNA"] * 5
block_sizes = [49, 49, 49, 49, 49]
for i, (block_name, data_type, block_size) in enumerate(zip(block_names * 2, data_types, block_sizes)):
    for num in range(sum(block_sizes[:i]), sum(block_sizes[:i+1])):
        feature_name.append(f"{block_name}_{count}_{data_type}")
        count += 1
        if count > 49:
            count = 1

# Create a dictionary with old and new column names
new_column_names = {i: name for i, name in enumerate(feature_name, start=0)}

# Rename the columns from the second element (index 1)
df = df.rename(columns=new_column_names)

# Specify the path and filename for saving the CSV
csv_path = os.path.join(current_directory, "combined_features_vgg16.csv")
df.to_csv(csv_path, index=False)

print("Processing complete")
