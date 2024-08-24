import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn import preprocessing
from sklearn.datasets import load_iris
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.preprocessing import StandardScaler
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.covariance import EllipticEnvelope
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, transform, img_as_float
from skimage.util import montage

current_directory = os.getcwd()
print("You are in the directory: ", current_directory)
if not os.path.isdir(os.path.join(current_directory, "pca_results")):
    # Create the directory if it does not exist
    os.makedirs(os.path.join(current_directory, "pca_results"))

base_output_dir = os.path.join(current_directory + "/pca_results")

path2 = None
path1 = None

# Run the script with path1 or path2 -> does not work with both at the same time
# path1 = os.path.join(current_directory, "combined_features_manual.csv")
path2 = os.path.join(current_directory, "combined_features_vgg16.csv")

#################################################################################
# Visualize PCA with the entire dataset
#################################################################################

# Fix the seed for reproducibility
np.random.seed(42)

def normalize_data():
    if path1 is not None:
        df = pd.read_csv(path1)
    else:
        df = pd.read_csv(path2)
    # Remove any non-numeric columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    df = df[numeric_cols]
    # Apply StandardScaler to these columns
    X = StandardScaler().fit_transform(df)
    # Dimensionality reduction using PCA
    if path1 is not None:
        pca = PCA(n_components=2000)
    else:
        pca = PCA(n_components=200)
    X_pca = pca.fit_transform(X)
    # Calculate the explained variance ratio and cumulative sum
    exp_var_pca = pca.explained_variance_ratio_
    cum_sum_eigenvalues = np.cumsum(exp_var_pca)
    print(cum_sum_eigenvalues)
    # Find the minimum number of components that explain 95% of the variance
    index = np.argmax(cum_sum_eigenvalues > 0.95)
    print(index + 1)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # Normalize the data using MinMaxScaler
    scaler = preprocessing.MinMaxScaler()
    normalized_df = scaler.fit_transform(X_pca)
    normalized_df = pd.DataFrame(normalized_df, columns=['PCA1', 'PCA2'])  # Convert to DataFrame
    if path1:
        normalized_df.to_csv(os.path.join(current_directory + "/pca_results/DATA_manual_normalized.csv"))  # Save normalized data to CSV file
    else:
        normalized_df.to_csv(os.path.join(current_directory + "/pca_results/DATA_vgg16_normalized.csv"))  # Save normalized data to CSV file    
    return normalized_df

def pca_plots(k_value):
    # Load the original dataframe that contains the image names
    if path1 is not None:
        dataset = pd.read_csv(path1)
    else:
        dataset = pd.read_csv(path2)
    # Load the PCA-reduced dataframe
    if path1 is not None:
        df_pca = pd.read_csv(os.path.join(current_directory + "/pca_results/test_manual_full_k=" + str(k_value) + '.csv'))
    else:
        df_pca = pd.read_csv(os.path.join(current_directory + "/pca_results/test_vgg16_full_k=" + str(k_value) + '.csv'))

    # Merge the two dataframes on the index
    df = pd.concat([dataset, df_pca], axis=1)

    x = df.iloc[:, -2:].values
    y = df['Cluster'].values

    # Add a column for the sample type (control or drug)
    df['Sample_Type'] = np.where(df['image_name'].str.contains('control'), 'control', 'drug')

    # Use the 'style' argument to specify the marker shape based on the sample type
    sns.scatterplot(x="PCA1", y="PCA2", hue=y, style=df['Sample_Type'],
                    palette=sns.color_palette("hls", int(k_value)), markers=['o', 's'],
                    data=df).set(title='PCA Projection k = ' + str(k_value))
    plt.show()

if __name__ == '__main__':
    data = normalize_data()
    numeric_cols = data.select_dtypes(include=['int64', 'float64'])
    model = MiniBatchKMeans()
    inertias = []
    # Optimize k [2-12]
    for k in range(2, 13):
        kmeans = MiniBatchKMeans(n_clusters=k)
        kmeans.fit(numeric_cols)
        inertias.append(kmeans.inertia_)
        # Adds indicator (class-label) to file
        label = kmeans.labels_
        data["Cluster"] = label
        data["Cluster"] = data["Cluster"].astype("int")
        # Save each df with different k
        if path1 is not None:
            data.to_csv(os.path.join(current_directory + "/pca_results/test_manual_full_k=" + str(k) + '.csv'))
        else:
            data.to_csv(os.path.join(current_directory + "/pca_results/test_vgg16_full_k=" + str(k) + '.csv'))
    visualizer = KElbowVisualizer(model, k=(2, 13), metric='distortion', timings=False,
                                  title=('Mini Batch K-Means Clustering'), X=numeric_cols)

    visualizer.fit(data)  # Fit the data to the visualizer
    k_scores = visualizer.k_scores_
    visualizer.show()  # Finalize and render the figure
    print('-- Average Distance to Centroid --')
    for i in range(len(k_scores)):
        # Print average distances for each k
        print('K score: ' + str(i+2) + ' ' + str(k_scores[i]))

    best_k = visualizer.elbow_value_
    print('-- Best Value of K: ', best_k, ' --')

    # PCA projection best_k
    pca_plots(best_k)
    # PCA projection best_k -1
    # pca_plots(best_k-1)
    # PCA projection best_k +1
    # pca_plots(best_k+1)


#################################################################################
# Visualize a PCA by removing outliers
#################################################################################

def normalize_data():
    if path1 is not None:
        df = pd.read_csv(path1)
    else:
        df = pd.read_csv(path2)
    # Keep the index (image names)
    index = df.index
    # Remove any non-numeric columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    df = df[numeric_cols]
    # Apply StandardScaler to these columns
    X = StandardScaler().fit_transform(df)
    # Dimensionality reduction using PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    # Detect outliers using the elliptic envelope method
    ee = EllipticEnvelope()
    ee.fit(X_pca)
    outliers = ee.predict(X_pca) == -1
    # Remove outliers
    df = df.loc[~outliers]
    X_pca = X_pca[~outliers]
    # Normalize the data using MinMaxScaler
    scaler = preprocessing.MinMaxScaler()
    normalized_df = scaler.fit_transform(X_pca)
    normalized_df = pd.DataFrame(normalized_df, columns=['PCA1', 'PCA2'])  # Convert to DataFrame

    # Use the original index
    normalized_df.index = index[~outliers]

    if path1 is not None:
        normalized_df.to_csv(os.path.join(current_directory + "/pca_results/DATA_manual_normalized.csv"))  # Save normalized data to CSV file
    else:
        normalized_df.to_csv(os.path.join(current_directory + "/pca_results/DATA_vgg16_normalized.csv"))  # Save normalized data to CSV file    
    
    return normalized_df, index

def pca_plots(k_value, index):
    # Load the original dataframe that contains the image names
    if path1 is not None:
        dataset = pd.read_csv(path1)
    else:
        dataset = pd.read_csv(path2)
    # Load the PCA-reduced dataframe
    if path1 is not None:
        df_pca = pd.read_csv(os.path.join(current_directory + "/pca_results/test_manual_full_k=" + str(k_value) + '.csv'))
    else:
        df_pca = pd.read_csv(os.path.join(current_directory + "/pca_results/test_vgg16_full_k=" + str(k_value) + '.csv'))
    df = pd.merge(dataset, df_pca, left_index=True, right_index=True)

    x = df.iloc[:, -2:].values
    y = df['Cluster'].values

    # Add a column for the sample type (control or drug)
    df['Sample_Type'] = np.where(df['image_name'].str.contains('control'), 'control', 'drug')
    # Use the 'style' argument to specify the marker shape based on the sample type
    sns.scatterplot(x="PCA1", y="PCA2", hue=y, style=df['Sample_Type'],
                    palette=sns.color_palette("hls", int(k_value)), markers=['o', 's'],
                    data=df).set(title='PCA Projection k = ' + str(k_value))
    # Save the PCA plot
    if path1 is not None:
        plt.savefig(os.path.join(current_directory + f"/pca_results/pca_plot_manual_k={k_value}.png"))
    else:
        plt.savefig(os.path.join(current_directory + f"/pca_results/pca_plot_vgg16_k={k_value}.png"))
    plt.show()

    # Create a text file for best_k cluster
    if k_value == best_k:
        for i in range(k_value):
            cluster_df = df[df['Cluster'] == i]
            if path1 is not None:
                with open(os.path.join(current_directory, "pca_results", f"cluster_manual_{i}.txt"), 'w') as f:
                    for name in cluster_df['image_name']:
                        f.write(f'{name}\n')
            else:
                with open(os.path.join(current_directory, "pca_results", f"cluster_vgg16_{i}.txt"), 'w') as f:
                    for name in cluster_df['image_name']:
                        f.write(f'{name}\n')

if __name__ == '__main__':
    data, index = normalize_data()
    numeric_cols = data.select_dtypes(include=['int64', 'float64'])
    model = MiniBatchKMeans()
    inertias = []
    # Optimize k [2-12]
    for k in range(2, 13):
        kmeans = MiniBatchKMeans(n_clusters=k)
        kmeans.fit(numeric_cols)
        inertias.append(kmeans.inertia_)

        # Adds indicator (class-label) to file
        label = kmeans.labels_
        data["Cluster"] = label
        data["Cluster"] = data["Cluster"].astype("int")
        # Save each df with different k
        if path1 is not None:
            data.to_csv(os.path.join(current_directory + "/pca_results/test_manual_k=" + str(k) + '.csv'))
        else:
            data.to_csv(os.path.join(current_directory + "/pca_results/test_vgg16_k=" + str(k) + '.csv'))
    
    visualizer = KElbowVisualizer(model, k=(2, 13), metric='distortion', timings=False,
                                  title=('Mini Batch K-Means Clustering'), X=numeric_cols)
    visualizer.fit(data)  # Fit the data to the visualizer
    k_scores = visualizer.k_scores_

    visualizer.show()  # Finalize and render the figure
    print('-- Average Distance to Centroid --')

    for i in range(len(k_scores)):
        # Print average distances for each k
        print('K score: ' + str(i+2) + ' ' + str(k_scores[i]))
    best_k = visualizer.elbow_value_
    print('-- Best Value of K: ', best_k, ' --')
    # Load the original dataframe that contains the image names
    if path1 is not None:
        dataset = pd.read_csv(path1)
    else:
        dataset = pd.read_csv(path2)
    # Load the PCA-reduced dataframe for best_k
    if path1 is not None:
        df_pca = pd.read_csv(os.path.join(current_directory + "/pca_results/test_manual_k=" + str(best_k) + '.csv'))
    else:
        df_pca = pd.read_csv(os.path.join(current_directory + "/pca_results/test_vgg16_k=" + str(best_k) + '.csv'))
    
    # Merge the two dataframes on the index
    df = pd.merge(dataset, df_pca, left_index=True, right_index=True)

    # Create a text file for best_k cluster
    for i in range(best_k):
        cluster_df = df[df['Cluster'] == i]
        if path1 is not None:
            with open(os.path.join(current_directory, "pca_results", f"cluster_manual_{i}.txt"), 'w') as f:
                for name in cluster_df['image_name']:
                    f.write(f'{name}\n')
        else:
            with open(os.path.join(current_directory, "pca_results", f"cluster_vgg16_{i}.txt"), 'w') as f:
                for name in cluster_df['image_name']:
                    f.write(f'{name}\n')

    # PCA projection best_k
    pca_plots(best_k, index)
    # PCA projection best_k -1
    # pca_plots(best_k-1, index)
    # PCA projection best_k +1
    # pca_plots(best_k+1, index)

#################################################################################
# Extract images from PCA on the dataset obtained manually
#################################################################################

path = os.path.join(current_directory + "/PCNA")
img_height = img_width = 256

# Read the cluster[num].txt files
num_clusters = best_k

for i in range(num_clusters):
    if path1 is not None:
        with open(os.path.join(base_output_dir, f"cluster_manual_{i}.txt"), 'r') as f:
            files = [line.strip() for line in f.readlines()]
    else:
        with open(os.path.join(base_output_dir, f"cluster_vgg16_{i}.txt"), 'r') as f:
            files = [line.strip() for line in f.readlines()]

    data = []
    n_images = len(files)
    n_rows = n_cols = int(np.ceil(np.sqrt(n_images)))
    montage_img = np.zeros((n_rows * img_height, n_cols * img_width), dtype=np.uint8)
    for j, file in enumerate(files):
        filepath = os.path.join(path, file)
        img = io.imread(filepath, as_gray=True)
        # Scale the image to 256x256 using bilinear interpolation
        img = transform.resize(img, (img_height, img_width), anti_aliasing=True)
        img = img_as_float(img)
        min_val = np.min(img)
        max_val = np.max(img)
        # Normalize the pixel values to [0, 255]
        img = 255 * (img - min_val) / (max_val - min_val)
        data.append(img)
    # Create a montage of all images in data
    n_rows = n_cols = int(np.ceil(np.sqrt(len(data))))
    montage_img = montage(data, grid_shape=(n_rows, n_cols))
    # Display the montage image using matplotlib
    plt.figure(figsize=(12,12))
    plt.title(f'Cluster {i}')
    plt.imshow(montage_img, cmap="gray")
    plt.axis('off')
    plt.draw()
    if path1 is not None:
        file_path_to_save = os.path.join(base_output_dir, f"Cluster_manual_{i}.png")
    else:
        file_path_to_save = os.path.join(base_output_dir, f"Cluster_vgg16_{i}.png")
    plt.savefig(file_path_to_save, dpi=300)
    plt.show()

#################################################################################
# Table for manually extracted cells followed by PCA
#################################################################################

def create_table(num):
    if path1 is not None:
        file_path = os.path.join(base_output_dir, f"cluster_manual_{num}.txt")
    else:
        file_path = os.path.join(base_output_dir, f"cluster_vgg16_{num}.txt")    
    with open(file_path, "r") as file:
        text = file.read().splitlines()
    drug = []
    control = []
    for item in text:
        test = item.split("_")
        if test[1] == "control" or test[1] == "contol":
            control.append(item)
        elif test[1] == "drug":
            drug.append(item)
        else:
            print(f"Error processing {item}")

    # Create the dictionary
    data_dict = {
        "drug": len(drug),
        "control": len(control)
    }
    return data_dict, drug, control

df = pd.DataFrame()
for i in range(best_k): # best_k to be retrieved from PCA_feature_manual.py
    data = create_table(i)[0]
    df[f"cluster_{i}"] = pd.Series(data)

if path1 is not None:
    print("Table for manually extracted cells followed by PCA")
else:
    print("Table for automatically extracted cells followed by PCA")

print(df)

#################################################################################
# Statistical analysis to see if treatment influences classification
#################################################################################

import scipy.stats as stats

# Assume df is your contingency table
# Calculate the chi2 test
chi2, p, dof, expected = stats.chi2_contingency(df)

# Print the results
print("Chi2 statistic value:", chi2)
print("p-value:", p)
print("Degrees of freedom:", dof)
print("Expected frequency table:")
print(expected)

# Significance level alpha
alpha = 0.05

# Conclusions based on the p-value
if p > alpha:
    print("The null hypothesis HO of independence between variables is accepted.")
    print("There is no significant difference between observed proportions.")
else:
    print("The null hypothesis HO of independence between variables is rejected.")
    print("There is a significant difference between observed proportions.")
