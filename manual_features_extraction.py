import os
import numpy as np
import cv2
import pandas as pd
from skimage import exposure, transform, img_as_float
from skimage.feature import entropy
from skimage.morphology import disk, convex_hull_image
from skimage.filters import threshold_multiotsu, difference_of_gaussians
from skimage.measure import euler_number, label
from skimage import io
import pickle
from tqdm import tqdm
from scipy.stats import skew, kurtosis

current_directory = os.getcwd()
path = os.path.join(current_directory, "PCNA")
os.chdir(path)

# This list holds all the image filenames
nuclei = []
# Creates a ScandirIterator aliased as files
with os.scandir(path) as files:
    # Loops through each file in the directory
    for file in files:
        if file.name.endswith('.tif'):
            # Adds only the image files to the nuclei list
            nuclei.append(file.name)

def extract_Pixel_values_and_Gabor(image):
    img2 = image.reshape(-1)
    mean_Original = np.mean(img2)
    var_Original = np.var(img2)
    median_Original = np.median(img2)
    total_intensity = np.sum(img2)
    std_deviation = np.std(img2)
    skewness = skew(img2)
    kurt = kurtosis(img2)
    # Generate Gabor features
    features_Pixel_Gabor = [mean_Original, var_Original, median_Original, total_intensity, std_deviation, kurt]
    num = 1  # To count numbers in order to give Gabor features a label in the data frame
    kernels = []  # Create an empty list to hold all kernels that we will generate in a loop
    for theta in range(6):   # Define number of thetas. Here only 2 theta values 0 and 1/4. pi
        theta = theta / 4. * np.pi
        for sigma in (1, 6):  # Sigma with values of 1 and 6
            for lamda in np.arange(0, np.pi, np.pi / 4):   # Range of wavelengths
                for gamma in (0.05, 0.5):   # Gamma values of 0.05 and 0.5
                    gabor_label = 'Gabor' + str(num)  # Label Gabor columns as Gabor1, Gabor2, etc.
                    ksize = 9
                    kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, 0, ktype=cv2.CV_32F)
                    kernels.append(kernel)
                    # Now filter the image and add values to a new column
                    fimg = cv2.filter2D(img2, cv2.CV_8UC3, kernel)
                    filtered_img = fimg.reshape(-1)
                    mean_filtered_img = np.mean(filtered_img)
                    var_filtered_img = np.var(filtered_img)
                    median_filtered_img = np.var(filtered_img)
                    features_Pixel_Gabor.extend([mean_filtered_img, var_filtered_img, median_filtered_img])
                    num += 1  # Increment for Gabor column label
    return features_Pixel_Gabor

def binarize_image_otsu(image):
    # Convert image to grayscale if it is not already
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Otsu's thresholding
    _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary_image

def extract_entropy_noise(image):
    # Generate other features
    features_entropy_noise = []
    rng = np.random.default_rng()
    noise_mask = image
    noise = (noise_mask * rng.random(noise_mask.shape) - 0.9 * noise_mask).astype(np.uint8)
    img_entropy = noise + 128
    entr_img = entropy(img_entropy, disk(15))
    entr1 = entr_img.reshape(-1)
    mean_entropy_img = np.mean(entr1)
    var_entropy_img = np.var(entr1)
    median_entropy_img = np.var(entr1)
    noise_img = img_entropy
    noise1 = noise_img.reshape(-1)
    mean_noise_img = np.mean(noise1)
    var_noise_img = np.var(noise1)
    median_noise_img = np.var(noise1)
    features_entropy_noise.extend([mean_entropy_img, var_entropy_img, median_entropy_img,
                                   mean_noise_img, var_noise_img, median_noise_img])
    return features_entropy_noise

def extract_multiotsu_euler(image):
    features_multiotsu_euler = []
    # Applying multi-Otsu threshold for the default value, generating three classes.
    wimage = image * window('hann', image.shape)  # Window image to improve FFT
    filtered_image = difference_of_gaussians(wimage, 1, 3)
    thresholds = threshold_multiotsu(filtered_image, classes=3)
    # Using the threshold values, we generate the three regions.
    regions = np.digitize(image, bins=thresholds)
    regions1 = regions.reshape(-1)
    chull = convex_hull_image(regions)
    chull1 = chull.reshape(-1)
    # Using the threshold values, we generate the three regions.
    mean_multi_otsu = np.mean(regions1)
    var_multi_otsu = np.var(regions1)
    perc_10_multi_otsu = np.percentile(regions1, 10)
    perc_25_multi_otsu = np.percentile(regions1, 25)
    median_multi_otsu = np.median(regions1)
    perc_75_multi_otsu = np.percentile(regions1, 75)
    perc_90_multi_otsu = np.percentile(regions1, 90)
    e4 = euler_number(regions, connectivity=1)
    object_nb_4 = label(regions, connectivity=1).max()
    holes_nb_4 = object_nb_4 - e4
    e8 = euler_number(regions, connectivity=2)
    object_nb_8 = label(regions, connectivity=2).max()
    holes_nb_8 = object_nb_8 - e8
    mean_chull = np.mean(chull1)
    var_chull = np.var(chull1)
    features_multiotsu_euler.extend([mean_multi_otsu, var_multi_otsu, perc_10_multi_otsu,
                                     perc_25_multi_otsu, median_multi_otsu, perc_75_multi_otsu,
                                     perc_90_multi_otsu, e4, object_nb_4, holes_nb_4, e8, object_nb_8,
                                     holes_nb_8])
    features_multiotsu_euler.extend(regions1)
    return features_multiotsu_euler

def extract_multi_spot_intensity(image):
    features_multi_spot_intensity = []
    image = transform.resize(image, (64, 64), anti_aliasing=True)
    image = img_as_float(image)
    img2 = image.reshape(-1)
    top_left = image[:10, :10]
    top_right = image[:10, -10:]
    top_middle = image[:10, 27:37]
    middle_left = image[27:37, :10]
    middle = image[14:50, 14:50]
    middle_right = image[27:37, -10:]
    bottom_left = image[-10:, :10]
    bottom_right = image[-10:, -10:]
    bottom_middle = image[-10:, 27:37]
    tl_mean = np.mean(top_left)
    tr_mean = np.mean(top_right)
    tm_mean = np.mean(top_middle)
    ml_mean = np.mean(middle_left)
    m_mean = np.mean(middle)
    mr_mean = np.mean(middle_right)
    bl_mean = np.mean(bottom_left)
    br_mean = np.mean(bottom_right)
    bm_mean = np.mean(bottom_middle)
    value = m_mean / np.mean([tl_mean, tr_mean, tm_mean, bl_mean, br_mean, bm_mean, ml_mean, mr_mean])
    features_multi_spot_intensity.extend([tl_mean, tr_mean, tm_mean, ml_mean, m_mean, mr_mean,
                                           bl_mean, br_mean, bm_mean, value])
    features_multi_spot_intensity.extend(img2)
    return features_multi_spot_intensity

def extract_multi_plot_profile_intensity(image):
    features_multi_plot_profile_intensity = []
    start = (0, 0)
    end = (image.shape[0] - 1, image.shape[1] - 1)
    start2 = (0, image.shape[1] - 1)
    end2 = (image.shape[0] - 1, 0)
    start3 = (image.shape[0] // 2, 0)  # Top center of the image
    end3 = (image.shape[0] // 2, image.shape[1] - 1)
    start4 = (0, image.shape[1] // 2)  # Center of the image
    end4 = (image.shape[0] - 1, image.shape[1] // 2)
    # Get profile line
    profile = skimage.measure.profile_line(image, start, end)
    profile2 = skimage.measure.profile_line(image, start2, end2)
    profile3 = skimage.measure.profile_line(image, start3, end3)
    profile4 = skimage.measure.profile_line(image, start4, end4)
    features_multi_plot_profile_intensity.extend(profile)
    features_multi_plot_profile_intensity.extend(profile2)
    features_multi_plot_profile_intensity.extend(profile3)
    features_multi_plot_profile_intensity.extend(profile4)
    return features_multi_plot_profile_intensity

data = {}
feature_vectors = []

# Loop through each image in the dataset
for nucleus in tqdm(nuclei):
    img = io.imread(nucleus, as_gray=True)
    # Scale the image to 64x64 using bilinear interpolation
    img_resized = transform.resize(img, (64, 64), anti_aliasing=True)
    image = img_as_float(img_resized)
    # Normalize the pixel values to [0, 255]
    min_val = np.min(image)
    max_val = np.max(image)
    image = 255 * (image - min_val) / (max_val - min_val)
    # Normalization by percentile
    p5, p95 = np.percentile(image, (5, 95))
    image = exposure.rescale_intensity(image, in_range=(p5, p95))
    # Normalize by median absolute deviation (MAD)
    median = np.median(image)
    mad = np.median(np.abs(image - median))
    if mad == 0:
        mad = 1e-5  # Add small constant to denominator
    image = (image - median) / mad
    # Normalize by interquartile range (IQR)
    q1, q3 = np.percentile(image, [25, 75])
    iqr = q3 - q1
    if iqr == 0:
        iqr = 1e-5  # Add small constant to denominator
    image = (image - q1) / iqr
    image = (image - img.mean()) / image.std()
    image = exposure.equalize_hist(image)
    # Try to extract the features and update the dictionary
    try:
        features = []
        features.extend(extract_Pixel_values_and_Gabor(image))
        features.extend(extract_entropy_noise(image))
        features.extend(extract_multiotsu_euler(image))
        features.extend(extract_multi_spot_intensity(image))
        features.extend(extract_multi_plot_profile_intensity(image))
        feature_vectors.append(features)
        data[nucleus] = features

    # If something fails, save the extracted features as a pickle file (optional)
    except:
        with open(path, 'wb') as file:
            pickle.dump(data, file)

print(features)
# Get a list of the filenames
filenames = np.array(list(data.keys()))

# Get a list of just the features
feat = np.array(list(data.values()))

# Reshape so that there are ### samples of feat.shape[1] vectors
feat = feat.reshape(-1, feat.shape[1])
print(feat)
print("Processing finished")

#####################################################################################
# We can now save this information in a CSV file for later processing
#####################################################################################

# Replace the previous code
all_image_names = []
for filename in os.listdir(path):
    image_path = os.path.join(path, filename)
    all_image_names.append(os.path.basename(image_path))

# Create a pandas DataFrame from the NumPy array and add a column with image names
df = pd.DataFrame(feat)
df.insert(0, "image_name", all_image_names)
# Display the DataFrame
print(df)

# Specify the path and filename to save the CSV
csv_path = os.path.join(current_directory, "combined_features_manual.csv")

# Save the pandas DataFrame to a CSV file
df.to_csv(csv_path, index=False)
print("Processing finished")
