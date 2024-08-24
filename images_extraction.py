from stardist.models import StarDist2D
from stardist.data import test_image_nuclei_2d
from stardist.plot import render_label
from stardist import random_label_cmap, _draw_polygons, export_imagej_rois
from csbdeep.utils import normalize
import matplotlib.pyplot as plt
import numpy as np
import skimage
import skimage.io
from skimage import data, color
from skimage.transform import rescale, resize
from PIL import Image
import skimage.transform
import os
import cv2

#################################################################################
# Creation of the necessary directories
#################################################################################

current_directory = os.getcwd()

# Check if the directory exists in the current directory
if not os.path.isdir(os.path.join(current_directory, "HOECHST")):
    # Create the directory if it does not exist
    os.makedirs(os.path.join(current_directory, "HOECHST"))
if not os.path.isdir(os.path.join(current_directory, "PCNA")):
    # Create the directory if it does not exist
    os.makedirs(os.path.join(current_directory, "PCNA"))
if not os.path.isdir(os.path.join(current_directory, "Mask")):
    # Create the directory if it does not exist
    os.makedirs(os.path.join(current_directory, "Mask"))


#################################################################################
# Extract images from the control cell grids
#################################################################################

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # This will hide INFO and WARNING messages.

# Define the path to the directory containing the .tif images
path = os.path.join(current_directory, "tapis_cellulaire_Control")

# Change the working directory to the path where the images are located
os.chdir(path)

# This list holds all the image filenames
whole_image = []

# Creates a ScandirIterator aliased as files
with os.scandir(path) as files:
    # Loops through each file in the directory
    for file in files:
        if file.name.endswith('.tif'):
            # Adds only the image files to the nuclei list
            whole_image.append(file.name)

# Create the destination directories if they do not exist
            path = os.path.join(current_directory, "tapis_cellulaire_Control")

path_Channel_2 = os.path.join(current_directory,"PCNA")
path_Channel_1 = os.path.join(current_directory,"HOECHST")
path_masks = os.path.join(current_directory,"Mask")

# Iterate over each image in the nuclei list
for image_name in whole_image:
    # Set the path of the original image to a variable and target path
    image_path = os.path.join(path, image_name)

    # Load the two-channel image stack
    image_stack = skimage.io.imread(image_path)

    plt.figure(figsize=(8,8))
    plt.subplot(1,2,1)
    plt.imshow(image_stack[0, :, :], cmap='gray')
    plt.axis("off")
    plt.title("Channel_1")
    plt.subplot(1,2,2)
    plt.imshow(image_stack[1, :, :], cmap='gray')
    plt.axis("off")
    plt.title("Channel_2")

     # Select the first channel 0 or second channel 1 for nuclei segmentation
    image_full = image_stack[0, :, :]
    image = rescale(image_full, 0.33, anti_aliasing=False)
    #image=transform.rescale(image_full, 1/3, anti_aliasing=False, multichannel=False, mode='reflect')

    # Load a pre-trained StarDist model
    model = StarDist2D.from_pretrained('2D_versatile_fluo')


    # Predict instance segmentation on the example image
    labels, details = model.predict_instances(normalize(image))

    def remove_border_objects(labels):
      # Create a binary mask of the labels
      mask = labels > 0

      # Remove objects touching the border of the image
      border = np.zeros_like(mask)
      border[0, :] = True
      border[-1, :] = True
      border[:, 0] = True
      border[:, -1] = True
      mask &= ~border

      # Update the labels to exclude objects touching the border
      labels[~mask] = 0
      return labels

    # Remove objects touching the border of the image
    labels = remove_border_objects(labels)

    # Upsample the labels to the original size
    upsampled_labels = resize(labels, image_full.shape, order=0, anti_aliasing=False, preserve_range=True)
    labels = upsampled_labels
    # Get the list of unique labels
    unique_labels = np.unique(labels)

    # Get the base name of the image file
    base_name, _ = os.path.splitext(os.path.basename(image_path))

    # Iterate over each label to save the nuclei as independent images
    for label in unique_labels[1:]:
        mask = labels == label
        nucleus = np.zeros_like(image_stack)
        nucleus[0, :, :] = image_stack[0, :, :] * mask
        nucleus[1, :, :] = image_stack[1, :, :] * mask
        # Crop the nucleus by finding the bounds of the nucleus in the mask
        coords = np.column_stack(np.where(mask))
        top_left = coords.min(axis=0)
        bottom_right = coords.max(axis=0)
        crop = nucleus[:, top_left[0]:bottom_right[0]+1, top_left[1]:bottom_right[1]+1]
        file_name = "{}_nucleus_{}_Channel_1.tif".format(base_name, label)
        full_path = os.path.join(path_Channel_1, file_name)
        skimage.io.imsave(full_path, crop[0, :, :].astype(np.uint16))
        file_name = "{}_nucleus_{}_Channel_2.tif".format(base_name, label)
        full_path = os.path.join(path_Channel_2, file_name)
        skimage.io.imsave(full_path, crop[1, :, :].astype(np.uint16))

    # Remove objects touching the border of the image
    labels = remove_border_objects(labels)
    # Save the image displayed by plt.imshow to a file object
    fig = plt.imshow(render_label(labels, img=image_full))
    plt.axis("off")
    plt.savefig("output.tif", format='tif')
    plt.gcf().clear()
    plt.close()

    # Read the image data from the file object
    image_data = skimage.io.imread("output.tif")
    full_path = os.path.join(path_masks, "{}_mask.tif".format(base_name))
    # Save the image data as TIF
    skimage.io.imsave(full_path, image_data)

    # Save the image displayed by plt.imshow to a file object
    fig = plt.imshow(render_label(labels, img=image_full))
    plt.axis("off")
    plt.savefig("output_border_removed.tif", format='tif')
    plt.gcf().clear()



#################################################################################
# Then extract images from the drug-treated cell grids
#################################################################################

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # This will hide INFO and WARNING messages.

# Define the path to the directory containing the .tif images
path = os.path.join(current_directory,"tapis_cellulaire_Drug")

os.chdir(path)
whole_image = []

# Creates a ScandirIterator aliased as files
with os.scandir(path) as files:
    # Loops through each file in the directory
    for file in files:
        if file.name.endswith('.tif'):
            # Adds only the image files to the nuclei list
            whole_image.append(file.name)

# Iterate over each image in the nuclei list
for image_name in whole_image:
    # Set the path of the original image to a variable and target path
    image_path = os.path.join(path, image_name)

    # Load the two-channel image stack
    image_stack = skimage.io.imread(image_path)

    plt.figure(figsize=(8,8))
    plt.subplot(1,2,1)
    plt.imshow(image_stack[0, :, :], cmap='gray')
    plt.axis("off")
    plt.title("Channel_1")
    plt.subplot(1,2,2)
    plt.imshow(image_stack[1, :, :], cmap='gray')
    plt.axis("off")
    plt.title("Channel_2")

     # Select the first channel 0 or second channel 1 for nuclei segmentation
    image_full = image_stack[0, :, :]
    image = rescale(image_full, 0.33, anti_aliasing=False)
    #image=transform.rescale(image_full, 1/3, anti_aliasing=False, multichannel=False, mode='reflect')

    # Load a pre-trained StarDist model
    model = StarDist2D.from_pretrained('2D_versatile_fluo')

    # Predict instance segmentation on the example image
    labels, details = model.predict_instances(normalize(image))

    def remove_border_objects(labels):
      # Create a binary mask of the labels
      mask = labels > 0

      # Remove objects touching the border of the image
      border = np.zeros_like(mask)
      border[0, :] = True
      border[-1, :] = True
      border[:, 0] = True
      border[:, -1] = True
      mask &= ~border

      # Update the labels to exclude objects touching the border
      labels[~mask] = 0

      return labels

    # Remove objects touching the border of the image
    labels = remove_border_objects(labels)

    # Upsample the labels to the original size
    upsampled_labels = resize(labels, image_full.shape, order=0, anti_aliasing=False, preserve_range=True)
    labels = upsampled_labels
    # Get the list of unique labels
    unique_labels = np.unique(labels)

    # Get the base name of the image file
    base_name, _ = os.path.splitext(os.path.basename(image_path))

    # Iterate over each label to save the nuclei as independent images
    for label in unique_labels[1:]:
        mask = labels == label
        nucleus = np.zeros_like(image_stack)
        nucleus[0, :, :] = image_stack[0, :, :] * mask
        nucleus[1, :, :] = image_stack[1, :, :] * mask
        # Crop the nucleus by finding the bounds of the nucleus in the mask
        coords = np.column_stack(np.where(mask))
        top_left = coords.min(axis=0)
        bottom_right = coords.max(axis=0)
        crop = nucleus[:, top_left[0]:bottom_right[0]+1, top_left[1]:bottom_right[1]+1]
        file_name = "{}_nucleus_{}_Channel_1.tif".format(base_name, label)
        full_path = os.path.join(path_Channel_1, file_name)
        skimage.io.imsave(full_path, crop[0, :, :].astype(np.uint16))
        file_name = "{}_nucleus_{}_Channel_2.tif".format(base_name, label)
        full_path = os.path.join(path_Channel_2, file_name)
        skimage.io.imsave(full_path, crop[1, :, :].astype(np.uint16))

    # Remove objects touching the border of the image
    labels = remove_border_objects(labels)
    # Save the image displayed by plt.imshow to a file object
    fig = plt.imshow(render_label(labels, img=image_full))
    plt.axis("off")
    plt.savefig("output.tif", format='tif')
    plt.gcf().clear()
    plt.close()

    # Read the image data from the file object
    image_data = skimage.io.imread("output.tif")
    full_path = os.path.join(path_masks, "{}_mask.tif".format(base_name))
    # Save the image data as TIF
    skimage.io.imsave(full_path, image_data)

    # Save the image displayed by plt.imshow to a file object
    fig = plt.imshow(render_label(labels, img=image_full))
    plt.axis("off")
    plt.savefig("output_border_removed.tif", format='tif')
    plt.gcf().clear()

#################################################################################
# The image names are then processed to facilitate their subsequent use
#################################################################################

# Path to the directory containing files with names to be modified
directories = [os.path.join(current_directory, "HOECHST"), os.path.join(current_directory, "PCNA")]
for directory in directories:
    # List all files in the directory
    files = os.listdir(directory)

    # Iterate over all files in the directory
    for file in files:
        # Full path of the file
        file_path = os.path.join(directory, file)

        # New file name with characters '-' and '#' replaced by '_' and removal of channel names
        new_name = file.replace('-', '_').replace(' #', '_').replace('.czi', "_").replace("__", "_").replace(" _", "_").replace("_Channel_1", "").replace("_Channel_2", "").replace("_PCNA_livecells_WT_control_", "_control_").replace("_1_", "_")

        # Rename the file with the new name if it is different from the old one
        if new_name != file:
            new_file_path = os.path.join(directory, new_name)
            os.rename(file_path, new_file_path)
            print(f"Renamed {file} to {new_name}")

            # Check if the file is an image file (e.g., with a .tif extension)
            if os.path.isfile(new_file_path) and new_name.endswith(('.tif', '.tiff', '.png', '.jpg')):
                image = cv2.imread(new_file_path)
                if image is not None:
                    print(f"Successfully read the image: {new_name}")
                else:
                    print(f"Failed to read the image: {new_name}")
            else:
                print(f"Ignored unsupported file: {new_name}")
        else:
            print(f"Ignored {file} as it does not need renaming.")


#################################################################################
# To complete this step of preprocessing raw data, we will filter the obtained images
#################################################################################

# Filter images based on their size and weight
directory = os.path.join(current_directory, "HOECHST")

for filename in os.listdir(directory):
    if filename.endswith(".tif") or filename.endswith(".jpeg") or filename.endswith(".png"):
        filepath = os.path.join(directory, filename)
        filesize = os.path.getsize(filepath)
        if filesize < 9000:
            print(filepath, "removed")
            os.remove(filepath)
        else:
            with Image.open(filepath) as img:
                width, height = img.size
                if abs(width / height - 1) > 0.2:
                    os.remove(filepath)
                    print(filepath, "removed")

directory = os.path.join(current_directory, "PCNA")

for filename in os.listdir(directory):
    if filename.endswith(".tif") or filename.endswith(".jpeg") or filename.endswith(".png"):
        filepath = os.path.join(directory, filename)
        filesize = os.path.getsize(filepath)
        if filesize < 9000:
            print(filepath, "removed")
            os.remove(filepath)
        else:
            with Image.open(filepath) as img:
                width, height = img.size
                if abs(width / height - 1) > 0.2:
                    print(filepath, "removed")
                    os.remove(filepath)
