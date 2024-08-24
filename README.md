# This work was carried out as part of my Master 1 training.

# The task involved automatically determining phases of the cell cycle through a combination of fluorescence imaging and machine learning.

To visualize DNA and its replication, cells were stained using Hoechst dye and a marker for the PCNA protein. The obtained images were processed with the StarDist 2D plugin. Following this, we performed parameter extractions both manually and automatically, and then conducted various analyses (PCA, t-SNE using K-means). The goal was to use clustering from these analyses to understand the dynamics of cells in different phases of the cell cycle and to investigate whether a drug inhibiting the PARP1 protein affects this.

Since the cells were not fixed, we observed them using a confocal microscope at different wavelengths with an air objective: 450 nm for Hoechst staining and 561 nm for PCNA targeting.

Images in **.tif** format were extracted, and these are the ones we studied.

# images_extraction.py
Creates three repositories: *PCNA*, *HOECHST*, and *Mask*.
The code extracts each nucleus using a mask. The mask of the cell grid will be found in the Mask folder (to check the result), and the cells will be placed in the file corresponding to their treatment, with identical names to ensure proper tracking of the extraction.
Only sufficiently large nuclei will be retained to filter out false positives.
The image names are then processed to improve readability.

# manual_features_extraction.py
Extracts various features from images using different filters.

# vgg16_features_extraction.py
Extracts features from the first five layers of the VGG16 model.

# PCA_vgg16_and_manual.py and TSNE_vgg16_and_manual.py
Provide statistical results and interpretations of datas. 
They also allow for stacking images to better understand clustering and visualize the results.
