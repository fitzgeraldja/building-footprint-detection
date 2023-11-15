<!-- omit from toc --> 
# building-footprint-detection
<!-- omit from toc --> 
## Example repo for building footprint and damage estimation from satellite/aerial images

<!-- omit from toc --> 
### Table of contents
- [1. Problem Statement](#1-problem-statement)
- [2. Loading and processing satellite/aerial images](#2-loading-and-processing-satelliteaerial-images)
- [3. Object detection](#3-object-detection)
  - [3.1. Buildings / cars / airplanes](#31-buildings--cars--airplanes)
  - [3.2. Precise building footprints](#32-precise-building-footprints)
- [4. Outputs](#4-outputs)
  - [4.1. Images with footprints overlaid](#41-images-with-footprints-overlaid)
  - [4.2. GeoJSON with building footprints](#42-geojson-with-building-footprints)
- [5. Next steps](#5-next-steps)
  - [5.1. Building footprint estimation model performance improvement](#51-building-footprint-estimation-model-performance-improvement)
  - [5.2. Multi-class segmentation (e.g. building / car / airplane)](#52-multi-class-segmentation-eg-building--car--airplane)
  - [5.3. Damage estimation](#53-damage-estimation)

## 1. Problem Statement

Aim to identify building footprints within numerous images, and subsequently evaluate their structural integrity. Specifically, to develop a computer vision model that can

- Import satellite or aerial images from the designated data directory.
- Conduct image preprocessing to optimize data quality.
- Detect major types of objects such as buildings, cars or airplanes.
- Employ detection techniques to identify the precise building footprints.
- Generate a GeoJSON file containing polygon representations of the detected building footprints.
- Create new images with the building footprints superimposed for visualization.
- Obtain or generate tagged images of damaged buildings.
- Determine the presence or absence of damage to the buildings.
- Assess the degree of damage, categorizing it on a scale ranging from 'none' to 'low,' 'medium,' 'high,' or 'full.'

Due to time constraints, we restrict ourselves herein to a preliminary method of building footprint detection, with some comments as to how to improve model performance, and pursue subsequent development to address the remaining aims.

Predicted masks can be reproduced by opening the example notebook in Google Colab and running all cells.


## 2. Loading and processing satellite/aerial images

Due to the short timeframe of this project, we use existing packages for loading and processing the data. The most immediate issue when working with satellite images in particular, but also some aerial images, is the size of the file. As such, rather than passing the whole image to the model simultaneously, we specify a patch/fragment size, and process each fragment separately before recombining the pieces into a mask for the whole image. 

Further, detection/segmentation models can display a performance bias towards objects at a certain orientation - as such, even when using a pre-trained model, we use test-time augmentation, in which multiple transformations (e.g. reflection, rotation) are applied to each patch, and these transformed images are fed into the model. The resulting predictions are then passed through the inverse transform, and aggregated together across all such transformation processes to produce a single prediction mask. In this study, we simply take the mean prediction across all transformation processes.

To summarise, the process for each image will be

1. Break the image up into fragments of a specified size
2. For each fragment, apply one of the specified transforms
3. For each transformed fragment, apply the model to obtain a predicted segmentation mask (into building footprint / background)
4. Apply the inverse transform to this mask to return to the original image space
5. Take the average across all such predicted masks to obtain a single predicted segmentation mask for the fragment
6. Piece all fragment-level predictions back together to obtain a predicted segmentation mask for the whole image

## 3. Object detection
### 3.1. Buildings / cars / airplanes
While not covered in the example notebook, we also tested a model trained to predict certain object classes - including cars and some airplanes - within aerial images to some success. The model chosen, [LSKnet](https://github.com/zcablii/LSKNet/tree/895ae73dbc12b6fb966642580614a1ae1329f7ea), was selected due to highest reported performance on a remote-sensing benchmark dataset, and found through [Papers with Code](https://paperswithcode.com/) - we find this to usually be the fastest way to discover and implement state-of-the-art models for a given new task. 

The resulting out-of-the-box predictions were quite accurate for vehicle classes, but as the focus of the task seemed to be building footprints (and subsequently damage), we did not pursue a more polished implementation.

### 3.2. Precise building footprints

Following the same process of model discovery via Papers with Code, the suggested models for building footprint detection faced many more issues in implementation. In particular, typically either (i) pre-trained model checkpoints were not released alongside code, meaning we would have had to perform our own (potentially expensive in both time and money) model training to produce even preliminary results, or (ii) models required earlier versions of python / certain key packages that caused significant issues when trying to run via Google Colab (as we typically use for GPU access for simple small projects such as this).

As such, our solution was to use an existing package, [building-footprint-segmentation](https://github.com/fuzailpalnak/building-footprint-segmentation/tree/main), which also included pre-trained models for the task. There were various issues with their examples, but after some corrections and modifications, this enabled us to obtain our preliminary predictions.

In the images with the predicted masks overlaid we can observe immediately that this pretrained model performs poorly for this task - this may have been due to various differences between train and test data, e.g.: it expected images at a different scale and/or resolution; it expected on-nadir (i.e. directly overhead) rather than angled as in some example aerial images; or input images may have required some further processing (e.g. colour normalisation) to better match with train data. The key step to resolve the former issues, especially for on-nadir images, would be to simply perform some additional fine-tuning of the model. To do so however, we would first need to obtain test masks for the images - this would require manual annotation, which we did not perform due to time constraints, but is a natural next step that requires minimal additional work (e.g. using existing image annotation tools). 

The latter issue would require image processing steps such as colour histogram matching to the training set, but again we did not find time to do so for this short project.

## 4. Outputs

### 4.1. Images with footprints overlaid
Having obtained predicted binary segmentation masks, the most immediate output is simply the images with the these estimated footprints overlaid. To do so, we only need plot the image, then plot the mask on the same figure, with some alpha > 0 to ensure its transparency. These are available in the `pred_imgs` directory, and as previously stated clearly show the need for further model training and/or input image processing for the task.

### 4.2. GeoJSON with building footprints

To convert these binary segmentation masks into a GeoJSON containing the predicted building footprint boundaries, we take several steps. 

Firstly, to merge nearby prediction mask patches into a contiguous patch, we apply binary dilation - expanding the mask into adjacent pixels. Herein we only do so a small number of times (twice) but this could be altered should it be found to improve performance whatsoever, after the model itself has been improved further. 

Next, we use the `findContours` from the cv2 package to find the minimal convex hull arrays containing each patch within the segmentation masks. 

We then take these arrays, and process them into Shapely polygons, which finally the GeoPandas package can convert into a GeoDataFrame, which can immediately be saved as GeoJSON. 

The resulting GeoJSON masks can be found in the `pred_masks` directory.

## 5. Next steps
### 5.1. Building footprint estimation model performance improvement

Initial model performance improvement for the task of building footprint estimation is as described above, in brief reducing to three tasks, in increasing order of complexity:

1. Manually annotate the test images to allow fine-tuning - though note we would then have to evaluate on held-out image fragments, as any fragments used for fine-tuning would have been seen by the model;
2. Better understand the current model used, and in particular the expected data format, and implement further image preprocessing if required (e.g. colour histogram matching);
3. Change to another (better) model, and perform the above steps again if required to further increase performance.

### 5.2. Multi-class segmentation (e.g. building / car / airplane)

Once again, as stated above there are alternative existing models for remote-sensing multi-class segmentation / object instance detection, for instance LSKnet. It seems that typically such models are trained to predict objects _other_ than buildings, so to detect both together we would ensemble such alongside building-specific models, and again fine-tune etc if necessary. 

### 5.3. Damage estimation

The key unaddressed aim for the model was building damage estimation - first as a binary segmentation problem (presence/absence of damage), then as a multi-class segmentation problem (degree of damage on 5-point scale from 'none' to 'full'). The current state-of-the-art approach seems to be models trained on the [xBD dataset](https://paperswithcode.com/sota/2d-semantic-segmentation-on-xbd), however these cannot be applied to single images such as those provided. 

Instead, they require pre- and post-disaster images, to better understand how buildings have _changed_, rather than assuming 'damage' can be understood uniformly across all contexts. From a preliminary search, it appears that models based off single images perform significantly worse at the task, likely principally due to the non-uniformity of the task itself (e.g. some buildings may appear superficially damaged, but were built that way / simply aged etc).

In any case, once again the first step to address this task would be to obtain labelled images, then further fine-tune the model for the context of interest if required.