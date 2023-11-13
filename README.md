<!-- omit from toc --> 
# building-footprint-detection
<!-- omit from toc --> 
## Example repo for building footprint and damage estimation from satellite/aerial images

<!-- omit from toc --> 
### Table of contents
- [1. Problem Statement](#1-problem-statement)

## 1. Problem Statement

Aim to identify building footprints within numerous images, and subsequently evaluate their structural integrity. Specifically, to develop a CV model that can

- Import satellite or aerial images from the designated data directory.
- Conduct image preprocessing to optimize data quality.
- Detect major types of objects such as buildings, cars or airplanes.
- Employ detection techniques to identify the precise building footprints.
- Generate a GeoJSON file containing polygon representations of the detected building footprints.
- Create new images with the building footprints superimposed for visualization.
- Obtain or generate tagged images of damaged buildings.
- Determine the presence or absence of damage to the buildings.
- Assess the degree of damage, categorizing it on a scale ranging from 'none' to 'low,' 'medium,' 'high,' or 'full.'

## 2. Loading and processing satellite/aerial images



## 3. Object detection
### 3.1. Buildings / cars / airplanes

### 3.2. Precise building footprints


## 4. Outputs

### 4.1. GeoJSON with building footprints


### 4.2. Images with footprints overlaid


## 5. Next steps

### 5.1. Damage estimation

- Label presence / absence of damage to buildings
- Determine presence / absence of damage to buildings (binary segmentation)
- Assess degree of damage on 5-point scale from 'none' to 'full'

## 6. Submission
- Make comprehensive report detailing your observations and insights
  - If any requirements cannot be met, provide a detailed plan outlining approach to fulfill them
- Make public