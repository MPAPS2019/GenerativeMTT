[//]: # (This directory provides definitions for a few common models, dataloaders, scheduler,)

[//]: # (and optimizers that are often used in training.)

[//]: # (The definition of these objects are provided in the form of lazy instantiation:)

[//]: # (their arguments can be edited by users before constructing the objects.)

[//]: # (They can be imported, or loaded by `model_zoo.get_config` API in users' own configs.)


# Generating Mean Transition Time (MTT) Images from Label-Free Optical Imaging Using MA-DenseUNet

This repository contains the code for generating Mean Transition Time (MTT) images of indocyanine green (ICG) fluorescence imaging from label-free optical imaging modalities, including Laser Speckle Contrast Imaging (LSCI) and White Light Imaging (WLI). This work leverages the Mixed Attention Dense UNet (MA-DenseUNet) model, which integrates dense blocks and mixed attention mechanisms into the UNet architecture to enhance feature extraction and accurately generate MTT images.

## Overview

In modern neurosurgical procedures, especially in minimally-invasive and robot-assisted surgeries, accurate visualization of hemodynamics is crucial. However, agent-related issues associated with traditional fluorescence imaging can pose challenges. To address this, our approach focuses on using deep learning to generate MTT images from label-free modalities, offering comprehensive hemodynamic information that can assist surgeons in differentiating arteries from veins and understanding blood flow direction.

### Key Contributions
1. **Statistical Gating**: Applied to laser speckle contrast images to match the imaging depth of fluorescence imaging.
2. **MA-DenseUNet Model**: Integrates dense blocks and mixed attention mechanisms into UNet for enhanced feature extraction and improved global and local context capture.
3. **Fluorescence-like Video Extraction**: Extracted from the generated MTT images to visually present blood flow direction, providing surgeons with intuitive and direct hemodynamic information.

For detailed information, please refer to our paper: [xxx].

## Repository Structure

- `code/`: Contains the implementation of the MA-DenseUNet model and related scripts for data processing and model training.
- `data/`: Placeholder for datasets (not included due to size or privacy concerns; please follow the instructions below to access the data).
- `results/`: Example outputs including generated MTT images and fluorescence-like videos.
- `notebooks/`: Jupyter notebooks for step-by-step tutorials and demonstrations.

## Getting Started

### Prerequisites

- Python 3.x
- PyTorch
- Other dependencies as listed in `requirements.txt`

### Installation

Clone this repository and install the required packages:

```bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
pip install -r requirements.txt