# IPEO Project Group 4 - Large Rocks Detection

**Author 1 (sciper):** Jan Peter Reinhard Clevorn (377937)  
**Author 2 (sciper):** Samuel Darmon (325211)   
**Author 3 (sciper):** Sven Hominal (XXXXX)   

This project focuses on automating the detection of large rocks (≥5x5 meters) in the Swiss Alps using high-resolution aerial imagery and digital surface models (DSM). The primary objective is to develop an efficient object detection pipeline that can assist the Federal Office for Topography (swisstopo) in updating topographic maps with greater accuracy and efficiency.

## Dataset

The dataset comprises:

- **RGB Aerial Images**: High-resolution images from **swissIMAGE** with a standardized spatial resolution of 50 cm.
- **Digital Surface Models (DSM)**: Elevation data from **swissSURFACE3D**, also standardized to a 50 cm resolution.
- **Hillshade Rasters**: Derived from DSM data to enhance terrain visualization.
- **Annotations**: 2,625 manually annotated large rocks provided by swisstopo.

The study area includes regions in Valais, Ticino, and Graubünden. Images were segmented into 640x640 pixel tiles with a 25% overlap to facilitate model training.

## Model

We employed **YOLOv8**, a state-of-the-art object detection model known for its real-time detection capabilities and high accuracy. Among its variants, **YOLOv8X** was selected for its optimal balance between performance and computational efficiency in detecting large rocks.

## Training and Augmentation

To enhance model robustness and prevent overfitting, we applied several data augmentation techniques:

- **Color Adjustments**: Modified hue and value components to account for varying lighting conditions.
- **Geometric Transformations**: Applied translations to simulate different perspectives.
- **Random Erasing**: Introduced with a probability of 0.4 to improve the model's ability to generalize.
- **Auto Augmentation**: Utilized the `randaugment` policy to apply a diverse set of augmentations during training.

## Computational Resources

Training was conducted on the **Scientific IT and Application Support (SCITAS)** infrastructure at EPFL. SCITAS provides advanced computational resources and High-Performance Computing (HPC) expertise, facilitating efficient processing of large datasets and complex model architectures.

## Results

The model achieved significant accuracy in detecting large rocks across diverse terrains in the Swiss Alps. The integration of RGB and DSM data, combined with tailored augmentation strategies, contributed to the model's performance.

## Conclusion

This project demonstrates the potential of leveraging advanced object detection models like YOLOv8, combined with high-resolution geospatial data, to automate and enhance the accuracy of topographic mapping processes. Future work may explore the integration of additional data modalities and further optimization of model parameters to improve detection capabilities.

## Important notes

Please adhere to the folder structure given below, this will allow smoothly running this code. Rerunning all parts of this code will take a while, as generating dataset, data augmentation, etc. takes time. 


## Please adhere to the following file structure

```plaintext
root/
├── Images                             Images of Results
├── testing.ipynb                      Main notebook motivating and describing our process
├── enironment.yml                     Requirement file (conda IPEO create -f environment.yml)
├── inference.ipynb                    Notebook to run Inference
├── Dataset.py                         Data Processing for the Dataset
├── utils.py                           Function used for analysis
├── YoloTraining.ipnyb                 notebook for example YOLO training
│
├── swissImage_50cm_patches            Data folder to be downloaded from [data]
├── swissSURFACE3D_hillshade_patches   Data folder to be downloaded from [data]
├── swissSURFACE3D_patches             Data folder to be downloaded from [data]
└── inference.ipynb                    Data folder to be downloaded from [data]


[data]: https://enacshare.epfl.ch/bY2wS5TcA4CefGks7NtXg
