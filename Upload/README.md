# IPEO Project Group 4 - Large Rocks Detection

**Author 1 (sciper):** Jan Peter Reinhard Clevorn (377937)  
**Author 2 (sciper):** Samuel Darmon (XXXXXX)   
**Author 3 (sciper):** Sven Hominal (XXXXX)   



## Important notes

Please adhere to the folder structure given below, this will allow smoothly running this code. Rerunning all parts of this code will take a while, as generating dataset, data augmentation takes time. 


## Please adhere to the following file structure

```plaintext
root/
├── Images                             # Images of Results
├── testing.ipynb                      # Main notebook motivating and describing our process
├── enironment.yml                     # Requirement file (conda IPEO create -f environment.yml)
├── inference.ipynb                    # Notebook to run Inference
├── Dataset.py                         # Data Processing for the Dataset
├── utils.py                           # Function used for analysis
│
├── swissImage_50cm_patches            # Data folder to be downloaded from [DATA][data]
├── swissSURFACE3D_hillshade_patches   # Data folder to be downloaded from [DATA][data]
├── swissSURFACE3D_patches             # Data folder to be downloaded from [DATA][data]
└── inference.ipynb                    # Data folder to be downloaded from [DATA][data]


[data]: https://enacshare.epfl.ch/bY2wS5TcA4CefGks7NtXg
