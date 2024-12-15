# IPEO Project Group 4 - Large Rocks Detection

### Run Testing.ipynb before running the Augmentation.ipynb, as the Augmentation.ipynb uses the Yolo Format created in Testing.ipynb

### Instructions:
1. Keep the folders:
   - `swissImage_50cm...`
   - `swissSURFACE3D_hillschade...`
   - `swissSURFACE3D_patches`
   
   **inside** the `IPEO_PROJECT_GROUP_4` directory.
   
2. Keep this structure to read the images.  
3. Perfom the data augmentation using `Augmentation.ipynb` file. The following augmentations are made : Brightness Augmentation and Random Flipping. The augmentation files are saved locally, they serve to better visualize initial data snd to augment the dataset in order to perform a train on more images, and avoid overfitting on a smaller dataset.  
4. Perform the training using `Training.ipynb` file. The model is trained on the augmented dataset. Several YOLOv8 architectures are used and tested.    
5. Perform a cross-validation to obtain the best hyperparameters.   
6. Perform the testing using `Testing.ipynb` file. The model is tested on the test set.  
7. Assess the results.    


