# Final Project
This project aims to utilize this kaggle dataset (https://www.kaggle.com/datasets/orvile/bone-fracture-dataset) of bone fracture classification data to train a classifier on the cloud. Additionally, we later included the FracAtlas dataset (https://figshare.com/articles/dataset/The_dataset/22363012?file=43283628). It is expected that the first dataset is palced in a directory "data" and the second a directory "data_new", with fracture samples in a subdirectory called "fracture" and healthy bones in one called "normal".

## Code
```main.py``` trains the three models on the first dataset and ```main_new.py``` does the same for the second dataset. The model is in ```models.py``` and the visualization code is i ```model_visualization.py``` 

## Division of Labor
Alex: data preprocessing, augmentation, and baselines
Kevin: model architecture
Ning: cloud training

## Instructions
Clone the repo and run ```pip3 install -r requirements.txt``` followed by ```python3 main.py``` to run the main file.