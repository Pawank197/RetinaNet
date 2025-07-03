The repo contains two folders: data and src. 

Data directory contains the annotations, imgs and labels.csv file

The src directory contains 3 sub directories: Dataset, models, utils. 
Dataset: It contains the files for loading the data in a model-readable format(dataset.py and transforms.py). It also provides method for visualizing the augmentation we are applying(transformsVisualization.py)
models: It contains the method for creating the model(retina-net in this case) and changing its anchors.
