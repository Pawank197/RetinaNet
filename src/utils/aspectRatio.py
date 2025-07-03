import pandas as pd

"""
This script calculates the average aspect ratio, height and width of bounding boxes in the dataset.
This was used in setting the anchor sizes and aspect ratios
"""

def calculateAvgAspectRatioHeightWidth(annotation_path):
    df = pd.read_csv(
        annotation_path,
        header=None,
        names=['image_path','xmin','ymin','xmax','ymax','label']
    )
    
    df['width']  = df['xmax'] - df['xmin']
    df['height'] = df['ymax'] - df['ymin']
    
    df = df[(df['width'] > 0) & (df['height'] > 0)]
    
    df['aspect_ratio'] = df['width'] / df['height']
    ans = {
        "aspect_ratio": df['aspect_ratio'].mean(),
        "height": df['height'].mean(),
        "width" : df['width'].mean() 
    }
    return ans


print("Train:", calculateAvgAspectRatioHeightWidth('data/annotations/train_annotations.csv'))
print("Val:  ", calculateAvgAspectRatioHeightWidth('data/annotations/val_annotations.csv'))
print("Test: ", calculateAvgAspectRatioHeightWidth('data/annotations/test_annotations.csv'))

"""
Results:
Train: {'aspect_ratio': np.float64(0.9123049557875287), 'height': np.float64(53.95754461871282), 'width': np.float64(44.979628628087255)}
Val:   {'aspect_ratio': np.float64(1.0514304426389827), 'height': np.float64(50.74631483166515), 'width': np.float64(49.67206551410373)}
Test:  {'aspect_ratio': np.float64(1.0338770442261975), 'height': np.float64(50.89265536723164), 'width': np.float64(49.441771459814106)}
"""
