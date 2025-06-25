
import cv2
import os

from segmentation.predict_unet import predict as predict_unet
from segmentation.predict_sam import predict as predict_sam
from data.create_df import create_df
from features.extract_features import save_features
import config
from features.process_df import 



if config.source in ['barcelona', 'rabin']: 
    df = create_df('../references', [source], list_labels_cat)
    filenames = ['../' + i for i in df.name.tolist()]
    labels = df.label.tolist()

elif config.source == 'tianjin':
    filenames = config.filenames


for i in range(len(filenames)):
    if i%10 == 0:
        print(i)
    filename =  filenames[i]
    image = cv2.imread(filename)
    image_cv = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    
    if source in ['barcelona', 'rabin']:
        masks = predict_unet(filename, config.model_path)
        label = labels[i]
    if source == 'tianjin':
        bbox_name = bbox_names[i]
        masks, label = predict_sam(filename, bbox_name)
        
    save_file = "features_{}.csv".format(config.source)

    try:
        save_features(image, masks, filename, label, source, save_file, show_images = False, save_cropped = False)
    except:
        pass