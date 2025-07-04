import glob

source = 'tianjin'
save_overlay = False
list_labels_cat = ['basophil', 'eosinophil', 'erythroblast', 'lymphocyte', 'monocyte', 'neutrophil']

lr = 0.0001
num_epochs = 6
batch_size = 32

if source == 'tianjin':
    images_test_path = 'data/test/tianjin/'

    filenames = [i for i in glob.glob('data/train/tianjin_YOLO/test/*') if i[-4:] == '.png']
    bbox_names = ['data/train/tianjin_YOLO/test/'+ image_name.split('/')[-1][:-4] + '.xml' for image_name in image_names]

    

elif source == 'barcelona': 

    model_path = 'model/barcelona/unet_3_classes_without_transform' 
    images_test_path = 'data/test/barcelona/'


elif source == 'rabin': 

    model_path = 'model/rabin/unet_3_classes_without_transform' 
    images_test_path = 'data/test/rabin/'
    