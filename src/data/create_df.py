import pandas as pd
import pickle

def create_df(reference_path, source, list_labels_cat):
    '''
    input: 
    - reference_path : the path to go to 'reference' folder, 
    - source : the name of the dataset(s), ex : ['barcelona', 'saint_antoine']
    - list_label_cat:list of labels, ex: ['basophil', 'eosinophil', 'eyrtroblast', 'monocyte']
    return: 
    - a dataframe [name, label, dataset], ex: ['data/Single_cells/matek/NGB/NGB_01600.jpg', 'neutrophil', 'matek'] with normalised label, and images from source dataset.
    '''

    file = open(reference_path + '/variables/dic_classes.obj', 'rb')
    dic_classes = pickle.load(file)
    # dic_classes = new_dic
    dataframe = pd.read_csv(reference_path + '/variables/dataframes/df_labeled_images.csv')  # dataframe with : [image_path,image_name,image_class,image_dataset,size,transformed_image_path]
    source_mask = dataframe.image_dataset.isin(source) # take images from the dataset source
    dataframe = dataframe.loc[source_mask,['image_path', 'image_class', 'image_dataset']] #keep only columns 'image_path', 'image_class', 'image_dataset'
    dataframe.image_class = [dic_classes[x] for x in dataframe.image_class]  # maps the class names to corresponding normalised name
    label_mask = dataframe.image_class.isin(list_labels_cat) # keeps only classes that are in list_labels_cat
    dataframe = dataframe.loc[label_mask]

    dataframe = dataframe.reset_index()
    dataframe= dataframe.rename(columns={'image_path': 'name', 'image_class': 'label', 'image_dataset': 'dataset'})
    return dataframe