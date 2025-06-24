import pandas as pd

def create_df(reference_path, source, list_labels_cat, new_dic = None):
    if new_dic is not None:
        dic_classes = new_dic
    else:
        file = open(reference_path + '/variables/dic_classes.obj', 'rb')
        dic_classes = pickle.load(file)
    dataframe = pd.read_csv(reference_path + '/variables/dataframes/df_labeled_images.csv') 
    source_mask = dataframe.image_dataset.isin(source) 
    dataframe = dataframe.loc[source_mask,['image_path', 'image_class', 'image_dataset']]
    dataframe.image_class = [dic_classes[x] for x in dataframe.image_class]  
    label_mask = dataframe.image_class.isin(list_labels_cat) 
    dataframe = dataframe.loc[label_mask]

    dataframe = dataframe.reset_index()
    dataframe= dataframe.rename(columns={'image_path': 'name', 'image_class': 'label', 'image_dataset': 'dataset'})
    return dataframe