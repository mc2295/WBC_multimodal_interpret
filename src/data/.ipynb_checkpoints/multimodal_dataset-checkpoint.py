from torch.utils.data import Dataset
from torchvision import transforms
import torch
from fastai.vision.all import PILImage

class MultimodalDataset(Dataset):
    def __init__(self, df, entry_path, transform= None, valid = False):
        self.transform = transform
        self.df = df
        self.entry_path = entry_path
        self.columns_tabular = [i for i in df.columns if i not in ['cell', 'label', 'dataset', 'Unnamed: 0', 'is_valid', 'index', 'Unnamed: 0.1', 'Unnamed: 0.1.1']]
        # self.dic_label = {k : i for i, k in enumerate(list_labels_cat)}
        self.dic_label = {k : i for i, k in enumerate(np.unique(df.label.tolist()))}
        if valid:
            self.df = df.loc[df['is_valid'] == True]
        else:
            self.df = df.loc[df['is_valid'] == False]
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        
        label_cat = self.df.iloc[idx]['label']
        label = self.dic_label[label_cat]

        img_path = self.df.iloc[idx]['cell']
        img = transforms.ToTensor()(PILImage.create( self.entry_path + img_path))
        
        
        img = transforms.Resize((224, 224))(img)
        if self.transform:
            img = self.transform(img)
        
        tabular = torch.tensor(df.iloc[idx][self.columns_tabular]).to(torch.float32)

        # Get label
        

        return {'name': self.df.iloc[idx]['cell'], 'image': img, 'tabular': tabular, 'label': label}