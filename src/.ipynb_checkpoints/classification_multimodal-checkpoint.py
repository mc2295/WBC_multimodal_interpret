import random
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import pandas as pd
import numpy as np
from fastai.vision.all import PILImage
import torch.optim as optim
from efficientnet_pytorch import EfficientNet
from umap import UMAP
import warnings
from data.features.process_df import split_columns, normalize_df, remove_nan, split_train_valid
warnings.filterwarnings("ignore")

import sys
sys.path.append('/home/manon/segmentation/src')
from visualisation.select_bounding_boxes import  extract_info_from_xml

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
        # Load and preprocess image
        
        label_cat = self.df.iloc[idx]['label']
        label = self.dic_label[label_cat]
        
        # img_path = self.df.iloc[idx]['cell'].split('/')[-1]
        # img_path = '/home/manon/classification/data/Single_cells/tianjin_reviewed_2/' + label_cat + '/' + img_path
        # index_box = img_path.split('.png')[0][-1]
        # bbox_name = '/home/manon/segmentation/data/train/tianjin_YOLO/train/' + img_path.split('-')[1] + '/' + img_path.split('/')[-1][:-6] + '.xml'
        
        # dic = extract_info_from_xml(bbox_name)
        # k = dic['bboxes'][int(index_box)]
        # bbox = np.array([k['xmin'], k['ymin'], k['xmax'], k['ymax']])
        # img = cv2.imread(img_path)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)[bbox[1]:bbox[3],bbox[0]:bbox[2],:]
        img_path = self.df.iloc[idx]['cell']
        img = transforms.ToTensor()(PILImage.create( self.entry_path + img_path))
        
        
        img = transforms.Resize((224, 224))(img)
        if self.transform:
            img = self.transform(img)
        
        tabular = torch.tensor(df.iloc[idx][self.columns_tabular]).to(torch.float32)

        # Get label
        

        return {'name': self.df.iloc[idx]['cell'], 'image': img, 'tabular': tabular, 'label': label}
class MultimodalNetwork(torch.nn.Module):
    def __init__(self, num_tabular_features, num_classes):
        super(MultimodalNetwork, self).__init__()
        # Image processing module (e.g., CNN)
        self.image_module = EfficientNet.from_pretrained('efficientnet-b0')
        self.linear_layer = torch.nn.Linear(1000, 256)
        # Tabular data processing module (e.g., fully connected network)
        self.tabular_module = torch.nn.Sequential(
            torch.nn.Linear(num_tabular_features, 500),
            torch.nn.ReLU(),
            torch.nn.Linear(500, 256),
            torch.nn.ReLU(),
        )

        # Final classification layer
        self.classifier = torch.nn.Linear(256 + 256, num_classes)

    def forward(self, image, tabular):
        image_features = self.image_module(image)
        image_features = self.linear_layer(image_features)
        tabular_features = self.tabular_module(tabular)
        combined_features = torch.cat((image_features, tabular_features), dim=1)
        output = self.classifier(combined_features)
        # output = torch.nn.functional.softmax(output, dim=1)
        return output

source = 'rabin'
df = pd.read_csv('features_'+source+'.csv')
index = [i for i in range(len(df)) if df.cell.tolist()[i].split('/')[-1][:2] == '95' or df.cell.tolist()[i].split('/')[-1][:4] == '2019']
df = df.loc[index, :]


train_dataset = MultimodalDataset(df, '../', valid = False)
val_dataset = MultimodalDataset(df, '../', valid = True)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

num_tabular_features = len(train_dataset.columns_tabular)
num_classes = len(np.unique(df.label.tolist()))
model = MultimodalNetwork(num_tabular_features, num_classes)
model = model.to('cuda')

from tqdm import tqdm 

# Instantiate the model, set the loss function, and the optimizer
criterion = torch.nn.CrossEntropyLoss()
lr = 0.0001
optimizer = optim.Adam(model.parameters(), lr=lr)
torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
# Training loop
num_epochs = 6 # Adjust as needed

for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0

    for batch in tqdm(train_loader):
        images, tabular_data, labels = batch['image'].to('cuda'), batch['tabular'].to('cuda'), batch['label'].to('cuda')

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images, tabular_data)

        # Compute the loss
        loss = criterion(outputs, labels.long())  # Assuming labels are integers

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Print average training loss for the epoch
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}')

# Validation loop (optional)
model.eval()  # Set the model to evaluation mode
correct_predictions = 0
total_samples = 0

with torch.no_grad():
    for batch in val_loader:
        images, tabular_data, labels = batch['image'].to('cuda'), batch['tabular'].to('cuda'), batch['label'].to('cuda')
        outputs = model(images, tabular_data)
        _, predicted = torch.max(outputs, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels.long()).sum().item()

accuracy = correct_predictions / total_samples
print(f'Validation Accuracy: {accuracy * 100:.2f}%')

torch.save(model, '../models/'+source +'_trained/multimodal_lr_' + str(lr)[2:] + '_epoch_' + str(num_epochs) + '_reduced')
