import pandas as pd
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
import torch
from features.process_df import process_df
from data.multimodal_dataset import MultimodalDataset
from model.multimodal_network import MultimodalNetwork
from tqdm import tqdm 
import config

def train(df, train_dataset,train_loader,val_loader, model_path):
    num_tabular_features = len(train_dataset.columns_tabular)
    num_classes = len(np.unique(df.label.tolist()))
    model = MultimodalNetwork(num_tabular_features, num_classes)
    model = model.to('cuda')

    
    criterion = torch.nn.CrossEntropyLoss()
    lr = config.lr
    optimizer = optim.Adam(model.parameters(), lr=lr)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
    # Training loop
    num_epochs = config.num_epochs 
    for epoch in range(num_epochs):
        model.train()  
        running_loss = 0.0
    
        for batch in tqdm(train_loader):
            images, tabular_data, labels = batch['image'].to('cuda'), batch['tabular'].to('cuda'), batch['label'].to('cuda')
            
            optimizer.zero_grad()                
            outputs = model(images, tabular_data)                
            loss = criterion(outputs, labels.long())  # Assuming labels are integers              
            loss.backward()
            optimizer.step()    
            running_loss += loss.item()
            
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}')
    
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
    
    torch.save(model,model_path )





def eval(model_path, val_loader):
    model = torch.load(model_path)
    label = []
    pred = []
    with torch.no_grad():
        for batch in val_loader:
            images, tabular_data, labels = batch['image'], batch['tabular'], batch['label']
            print(tabular_data.shape)
            outputs = model(images, tabular_data)
            label.append(labels)
            pred.append(outputs.argmax(dim=1))

    label_out = [i.item() for k in label for i in k]
    pred_out = [i.item() for k in pred for i in k]
    df = {'pred': pred_out, 'true': label_out}
    df.to_csv('../out/predictions_{}'.format(config.source))


df = pd.read_csv("features_{}.csv".format(config.source))
df = process_df(df)
#df.to_csv("features_{}.csv".format(config.source))
train_dataset = MultimodalDataset(df, '../', valid = False)
val_dataset = MultimodalDataset(df, '../', valid = True)

train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

model_path = '../models/' + config.source +'_trained/multimodal_lr_' + str(lr)[2:] + '_epoch_' + str(num_epochs) + '_reduced'