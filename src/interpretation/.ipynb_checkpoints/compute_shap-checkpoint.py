from model.multimodal_network import ModelTabular, MultimodalNetwork
from data.multimodal_dataset import MultimodalDataset
import pandas as pd
import config
import pickle
import shap

def compute_shap(source):
    df = pd.read_csv("features_{}.csv".format(source))
    train_dataset = MultimodalDataset(df, '../', valid = False)
    val_dataset = MultimodalDataset(df, '../', valid = True)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    model = torch.load('../models/{}_trained/multimodal_all_classes_normalised'.format(source))
    model_tab = ModelTabular(model, df)
    
    model = model.to('cpu')
    image_out = []
    
    X_imgs = []
    tabulars = []
    with torch.no_grad():
        for batch in tqdm(val_loader):
            X_img = model.image_module(batch['image'])
            X_img = model.linear_layer(X_img).cpu()
            X_imgs.append(X_img)
            tabulars.append(batch['tabular'])
    X_imgs = np.vstack(X_imgs)
    tabulars = np.vstack(tabulars)
    X = torch.cat((torch.Tensor(X_imgs), torch.Tensor(tabulars)), dim = 1)
    
    torch.save(X, 'X_features_{}'.format(source))
    print('creation of shap explainer')
    explainer = shap.DeepExplainer(model_tab, X)
    
    print('shap values')
    shap_values = explainer.shap_values(X)
    
    file = open('shap_{}'.format(source), 'wb')
    pickle.dump(shap_values, file)