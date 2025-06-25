import torch
from efficientnet_pytorch import EfficientNet

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

class ModelTabular(torch.nn.Module):
    def __init__(self, mm_model, df):
        super(ModelTabular, self).__init__()
        self.mm_model = mm_model
        self.dic_labels = {i:k for i, k in enumerate(np.unique(df.label.tolist()))}
    def forward(self, x):
        x_tab = x[:, 256:]

        x_tab = self.mm_model.tabular_module(x_tab)
        combined_features = torch.cat((x[:, :256], x_tab), dim = 1)
        output = self.mm_model.classifier(combined_features)
        return output
    def predict(self, x):
        out = self.forward(x)
        return [self.dic_labels[torch.argmax(out[i])] for i in range(len(out))]