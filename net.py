import torch
import torch.nn as nn
import torchvision.transforms as transforms
from pytorch_model_summary import summary
import torchvision.models as models


class Net(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(Net, self).__init__()

        self.branch1 = nn.Conv2d(in_channels, ch1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, ch3x3red, kernel_size=1),
            nn.Conv2d(ch3x3red, ch3x3, kernel_size=3, padding=1)
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, ch5x5red, kernel_size=1),
            nn.Conv2d(ch5x5red, ch5x5, kernel_size=5, padding=2)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self, x):
        print("Input shape:", x.shape)
        branch1_output = self.branch1(x)
        print("Branch 1 output shape:", branch1_output.shape)
        branch2_output = self.branch2(x)
        print("Branch 2 output shape:", branch2_output.shape)
        branch3_output = self.branch3(x)
        print("Branch 3 output shape:", branch3_output.shape)
        branch4_output = self.branch4(x)
        print("Branch 4 output shape:", branch4_output.shape)

        output = torch.cat([branch1_output, branch2_output, branch3_output, branch4_output], dim=1)
        print("Final output shape:", output.shape)
        return output


'''class InceptionNet(nn.Module):
    def __init__(self, num_classes: int, dropout: float=0.5):
        super(InceptionNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            Net(192, 64, 96, 128, 16, 32, 32),
            Net(256, 128, 128, 192, 32, 96, 64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            Net(480, 192, 96, 208, 16, 48, 64),
            Net(512, 160, 112, 224, 24, 64, 64),
            Net(512, 128, 128, 256, 24, 64, 64),
            Net(512, 112, 144, 288, 32, 64, 64),
            Net(528, 256, 160, 320, 32, 128, 128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            Net(832, 256, 160, 320, 32, 128, 128),
            Net(832, 384, 192, 384, 48, 128, 128)
            #Net(832, 384, 192, 384, 48, 128, 128)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(1024 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )
    
    def __init__(self, num_classes: int, dropout: float=0.5):
        super(InceptionNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.net1 = Net(192, 64, 96, 128, 16, 32, 32)
        self.net2 = Net(256, 128, 128, 192, 32, 96, 64)
        self.net3 = Net(480, 192, 96, 208, 16, 48, 64)
        self.net4 = Net(512, 160, 112, 224, 24, 64, 64)
        self.net5 = Net(512, 128, 128, 256, 24, 64, 64)
        self.net6 = Net(512, 112, 144, 288, 32, 64, 64)
        self.net7 = Net(528, 256, 160, 320, 32, 128, 128)
        self.net8 = Net(832, 256, 160, 320, 32, 128, 128)
        self.net9 = Net(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.Sequential(nn.AdaptiveMaxPool2d((7, 7)), nn.Flatten())

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(9408, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
            nn.Softmax(1)
        )

    @staticmethod
    def get_data_transformation():
        """
        Restituisce un oggetto di trasformazione da applicare ai dati.
        """
        transformations = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transformations

    def forward(self, x):
        print("Input shape:", x.shape)  # Stampa la forma dell'input
        x = self.features(x)
        print("Shape after features:", x.shape)  # Stampa la forma dopo il passaggio attraverso "features"
        x = self.avgpool(x)
        #x = torch.flatten(x, 1)
        print("Shape before classifier:", x.shape)  # Stampa la forma prima del passaggio attraverso "classifier"
        x = self.classifier(x)
        print("Shape after classifier:", x.shape)  # Stampa la forma dopo il passaggio attraverso "classifier"
        return x'''

class RegNet(nn.Module):
    def __init__(self, num_classes=10):
        
        super(RegNet, self).__init__()
        # Utilizziamo RegNet con una piccola architettura, ad esempio RegNetY_400MF
        self.model = models.regnet_y_400mf(weights=None)

        # Personalizzazione dell'ultimo livello di classificazione se necessario
        # Numero di classi per la classificazione (ad esempio, 10 classi)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    @staticmethod
    def get_data_transformation():
        """
        Restituisce un oggetto di trasformazione da applicare ai dati.
        """
        transformations = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize the image
            transforms.ToTensor(),          # Convert the image to a tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Normalize
        ])
        
        return transformations

    def forward(self, x):
        return self.model(x)

""" class VGG16(nn.Module):

    def __init__(self, num_classes=10):
        super(VGG16, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer7 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer8 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer9 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer10 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer11 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer12 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer13 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(7 * 7 * 512, 4096),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Linear(4096, num_classes))

    @staticmethod
    def get_data_transformation():
        Restituisce un oggetto di trasformazione da applicare ai dati.
        
        transformations = transforms.Compose([
            transforms.Resize((227, 227)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        ])
        return transformations

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out = self.layer10(out)
        out = self.layer11(out)
        out = self.layer12(out)
        out = self.layer13(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
        print(out)
        return out """

if __name__ == '__main__':
    
    batch_size = 1
    channels = 3
    width = 224
    height = 224

    # Definisci la dimensione dell'input che avrà la rete.
    input_shape = (batch_size, channels, width, height)

    # Crea l'oggetto che rappresenta la rete.
    # Fornisci il numero di classi.
    model = RegNet(num_classes=3)

    # Stampa delle dimensioni dei tensori all'interno dei moduli Inception
    #for layer in model.features:
        #if isinstance(layer, Net):
            #print(layer.__class__.__name__)
            #x = torch.randn((batch_size, 3, 224, 224))  # Esempio di input
            #print("Input shape:", x.shape)
            #x = layer(x)
            #print("Output shape:", x.shape)

    # Stampa delle dimensioni dei tensori dopo i moduli Inception
    x = torch.randn((batch_size, 3, 224, 224))  # Esempio di input
    x = model.features(x)
    print("Output shape after Inception modules:", x.shape)

    # Salva i parametri addestrati della rete.
    torch.save(model.state_dict(), './out/model_state_dict.pth')

    # Salva l'intero modello.
    torch.save(model, './out/model.pth')

    # Stampa informazioni generali sul modello.
    print(model)

    # Stampa un recap del modello.
    print(summary(model, torch.ones(size=input_shape)))