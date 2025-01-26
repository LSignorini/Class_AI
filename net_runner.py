import os
import shutil
import torch
import time
from datetime import datetime
import numpy as np
import torch.nn as nn
import torch.optim as optim 
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision
from torch.utils.data import DataLoader as torch_dataloader
from tqdm import tqdm

from pathlib import Path

from torch.utils.tensorboard.writer import SummaryWriter 
#from custom_dataset_shapes import CustomDatasetShapes 
#from visual_helper import show_image_grid 
#from net import VGG16
from net import RegNet

from sklearn.metrics import confusion_matrix

# Imposto seed per riproducibilità
torch.manual_seed(42)


class NetRunner():
    """
    Gestisce addestramento e test della rete di classificazione.
    """

    def __init__(self, cfg_object: object, clear_runs=False) -> None:
        """
        Inizializza il gestore e gli attributi necessari all'addestramento.
        
        Args:
            cfg_object (object): Oggetto creato a partire dalla struttura del config.json
        """
        print("Contents of cfg_object:", cfg_object)
        # Raccolgo gli iper-parametri definiti nel config.
        self.batch_size = cfg_object.hyper_parameters.batch_size
        self.lr = cfg_object.hyper_parameters.learning_rate
        self.momentum = cfg_object.hyper_parameters.momentum
        self.epochs = cfg_object.hyper_parameters.epochs
        
        # Raccolgo i parametri definiti nel config per l'early Stopping.
        self.patience = cfg_object.early_stop_parameters.patience

        # Raccolgo i parametri definiti nel config per il salvataggio del modello.
        self.save_model = cfg_object.save_model_parameters.save_model
        self.load_model_tr = cfg_object.save_model_parameters.load_model_tr
        self.load_model_te = cfg_object.save_model_parameters.load_model_te
        self.evaluation_perc = cfg_object.save_model_parameters.evaluation_perc
        self.loss_limit = cfg_object.save_model_parameters.loss_limit
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.runs_path = Path('./runs')
        if clear_runs and self.runs_path.is_dir():
            print("Clearing older runs...")
            shutil.rmtree(self.runs_path)
            time.sleep(5)
            print('Runs cleared!')
        
        timestamp = time.time()
        date_time = datetime.fromtimestamp(timestamp)
        str_date_time = date_time.strftime("%d-%m-%Y_%H-%M")
        name = 'exp_'
        run_name = name + str_date_time

        # Creo un writer dedicato a tenere traccia della rete, degli esperimenti, degli artefatti...
        self.writer = SummaryWriter(f'runs/{run_name}')
        
        # Procedo al caricamento dei dati in un dataset e, di conseguenza, nel dataloader.
        self.load_data(cfg_object.io.training_folder,
                       cfg_object.io.validation_folder,
                       cfg_object.io.test_folder,
                       cfg_object.io.use_custom_generator)
        
        self.outpath_sd = './out/trained_model_sd.pth'
        self.outpath = './out/trained_model.pth'
        
        # Definisco la rete e la salvo in Tensorboard
        self.net = RegNet(num_classes=self.num_classes) #VGG16(num_classes=self.num_classes)
        self.writer.add_graph(self.net, torch.randn(size=(1, 3, 224, 224)))
        self.writer.close()

        # Caricamento modello nel training per riprendere un addestramento da dove era rimasto
        if self.load_model_tr:
            self.net.load_state_dict(torch.load(self.outpath_sd))
        
        # Definisco la funzione di loss.
        self.loss_fn = nn.CrossEntropyLoss()
        
        # Definisco l'ottimizzatore da usare in addestramento.
        # passare learning rate dinamico(link WA) (molto opzionale), momentum tolto, passare numero epoche
        # Sostituire SGD con AdamW
        self.optimizer = optim.AdamW(self.net.parameters(), lr=self.lr)

    def load_data(self, tr_path: str, va_path: str, te_path: str, as_custom: bool = False):
        """
        Carica i dati di training/test dal filesystem con loader custom o meno.
        
        Args:
            tr_path (str): Percorso ai dati di training.
            te_path (str): Percorso ai dati di test. 
            as_custom (bool, optional): Indica se usare o meno il loader custom. Defaults to False.
        """
        
        # Definisco le trasformazioni da applicare ai dati
        self.transforms = RegNet.get_data_transformation()#VGG16.get_data_transformation()

        tr_path, va_path, te_path = Path(tr_path), Path(va_path), Path(te_path)
        
        print(tr_path, va_path, te_path)
        #as_custom_trainset = CustomDatasetShapes(root=tr_path, transform=self.transforms)
        #as_custom_validset = CustomDatasetShapes(root=va_path, transform=self.transforms)
        #as_custom_testset = CustomDatasetShapes(root=te_path, transform=self.transforms)
        trainset = torchvision.datasets.ImageFolder(root= os.path.join('Animal', "train"), transform=self.transforms)        
        validset = torchvision.datasets.ImageFolder(root=os.path.join('Animal', "val"), transform=self.transforms)
        testset = torchvision.datasets.ImageFolder(root= os.path.join('Animal', "test"), transform=self.transforms)
        
        #trainset = as_custom_trainset if as_custom else as_torch_trainset
        #validset = as_custom_validset if as_custom else as_torch_validset
        #testset = as_custom_testset if as_custom else as_torch_testset
        
        self.classes = trainset.classes
        self.num_classes = len(self.classes)
        self.tr_loader = torch_dataloader(trainset, batch_size=self.batch_size, shuffle=True)
        self.va_loader = torch_dataloader(validset, batch_size=self.batch_size, shuffle=False)
        self.te_loader = torch_dataloader(testset, batch_size=self.batch_size, shuffle=False)

    def train(self, preview: bool = False) -> None:
        """
        Esegue l'addestramentoContinua...

della rete.

        Args:
            preview (bool, optional): Indica se mostrare un'anteprima dei dati. Defaults to False.
        """
        
        if preview:
            show_image_grid(next(iter(self.tr_loader)))
        
        self.tr_losses_x, self.tr_losses_y, self.va_losses_x, self.va_losses_y = [], [], [], []
        
        ctr = 0
        ctr_va = 0
        
        # Definisco variabili per l'early Stopping
        ctr_worst = 0
        best_va_loss = float('inf')
        
        # Loop di addestramento per n epoche.
        for epoch in range(self.epochs):
            
            # Definisco variabili per la confusion matrix
            pred_tr = []
            real_tr = []
            pred_va = []
            real_va = []

            tr_running_loss = 0.0
            va_running_loss = 0.0

            # La rete entra in modalità addestramento.
            self.net.train()

            self.net.to(self.device)
            
            for i, data in (pbar := tqdm(enumerate(self.tr_loader, 0), total=len(self.tr_loader.dataset) // self.batch_size, desc='')):
                
                
                
                # Per ogni input tiene conto della sua etichetta.
                inputs, labels = data

                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # Azzeramento dei gradienti.
                self.optimizer.zero_grad()
                
                # L'input attraversa la rete e genera l'output.
                outputs = self.net(inputs)
                
                # Calcolo della funzione di loss sulla base delle predizioni e delle etichette.
                
                loss = self.loss_fn(outputs, labels)
                
                for j in range(len(labels)):
                    pred_tr.append(np.array(outputs.detach()[j].to('cpu')))
                    real_tr.append(np.array(labels.detach()[j].to('cpu')))
                

                # Retropropagazione del gradiente.
                loss.backward()

                # Passo di ottimizzazione.
                self.optimizer.step()

                # Monitoraggio delle statistiche.
                tr_running_loss += loss.item()
                
                self.writer.add_images('input training', inputs, ctr)
                pbar.set_description(f"Loss: {tr_running_loss / (i + 1):.5f}")
                
                ctr += 1
            
            tr_loss = tr_running_loss / len(self.tr_loader)
            self.tr_losses_x.append(epoch)
            self.tr_losses_y.append(tr_loss)

            # Calcolo della confusion matrix utilizzando sklearn.metrics
            cm = confusion_matrix(np.array(real_tr),
                                  np.argmax(np.array(pred_tr), axis=1))
            
            # Trasformazione della confusion matrix in un grafico di matplotlib
            fig = plt.figure()
            ax = fig.add_subplot(111)
            cax = ax.matshow(cm)
            plt.title('Confusion Matrix')
            fig.colorbar(cax)
            plt.xlabel('Predicted')
            plt.ylabel('True')
            
            # Aggiunta del grafico a TensorBoard utilizzando add_figure
            self.writer.add_figure('Confusion Matrix Training', fig, epoch)
            
            plt.close()

            self.net.to(self.device)
            self.net.eval()
            
            for _, data in enumerate(self.va_loader, 0):
                
                # Per ogni input tiene conto della sua etichetta.
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                with torch.no_grad():
                    
                    # L'input attraversa la rete e genera l'output.
                    outputs = self.net(inputs)
                
                    # Calcolo della funzione di loss sulla base delle predizioni e delle etichette.
                    loss = self.loss_fn(outputs, labels)
                
                    for j in range(len(outputs)):
                        pred_va.append(np.array(outputs.detach()[j].to('cpu')))
                        real_va.append(np.array(labels.detach()[j].to('cpu')))

                    # Monitoraggio delle statistiche.
                    va_running_loss += loss.item()
                    
                    self.writer.add_images('input validation', inputs, ctr_va)
                    
                ctr_va += 1

            va_loss = va_running_loss / len(self.va_loader)
            self.va_losses_x.append(epoch)
            self.va_losses_y.append(va_loss)
            
            # Calcolo della confusion matrix utilizzando sklearn.metrics
            cm = confusion_matrix(np.array(real_va),
                                  np.argmax(np.array(pred_va), axis=1))

            # Trasformazione della confusion matrix
            fig = plt.figure()
            ax = fig.add_subplot(111)
            cax = ax.matshow(cm)
            plt.title('Confusion Matrix')
            fig.colorbar(cax)
            plt.xlabel('Predicted')
            plt.ylabel('True')
            
            # Aggiunta del grafico a TensorBoard utilizzando add_figure
            self.writer.add_figure('Confusion Matrix Validazione', fig, epoch)
            
            plt.close()
            
            print(f'Epoca: {epoch + 1:3d} / training loss: {tr_loss:.6f} ---- validation loss: {va_loss:.6f}')
            
            self.writer.add_scalars('Losses', {'Training loss': tr_loss, 'Validation loss': va_loss}, epoch)
            
            # Condizione per il salvataggio del miglior modello
            if self.save_model:
                if tr_loss <= self.loss_limit and va_loss <= self.loss_limit and va_loss < tr_loss * (1 + self.evaluation_perc):
                    torch.save(self.net.state_dict(), self.outpath_sd)
                    torch.save(self.net, self.outpath)
            
            # Implementazione dell'early stopping che interrompe l'addestramento se la loss non migliora per patience epoche consecutive
            if va_loss < best_va_loss:
                best_va_loss = va_loss
                ctr_worst = 0
            else:
                ctr_worst += 1
            
            if ctr_worst >= self.patience:
                print('Stop training per earlyStopping')
                break
                
        print('Finished Training.')
        print('Model saved.')
        
        plt.plot(self.tr_losses_x, self.tr_losses_y, label='Training losses')
        plt.plot(self.va_losses_x, self.va_losses_y, label='Validation losses')
        plt.xlabel('Epoche')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    def test(self, preview: bool = False):
        """
        Esegue un test della rete
        
        Args:
            preview (bool, optional): Indica se mostrare un'anteprima dei dati. Defaults to False.
        """
        if preview:
            show_image_grid(next(iter(self.te_loader)))
        
        net = RegNet(num_classes=len(self.classes))#VGG16(num_classes=len(self.classes))

        if self.load_model_te:
            net.load_state_dict(torch.load(self.outpath_sd))
        
        total, correct = 0, 0
        correct_pred = {classname: 0 for classname in self.classes}
        total_pred = {classname: 0 for classname in self.classes}
        
        # La rete entra in modalità di valutazione
        net.eval()

        ctr = 0

        embeddings_list = None
        images_list = None
        labels_list = None
        
        # Definisco variabili per la confusion matrix
        pred_te = []
        real_te = []

        # Non è necessario calcolare i gradienti durante il passaggio dei dati nella rete
        with torch.no_grad():
        
            # Cicla sui campioni di test, batch per volta.
            for i, data in enumerate(self.te_loader, 0):
                
                # Dal batch si estraggono i dati e le etichette.
                images, labels = data

                # I dati passano attraverso la rete e generano l'output.
                outputs, embeddings = net(images)

                for j in range(len(labels)):
                    pred_te.append(np.array(outputs.detach()[j]))
                    real_te.append(np.array(labels.detach()[j]))
                
                if i == 0:
                    embeddings_list = embeddings
                    images_list = images
                    labels_list = labels
                else:
                    embeddings_list = torch.cat((embeddings_list, embeddings), dim=0)
                    images_list = torch.cat((images_list, images), dim=0)
                    labels_list = torch.cat((labels_list, labels), dim=0)
                
                # Dall'output si ottiene la predizione finale.
                _, predicted = torch.max(outputs.data, 1)
                
                # Aggiornamento dei totali e dei corretti
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                for label, prediction in zip(labels, predicted):
                    if label == prediction:
                        correct_pred[self.classes[label]] += 1
                    total_pred[self.classes[label]] += 1
                
                self.writer.add_images('images test', images, ctr)

                ctr += 1

        embs = torch.mean(embeddings_list, 1)
        class_labels = [str(lab.item()) for lab in labels_list]
        self.writer.add_embedding(embs.reshape(embs.shape[0], -1), metadata=class_labels, label_img=images_list) 
        
        # Calcolo della confusion matrix utilizzando sklearn.metrics
        cm = confusion_matrix(np.array(real_te),
                              np.argmax(np.array(pred_te), axis=1))
        
        # Trasformazione della confusion matrix
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(cm)
        plt.title('Confusion Matrix')
        fig.colorbar(cax)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        
        # Aggiunta del grafico a TensorBoard utilizzando add_figure
        self.writer.add_figure('Confusion Matrix Test', fig, 0)

        print(f'Total network accuracy: {100 * correct // total} %')
        
        for classname, correct_count in correct_pred.items():
            accuracy = 100 * float(correct_count) / total_pred[classname]
            print(f'Class accuracy: {classname:5s} is {accuracy:.1f} %')
