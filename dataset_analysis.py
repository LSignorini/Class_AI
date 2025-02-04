import os
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from torchvision import datasets, transforms
import random
import torch
from PIL import Image

# Imposta i percorsi ai dataset
DATASET_PATH = "./Animal"
TRAIN_PATH = os.path.join(DATASET_PATH, "train")
VAL_PATH = os.path.join(DATASET_PATH, "val")
TEST_PATH = os.path.join(DATASET_PATH, "test")

# Definisci le classi
CLASSES = ["cat", "dog", "elephant", "horse", "lion"]

# Funzione per contare il numero di immagini per classe
def count_images_per_class(path):
    class_counts = {cls: len(os.listdir(os.path.join(path, cls))) for cls in CLASSES}
    return class_counts

# Conta le immagini in ogni set
train_counts = count_images_per_class(TRAIN_PATH)
val_counts = count_images_per_class(VAL_PATH)
test_counts = count_images_per_class(TEST_PATH)

# Calcola i totali per ogni set
def print_class_counts(set_name, counts):
    total = sum(counts.values())
    print(f"Numero di immagini per classe nel {set_name} set ({total} totali):", counts)

# Stampa i risultati
print_class_counts("training", train_counts)
print_class_counts("validation", val_counts)
print_class_counts("test", test_counts)

# Creazione di un grafico della distribuzione
def plot_distribution(counts, title, folder_path):
    plt.figure(figsize=(8, 5))
    sns.barplot(x=list(counts.keys()), y=list(counts.values()), palette="viridis")
    plt.title(title)
    plt.xlabel("Classi")
    plt.ylabel("Numero di immagini")
    plt.savefig(os.path.join(folder_path, f"{title.lower().replace(' ', '_')}_distribution.png"))
    plt.close()

# Crea la cartella per salvare i grafici (eliminando la cartella esistente)
def create_analysis_folder():
    folder_path = "AnalisiDataset"
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)  # Elimina la cartella esistente
    os.makedirs(folder_path)  # Crea una nuova cartella
    return folder_path

folder_path = create_analysis_folder()

plot_distribution(train_counts, "Distribuzione delle immagini nel Training Set", folder_path)
plot_distribution(val_counts, "Distribuzione delle immagini nel Validation Set", folder_path)
plot_distribution(test_counts, "Distribuzione delle immagini nel Test Set", folder_path)

# Visualizza 3 immagini per ogni classe e salvale in una cartella separata per ogni set
def show_sample_images(path, num_samples=3, folder_path=None, set_name=""):
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    fig, axes = plt.subplots(len(CLASSES), num_samples, figsize=(10, 10))
    
    for i, cls in enumerate(CLASSES):
        cls_path = os.path.join(path, cls)
        images = random.sample(os.listdir(cls_path), num_samples)
        
        for j, img_name in enumerate(images):
            img_path = os.path.join(cls_path, img_name)
            image = Image.open(img_path).convert("RGB")
            image = transform(image)
            
            axes[i, j].imshow(image.permute(1, 2, 0))
            axes[i, j].axis("off")
            if j == 0:
                axes[i, j].set_title(cls, fontsize=12)

    plt.tight_layout()
    if folder_path:
        plt.savefig(os.path.join(folder_path, f"{set_name}_sample_images.png"))
    plt.close()

# Salva le immagini di anteprima per ogni set
show_sample_images(TRAIN_PATH, folder_path=folder_path, set_name="training")
show_sample_images(VAL_PATH, folder_path=folder_path, set_name="validation")
show_sample_images(TEST_PATH, folder_path=folder_path, set_name="test")
