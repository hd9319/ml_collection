import os

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class PokemonDataset(Dataset):
    """Pokemon Dataset"""

    def __init__(self, dataset, image_dir, transform=None):
        """
        Args:
            dataset (pd.DataFrame)
            image_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.pokemon_data = dataset
        self.image_dir = image_dir
        self.transform = transform
        self.attributes_list = ['Grass', 'Fire', 'Water', 'Bug', 'Normal', 'Poison', 'Electric',
       'Ground', 'Fairy', 'Fighting', 'Psychic', 'Rock', 'Ghost', 'Ice',
       'Dragon', 'Dark', 'Steel', 'Flying']
        self.attributes_dict = self._attributes_to_dict(self.attributes_list)
        self.y = self._class_to_onehot(self.pokemon_data['Type1'])
    
    @staticmethod
    def _attributes_to_dict(attributes):
        return {attribute: idx for idx, attribute in enumerate(attributes)}
    
    @staticmethod
    def _load_img(file_name):
        img = plt.imread(file_name)
        return img
            
    
    def _class_to_onehot(self, classes):
        indices = classes.map(self.attributes_dict)
        y = np.eye(len(self.attributes_list))[indices]
        return y

    def __len__(self):
        return len(self.pokemon_data)

    def __getitem__(self, idx):
        y = self.y[idx]
        
        try:
            img_name = os.path.join(self.image_dir,
                                    str(self.pokemon_data.iloc[idx]['Name']) + '.png')
            img = self._load_img(img_name)
        except OSError as e:
            img_name = os.path.join(self.image_dir,
                                    str(self.pokemon_data.iloc[idx]['Name']) + '.jpg')
            img = self._load_img(img_name)


        if self.transform:
            img = self.transform(img)
        
        return (img, torch.Tensor(y))

class TypeClassifier(nn.Module):
    """
    Convolutional Neural Network
    """
    def __init__(self):
        super().__init__()  # initialize parent class
        
        self.conv1 = torch.nn.Conv2d(4, 18, kernel_size=3, stride=1, padding=1)  # 18, 120, 120
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # 18, 60, 60
        self.conv2 = torch.nn.Conv2d(18, 32, kernel_size=3, stride=1, padding=1)  # 32, 60, 60
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # 32, 30, 30
        self.fc1 = torch.nn.Linear(32 * 30 * 30, 64)
        self.fc2 = torch.nn.Linear(64, len(train_pokemon.attributes_list))
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        
        x = x.view(-1, 32 * 30 * 30)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return(x)

def train_model(pokemon_file, model_path, epochs=5, test_size=0.3):
	pokemon = pd.read_csv(pokemon_file)
	x_train, x_test, _, _ = train_test_split(pokemon, pokemon, 
	                                         test_size=test_size, 
	                                         shuffle=True, 
	                                         stratify=pokemon['Type1'])

	transformations = transforms.Compose([
	                                        transforms.ToTensor(),
	                                        transforms.Normalize(mean=(0.5, 0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5, 0.5))

	                                    ])

	train_pokemon = PokemonDataset(x_train, image_dir, transform=transformations)
	test_pokemon = PokemonDataset(x_test, image_dir, transform=transformations)

	train_loader = DataLoader(train_pokemon, batch_size=BATCH_SIZE, shuffle=True)
	test_loader = DataLoader(train_pokemon, batch_size=BATCH_SIZE, shuffle=True)

	type_classifier = TypeClassifier()
	optimizer = optim.Adam(type_classifier.parameters(), lr=LEARNING_RATE)
	criterion = nn.MSELoss()

	if os.path.isfile(model_path):
		print('Loading Model: %s' model_path)
        checkpoint = torch.load(filename)
        type_classifier.load_state_dict(checkpoint)

	for epoch in range(epochs):
	    total_loss = 0
	    print('Starting Epoch: %s' % str(epoch + 1))
	    for X, y in train_loader:
	        type_classifier.zero_grad()
	        output = type_classifier(X)
	        loss = criterion(output, y)
	        total_loss += loss.item()
	        loss.backward()
	    print(total_loss)

	torch.save(type_classifier.state_dict(), model_path)











