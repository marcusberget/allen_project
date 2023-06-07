import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torchvision import models
from torchvision.io import read_image
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms

class AllenDataset(Dataset):
    def __init__(self, scenes_index, responses, pupil_data, transform=None):
        self.scenes_index = scenes_index
        self.n_samples = responses.shape[0]
        self.transform = transform

        responses = torch.from_numpy(responses)
        responses_mean = responses.mean(dim=0)
        responses_std = responses.std(dim=0)
        responses = (responses - responses_mean) / responses_std
        self.y = responses

        pupil_data = torch.from_numpy(pupil_data)
        pupil_data = pupil_data.float()
        # pupil_data = F.normalize(pupil_data, p=2, dim=1)
        # Replace NaN values with the mean
        for i in range(3):
            non_nan_values = pupil_data[:,i][~torch.isnan(pupil_data[:,i])]
            # mean_value = torch.mean(non_nan_values)
            mean = pupil_data[:,i].nanmean()
            std = non_nan_values.std()
            pupil_data[:,i] = (pupil_data[:,i] - mean) / std

            pupil_data[torch.isnan(pupil_data[:,i])] = 0

        self.pupil_data = pupil_data
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, index):
        img_path_string = f'../scenes/scene_{self.scenes_index[index]}.jpeg'
        img_path = Path(img_path_string)
        # image = read_image(img_path)
        image = Image.open(img_path)
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        image = transform(image)
        image = image.float()
        if self.transform:
            image = self.transform(image)
        responses = self.y[index, :]

        pupil_data = self.pupil_data[index]

        return image, responses, pupil_data
    

class my_vgg(nn.Module):
    def __init__(self, num_neurons, use_pupil_data):
        super(my_vgg, self).__init__()
        self.use_pupil_data = use_pupil_data
        vgg19 = models.vgg19(pretrained=True)
        self.pretrained = vgg19.features[0:36]#torch.load('vgg19_pretrained.pt')
        # print(self.pretrained)
        for param in self.pretrained.parameters():
            param.requires_grad = False
        self.fc1 = nn.Sequential(nn.Linear(4608, 4096), 
                                        nn.Tanh(),
                                        nn.Linear(4096, 4096),
                                        nn.Tanh(),
                                        nn.Linear(4096, 1000),
                                        nn.Tanh(),
                                        nn.Linear(1000, 250),
                                        nn.Tanh(),
                                        nn.Linear(250,num_neurons))
        # for layer in self.fc1:
        #     if isinstance(layer, nn.Linear):
        #         nn.init.xavier_normal_(layer.weight)
        # self.fc1 = nn.Sequential(nn.Linear(1000, 250),
        #                          nn.ReLU(),
        #                          nn.Linear(250, num_neurons))
        if use_pupil_data:
            self.fc2 = nn.Sequential(nn.ReLU(),
                                     nn.Linear(num_neurons + 3, 25),
                                     nn.ReLU(),
                                     nn.Linear(25, num_neurons))
        # else:
            # self.fc2 = nn.Sequential()

    def forward(self, x, pupil_data):
        x = self.pretrained(x)
        # print(self.pretrained.requires_grad_())
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        if self.use_pupil_data:
            # x = F.normalize(x, p=2, dim=0)
            x = torch.cat((x, pupil_data), dim=1)
            x = self.fc2(x)
        return x