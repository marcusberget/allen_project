from torchvision import transforms
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from VGG_model import AllenDataset, my_vgg
import matplotlib.pyplot as plt

def r_squared(y_true, y_pred):
    ssr = torch.sum((y_true - y_pred) ** 2)
    sst = torch.sum((y_true - torch.mean(y_true)) ** 2)
    r2 = 1 - (ssr / sst)
    return r2

num_epochs = 6
batch_size = 16
learning_rate = 5
criterion = nn.MSELoss()


im_labels = np.load('../data/ophys_im_labels.npy')
responses = np.load('../data/ophys_max_events.npy')
responses = responses.reshape((5950, 212))
responses = np.float32(responses)
num_neurons = responses.shape[1]
pupil_data = np.load('../data/ophys_pupil_data.npy')

# scenes_index = np.arange(-1, 118)
# scenes_index = np.repeat(scenes_index, 50)
# scenes_index = scenes_index.astype(str)

X_train, X_test, y_train, y_test = train_test_split(im_labels, responses, test_size=0.2, random_state=33)
_, _, pupil_data_train, pupil_data_test = train_test_split(im_labels, pupil_data, test_size=0.2, random_state=33)

transform = transforms.Compose([transforms.Resize((48,48))])

training_dataset = AllenDataset(X_train, y_train, pupil_data, transform)
test_dataset = AllenDataset(X_test, y_test, pupil_data, transform)

train_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = my_vgg(num_neurons, use_pupil_data=False).to(device)

print("Model is using GPUs:", next(model.parameters()).is_cuda)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)#, weight_decay=1e-4)

n_total_steps = len(train_loader)
loss_list = np.array([])
r2_list = np.array([])
output_list = np.array([])
responses_list = np.array([])

for epoch in range(num_epochs):
    for i, (images, responses, pupil_data) in enumerate(train_loader):

        # origin shape: [4, 3, 32, 32] = 4, 3, 1024
        # input_layer: 3 input channels, 6 output channels, 5 kernel size
        images = images.to(device)
        responses = responses.to(device)

        # Forward pass
        outputs = model(images, pupil_data)
        # print(outputs.dtype, labels.dtype)
        # quit()
        loss = criterion(outputs, responses)

        r2 = r_squared(responses, outputs)

        # plt.plot(responses[:,211].detach())
        # plt.show()
        # input()
        # # print(responses.shape)
        # input()

        if (epoch == 5):
            output_list = np.append(output_list, outputs[:,2].detach().numpy())
            responses_list = np.append(responses_list, responses[:,2].detach().numpy())
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 10 == 0:
            r2_list = np.append(r2_list, r2.item())
            loss_list = np.append(loss_list, loss.item())
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}, R2: {r2.item():.4f}')
            # print(outputs - responses)

print('Finished Training')
np.save('Loss_nopupil_deep.npy', loss_list)
np.save('r2_nopupil_deep.npy', r2_list)
np.save('output_nopupil_deep.npy', output_list)
np.save('responses_nopupil_deep.npy', responses_list)