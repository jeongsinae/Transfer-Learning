# here are the necessary imports
from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "6"
device = torch.device(f'cuda:0' if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)
print("GPU_number : ", "6", '\tGPU name:', torch.cuda.get_device_name(torch.cuda.current_device()))

num_output = 196
num_epochs = 100

model_load = True
model_name = 'new_4.812826503644074_L1_0.04477055092650167_E_21.pth'
mode = 'val'

model_folder = 'model/VGG16_False/'

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

def test_model(model, criterion, optimizer):
    running_corrects = 0

    model.eval()
    for inputs, labels in dataloders['val']:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        with torch.set_grad_enabled(False):
            outputs = model(inputs)

            _, preds = torch.max(outputs.data, 1)

        # statistics
        running_corrects += torch.sum((preds == labels.data).float())

    epoch_acc = running_corrects / dataset_sizes['val']

    print(running_corrects)

    print('Test Acc: {:.4f}'.format(epoch_acc))

def train_model(model, criterion, optimizer, scheduler, num_epochs=10):
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                # scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for data in dataloders[phase]:
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                inputs = Variable(inputs.to(device))
                labels = Variable(labels.to(device))

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)

                    _, preds = torch.max(outputs.data, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        scheduler.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum((preds == labels.data).float())

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
                state = {'model': model_ft.state_dict(), 'optim': optimizer_ft.state_dict()}
                torch.save(state, model_folder + model_name)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best test Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def visualize_model(model, num_images=8):
    images_so_far = 0
    model.eval()
    fig = plt.figure()

    for i, data in enumerate(dataloders['test']):
        inputs, labels = data
        # print(labels)
        inputs, labels = Variable(inputs.to(device)), Variable(labels.to(device))
        # print(labels)
        # _, lab = torch.max(labels.data, 1)
        outputs = model(inputs)
        # print(outputs)
        _, preds = torch.max(outputs.data, 1)
        # print(preds)

        for j in range(inputs.size()[0]):
            images_so_far += 1
            ax = plt.subplot(num_images // 2, 2, images_so_far)
            ax.axis('off')
            ax.set_title('class: {} predicted: {}'.format(class_names[labels.data[j]], class_names[preds[j]]))
            imshow(inputs.cpu().data[j])

            if images_so_far == num_images:
                return


if __name__ == '__main__':

    plt.ion()

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    data_dir = 'data/car_data'
    # loading datasets with PyTorch ImageFolder
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'val']}
    # defining data loaders to load data using image_datasets and transforms, here we also specify batch size for the mini batch
    dataloders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32,
                                                 shuffle=True, num_workers=0)
                  for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    # Get a batch of training data
    # inputs, classes = next(iter(dataloders['train']))
    #
    # # Make a grid from batch
    # out = torchvision.utils.make_grid(inputs)

    # imshow(out, title=[class_names[x] for x in classes])

    # Finetuning the convnet Load a pretrained Resnet 18 model and reset final fully connected layer.

    # model_ft = models.densenet161(pretrained=True)
    # num_ftrs = model_ft.classifier.in_features
    # model_ft.classifier = nn.Linear(num_ftrs, num_output)
    model_ft = models.vgg16(pretrained=False)
    num_ftrs = model_ft.classifier[6].in_features
    # features = list(model_ft.classifier.children())[:-1] # Remove last layer
    # features.extend([nn.Linear(num_ftrs, len(class_names))]) # Add our layer with 4 outputs
    # model_ft = nn.Sequential(*features) # Replace the model classifier

    model_ft.classifier[6] = nn.Linear(num_ftrs, 196)  # Linear(입력, 출력) 폴더갯수 200개씩임!

    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=1e-4)
    exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer_ft, 100)  # 100이 뭐지

    if model_load == True:
        # checkpoint = torch.load('path to model')
        model_ft.load_state_dict(torch.load(model_folder + model_name))
        #model_ft.load_state_dict(checkpoint['model'])
        #optimizer_ft.load_state_dict(checkpoint['optim'])

    # Train and evaluate
    if mode == 'val':
        test_model(model_ft, criterion, optimizer_ft)
