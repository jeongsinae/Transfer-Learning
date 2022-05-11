from __future__ import print_function, division
from torch.hub import load_state_dict_from_url

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import sys
import copy
from tqdm import tqdm


# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {  #이미지 막 변형
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224), #224만큼 띄어내
        transforms.RandomHorizontalFlip(), #이미지뒤집어
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256), #사진크기다시
        transforms.CenterCrop(224), #가운데 띄어내
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = 'data/car_data'
batch_size = 32
#{Key:Value for 요소 in 입력Sequence [if 조건식]},결과로 dict(Key:Value) 리턴
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), #os.path.join(a,b) 파이썬 경로병합, data/Cub200/train 이케
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                             shuffle=True, num_workers=0) #폴더이름으로 만들면 알아서 알긔, 이미지주소 알고잇서
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device(f'cuda:0' if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)
print("GPU_number : ", "0", '\tGPU name:', torch.cuda.get_device_name(torch.cuda.current_device()))

save_path='model/VGG16_True'
os.makedirs(save_path, exist_ok=True)


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
    plt.pause(0.001) 


# 학습 데이터의 배치를 얻습니다.
inputs, classes = next(iter(dataloaders['train']))

# 배치로부터 격자 형태의 이미지를 만듭니다.
out = torchvision.utils.make_grid(inputs)

imshow(out, title=[class_names[x] for x in classes])


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict()) #가중치업데이트
    best_acc = 0.0

    for epoch in tqdm(range(num_epochs)):
        # print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        # print('-' * 10)

        # 각 에폭(epoch)은 학습 단계와 검증 단계
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data. 데이터반복
            for inputs, labels in dataloaders[phase]: #레이블, 정답
                inputs = inputs.to(device) #.to(device) GPU로 실행
                labels = labels.to(device)

                # zero the parameter gradients, 매개변수 경사도 0으로 설정
                optimizer.zero_grad()

                # forward
                # track history if only in train, 학습시에만 연산기록 추정
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward() #역전파
                        optimizer.step() #최적화

                # statistics, 출력용
                running_loss += loss.item() * inputs.size(0) #len은 리스트의 길이(중첩이면 바깥), size는 3*4 2차원이면 12, *inputs.size(0) 크기 맞춰주기
                running_corrects += torch.sum(preds == labels.data)

            # 학습률 관리
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('/n')
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            savepath = save_path + '/new_{}_L1_{}_E_{}.pth' #.pth없으면 권한오류
            # deep copy the model, 최적의 가중치 찾기(?)
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                #torch.save(model.state_dict(), savepath)
                torch.save(model.state_dict(), savepath.format(epoch_loss, epoch_acc, epoch))

        print()

    #시간계산
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model #model 리턴해줘

def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)
        
#합성곱 신경망 미세조정
#model_ft = models.alexnet(pretrained=True)  //model 변경
#num_ftrs = model_ft.classifier[6].in_features

model_ft = models.vgg16(pretrained=True)
num_ftrs = model_ft.classifier[6].in_features

# features = list(model_ft.classifier.children())[:-1] # Remove last layer
# features.extend([nn.Linear(num_ftrs, len(class_names))]) # Add our layer with 4 outputs
# model_ft = nn.Sequential(*features) # Replace the model classifier

model_ft.classifier[6] = nn.Linear(num_ftrs, 196) #Linear(입력, 출력) 폴더개수

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.Adam(model_ft.parameters(), lr=1e-4)
exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer_ft, 100)

#학습 및 평가하기
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=100)
#visualize_model(model_ft)

sys.exit(0)

