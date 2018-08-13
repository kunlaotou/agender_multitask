# -*- coding:UTF-8 -*-
import time

import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import transforms, models
import torch.nn.functional as F


from random_erasing import RandomErasing
from ResNet50 import MyResNet50
from dataset import PETAData
from loss import AGLoss

from train_config import args, params

grad_list = []
def print_grad(grad):
    print(grad)
    grad_list.append(grad)

data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224,224)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.1,0.1,0.1,0),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            RandomErasing(probability=0.5, mean=[0.0, 0.0, 0.0])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            
            
        ]),
    }

train_datasets = PETAData(txt_path='data_list/PETA_RAP_train.txt',dataset='train', data_transforms = data_transforms)
test_datasets = PETAData(txt_path='data_list/PETA_RAP_test.txt', dataset='val', data_transforms = data_transforms)

image_datasets = {'train' : train_datasets, 'val' : test_datasets}

dataloders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                             batch_size=args.batch_size,
                                             shuffle=True) for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

# use pre_train
if args.pre_trained:
  print("Using cache")
  # model = models.densenet121(pretrained=True)
  model = MyResNet50()
  model.load_state_dict(torch.load('pre_train/PETA_RAP_ResNet50_params150.pkl'))
else:
  # model = MyDensenet121()
  model = MyResNet50()
  
# use GPU or multi-GPU
if args.cuda:
    ts = time.time()
    model.cuda()
    
    if args.multi_gpu:
        num_gpu = list(range(torch.cuda.device_count()))
        model = nn.DataParallel(model, device_ids=num_gpu)
    print("Finish cuda loading, time elapsed {}".format(time.time() - ts))

# define loss & optimizer
if args.optimizer == 'Adam':
  optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.w_decay)
elif args.optimizer == 'SGD':
  optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.w_decay)
elif args.optimizer == 'RMSprop':
  optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.w_decay)
  
model_scheduler = MultiStepLR(optimizer, milestones=[20,100,200],  gamma=0.1)

criterion = AGLoss()

def train(epoch):
    since = time.time()
    begin_time = time.time()
    idx = 0
    print('\nEpoch: %d' % epoch)
    model.train(True)
    
    train_loss = 0
    gender_running_loss = 0
    age_running_loss = 0
    Tshirt_running_loss = 0
    jacket_running_loss = 0
    skirt_running_loss = 0
    trousers_running_loss = 0

    
    gender_running_corrects = 0
    age_running_corrects = 0
    Tshirt_running_corrects = 0
    jacket_running_corrects = 0
    skirt_running_corrects = 0 
    trousers_running_corrects = 0

    running_corrects = 0

    for data in dataloders['train']:
        idx += 1
        # get the inputs
        inputs, gender_labels, age_labels, img_Tshirt_label, img_jacket_label, img_skirt_label, img_trousers_label = data
        
        # wrap them in Variable
        if args.cuda:
            inputs = Variable(inputs).cuda()
            gender_labels = Variable(gender_labels).cuda()
            age_labels = Variable(age_labels).cuda()
            img_Tshirt_label = Variable(img_Tshirt_label).cuda()
            img_jacket_label = Variable(img_jacket_label).cuda()
            img_skirt_label = Variable(img_skirt_label).cuda()
            img_trousers_label = Variable(img_trousers_label).cuda()

        else:
            inputs, gender_labels, age_labels, img_Tshirt_label, img_jacket_label, img_skirt_label, img_trousers_label = Variable(inputs), Variable(gender_labels), Variable(age_labels), Variable(img_Tshirt_label), Variable(img_jacket_label), Variable(img_skirt_label),Variable(img_trousers_label)

        
        # zero the parameter gradients
        optimizer.zero_grad()
        
        # hook the gradient with register_backward_hook
        # model.register_backward_hook(model.my_hook)

        
        gender_outputs, age_outputs, Tshirt_outputs, jacket_outputs, skirt_output, trousers_outputs = model(inputs)  
        
      

        _, g_index = torch.max(gender_outputs.data, 1)
        _, a_index = torch.max(age_outputs.data, 1)
        _, Tshirt_index = torch.max(Tshirt_outputs, 1)
        _, jacket_index = torch.max(jacket_outputs, 1)
        _, skirt_index = torch.max(skirt_output, 1)
        _, trousers_index = torch.max(trousers_outputs, 1)



        
        gender_loss, age_loss, Tshirt_loss, jacket_loss, skirt_loss, trousers_loss = criterion(gender_outputs, gender_labels, age_outputs, age_labels,Tshirt_outputs,img_Tshirt_label,jacket_outputs,img_jacket_label,skirt_output,img_skirt_label,trousers_outputs,img_trousers_label )

        loss = gender_loss + age_loss + Tshirt_loss + jacket_loss + skirt_loss + trousers_loss
        
        # backward + optimize only if in training phase
        loss.backward()
        
        # print the gradient
        # for param in model.parameters():
        #     print('{}:grad->{}'.format(param, param.grad))
        
        optimizer.step()
        
        # statistics
        gender_running_loss += gender_loss.data
        age_running_loss += age_loss.data
        Tshirt_running_loss += Tshirt_loss.data
        jacket_running_loss += jacket_loss.data
        skirt_running_loss += skirt_loss.data
        trousers_running_loss += trousers_loss.data

        train_loss += loss.data
        
        age_running_corrects += torch.sum(a_index == age_labels.data)
        gender_running_corrects += torch.sum(g_index == gender_labels.data)
        Tshirt_running_corrects += torch.sum(Tshirt_index == img_Tshirt_label.data)
        jacket_running_corrects += torch.sum(jacket_index == img_jacket_label.data)
        skirt_running_corrects += torch.sum(skirt_index == img_skirt_label.data)
        trousers_running_corrects += torch.sum(trousers_index == img_trousers_label.data)


        running_corrects += age_running_corrects + gender_running_corrects + Tshirt_running_corrects + jacket_running_corrects + skirt_running_corrects + trousers_running_corrects
        
        # print result every 100 batch
        if idx % 50 == 0:
            gender_batch_loss = gender_running_loss / (args.batch_size * idx)
            age_batch_loss = age_running_loss / (args.batch_size * idx)
            Tshirt_batch_loss = Tshirt_running_loss / (args.batch_size * idx)
            jacket_batch_loss = jacket_running_loss / (args.batch_size * idx)
            skirt_batch_loss  = skirt_running_loss / (args.batch_size * idx)
            trousers_batch_loss = trousers_running_loss /(args.batch_size * idx)
            
            gender_batch_acc = gender_running_corrects.float() / (args.batch_size * idx)
            age_batch_acc = age_running_corrects.float() / (args.batch_size * idx)
            Tshirt_batch_acc = Tshirt_running_corrects.float() / (args.batch_size * idx)
            jacket_batch_acc = jacket_running_corrects.float() / (args.batch_size * idx)
            skirt_batch_acc = skirt_running_corrects.float() /(args.batch_size * idx)
            trousers_batch_acc = trousers_running_corrects.float() / (args.batch_size * idx)


            print(
                '{} Epoch [{}] Batch [{}] genderLoss: {:.4f} ageLoss: {:.4f} TshirtLoss: {:.4f} jacketLoss: {:.4f} skirtLoss: {:.4f} trousersLoss: {:.4f} . \
                                             genderAcc: {:.4f} ageAcc: {:.4f} TshirtAcc: {:.4f} jacketAcc: {:.4f} skirtAcc: {:.4f} trousersAcc: {:.4f} Time: {:.4f}s'. \
                    format('train', epoch, idx, gender_batch_loss, age_batch_loss, Tshirt_batch_loss,  jacket_batch_loss, skirt_batch_loss,trousers_batch_loss,
                                                gender_batch_acc, age_batch_acc,Tshirt_batch_acc, jacket_batch_acc, skirt_batch_acc, trousers_batch_acc,time.time() - begin_time))
            begin_time = time.time()
            
    
    gender_epoch_loss = gender_running_loss / dataset_sizes['train']
    age_epoch_loss = age_running_loss / dataset_sizes['train']
    Tshirt_epoch_loss = Tshirt_running_loss / dataset_sizes['train']
    jacket_epoch_loss = jacket_running_loss / dataset_sizes['train']
    skirt_epoch_loss = skirt_running_loss / dataset_sizes['train']
    trousers_epoch_loss = trousers_running_loss / dataset_sizes['train']
    
    gender_epoch_acc = gender_running_corrects.float() / dataset_sizes['train']
    age_epoch_acc = age_running_corrects.float() / dataset_sizes['train']
    Tshirt_epoch_acc = Tshirt_running_corrects.float() / dataset_sizes['train']
    jacket_epoch_acc = jacket_running_corrects.float() / dataset_sizes['train']
    skirt_epoch_acc = skirt_running_corrects.float() / dataset_sizes['train']
    trousers_epoch_acc = trousers_running_corrects.float() / dataset_sizes['train']

    print(
        'genderLoss: {:.4f} ageLoss: {:.4f} TshirtLoss: {:.4f} jacketLoss: {:.4f} skirtLoss: {:.4f} trousersLoss: {:.4f} . \
                                     genderAcc: {:.4f} ageAcc: {:.4f} TshirtAcc: {:.4f} jacketAcc: {:.4f} skirtAcc: {:.4f} trousersAcc: {:.4f} Time: {:.4f}s'. \
            format(gender_epoch_loss, age_epoch_loss, Tshirt_epoch_loss,  jacket_epoch_loss, skirt_epoch_loss,trousers_epoch_loss,
                                        gender_epoch_acc, age_epoch_acc,Tshirt_epoch_acc, jacket_epoch_acc, skirt_epoch_acc, trousers_epoch_acc,time.time() - begin_time))
    begin_time = time.time()
            
    with open('loss_acc_pic/epoch/PETA_RAP_ResNet50_train_loss_acc.txt', 'a') as f:
        f.write(str(gender_epoch_loss.cpu().numpy()) + " " + str(age_epoch_loss.cpu().numpy()) + " " +
                str(Tshirt_epoch_loss.cpu().numpy()) + " " + str(jacket_epoch_loss.cpu().numpy()) + " " +
                str(skirt_epoch_loss.cpu().numpy()) + " " + str(trousers_epoch_loss.cpu().numpy()) + " " +

                str(gender_epoch_acc.cpu().numpy()) + " " + str(age_epoch_acc.cpu().numpy())+ " " +
                str(Tshirt_epoch_acc.cpu().numpy()) + " " + str(jacket_epoch_acc.cpu().numpy())+ " " +
                str(skirt_epoch_acc.cpu().numpy()) + " " + str(trousers_epoch_acc.cpu().numpy()))
        f.write('\n')
    f.close()
    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


def test(epoch):
    since = time.time()
    begin_time = time.time()
    idx = 0
    print('\nTest')
    model.eval()
    
    train_loss = 0
    gender_running_loss = 0
    age_running_loss = 0
    Tshirt_running_loss = 0
    jacket_running_loss = 0
    skirt_running_loss = 0
    trousers_running_loss = 0

    
    gender_running_corrects = 0
    age_running_corrects = 0
    Tshirt_running_corrects = 0
    jacket_running_corrects = 0
    skirt_running_corrects = 0 
    trousers_running_corrects = 0

    running_corrects = 0
    for data in dataloders['val']:
        idx += 1
        
        # get the inputs
        inputs, gender_labels, age_labels, img_Tshirt_label, img_jacket_label, img_skirt_label, img_trousers_label = data
        
        # wrap them in Variable
        if args.cuda:
            inputs = Variable(inputs).cuda()
            gender_labels = Variable(gender_labels).cuda()
            age_labels = Variable(age_labels).cuda()
            img_Tshirt_label = Variable(img_Tshirt_label).cuda()
            img_jacket_label = Variable(img_jacket_label).cuda()
            img_skirt_label = Variable(img_skirt_label).cuda()
            img_trousers_label = Variable(img_trousers_label).cuda()

        else:
            inputs, gender_labels, age_labels, img_Tshirt_label, img_jacket_label, img_skirt_label, img_trousers_label = Variable(inputs), Variable(gender_labels), Variable(age_labels), Variable(img_Tshirt_label), Variable(img_jacket_label), Variable(img_skirt_label),Variable(img_trousers_label)

        
       
        gender_outputs, age_outputs, Tshirt_outputs, jacket_outputs, skirt_output, trousers_outputs = model(inputs) 
        
        _, g_index = torch.max(gender_outputs.data, 1)
        _, a_index = torch.max(age_outputs.data, 1)
        _, Tshirt_index = torch.max(Tshirt_outputs, 1)
        _, jacket_index = torch.max(jacket_outputs, 1)
        _, skirt_index = torch.max(skirt_output, 1)
        _, trousers_index = torch.max(trousers_outputs, 1)

        gender_loss, age_loss, Tshirt_loss, jacket_loss, skirt_loss, trousers_loss = criterion(gender_outputs, gender_labels, age_outputs, age_labels,Tshirt_outputs,img_Tshirt_label,jacket_outputs,img_jacket_label,skirt_output,img_skirt_label,trousers_outputs,img_trousers_label )

        loss = gender_loss + age_loss + Tshirt_loss + jacket_loss + skirt_loss + trousers_loss
        
        # statistics
        gender_running_loss += gender_loss.data
        age_running_loss += age_loss.data
        Tshirt_running_loss += Tshirt_loss.data
        jacket_running_loss += jacket_loss.data
        skirt_running_loss += skirt_loss.data
        trousers_running_loss += trousers_loss.data

        train_loss += loss.data
        
        age_running_corrects += torch.sum(a_index == age_labels.data)
        gender_running_corrects += torch.sum(g_index == gender_labels.data)
        Tshirt_running_corrects += torch.sum(Tshirt_index == img_Tshirt_label.data)
        jacket_running_corrects += torch.sum(jacket_index == img_jacket_label.data)
        skirt_running_corrects += torch.sum(skirt_index == img_skirt_label.data)
        trousers_running_corrects += torch.sum(trousers_index == img_trousers_label.data)


        running_corrects += age_running_corrects + gender_running_corrects + Tshirt_running_corrects + jacket_running_corrects + skirt_running_corrects + trousers_running_corrects
        
        
        
        # print result every 100 batch
        if idx % 50 == 0:
            gender_batch_loss = gender_running_loss / (args.batch_size * idx)
            age_batch_loss = age_running_loss / (args.batch_size * idx)
            Tshirt_batch_loss = Tshirt_running_loss / (args.batch_size * idx)
            jacket_batch_loss = jacket_running_loss / (args.batch_size * idx)
            skirt_batch_loss  = skirt_running_loss / (args.batch_size * idx)
            trousers_batch_loss = trousers_running_loss /(args.batch_size * idx)
            
            gender_batch_acc = gender_running_corrects.float() / (args.batch_size * idx)
            age_batch_acc = age_running_corrects.float() / (args.batch_size * idx)
            Tshirt_batch_acc = Tshirt_running_corrects.float() / (args.batch_size * idx)
            jacket_batch_acc = jacket_running_corrects.float() / (args.batch_size * idx)
            skirt_batch_acc = skirt_running_corrects.float() /(args.batch_size * idx)
            trousers_batch_acc = trousers_running_corrects.float() / (args.batch_size * idx)
            print(
                '{} Epoch [{}] Batch [{}] genderLoss: {:.4f} ageLoss: {:.4f} TshirtLoss: {:.4f} jacketLoss: {:.4f} skirtLoss: {:.4f} trousersLoss: {:.4f} . \
                                             genderAcc: {:.4f} ageAcc: {:.4f} TshirtAcc: {:.4f} jacketAcc: {:.4f} skirtAcc: {:.4f} trousersAcc: {:.4f} Time: {:.4f}s'. \
                    format('test', epoch, idx, gender_batch_loss, age_batch_loss, Tshirt_batch_loss,  jacket_batch_loss, skirt_batch_loss,trousers_batch_loss,
                                                gender_batch_acc, age_batch_acc,Tshirt_batch_acc, jacket_batch_acc, skirt_batch_acc, trousers_batch_acc,time.time() - begin_time))
            begin_time = time.time()
    
    gender_epoch_loss = gender_running_loss / dataset_sizes['val']
    age_epoch_loss = age_running_loss / dataset_sizes['val']
    Tshirt_epoch_loss = Tshirt_running_loss / dataset_sizes['val']
    jacket_epoch_loss = jacket_running_loss / dataset_sizes['val']
    skirt_epoch_loss = skirt_running_loss / dataset_sizes['val']
    trousers_epoch_loss = trousers_running_loss / dataset_sizes['val']
    
    gender_epoch_acc = gender_running_corrects.float() / dataset_sizes['val']
    age_epoch_acc = age_running_corrects.float() / dataset_sizes['val']
    Tshirt_epoch_acc = Tshirt_running_corrects.float() / dataset_sizes['val']
    jacket_epoch_acc = jacket_running_corrects.float() / dataset_sizes['val']
    skirt_epoch_acc = skirt_running_corrects.float() / dataset_sizes['val']
    trousers_epoch_acc = trousers_running_corrects.float() / dataset_sizes['val']

    print(
        'genderLoss: {:.4f} ageLoss: {:.4f} TshirtLoss: {:.4f} jacketLoss: {:.4f} skirtLoss: {:.4f} trousersLoss: {:.4f} . \
                                     genderAcc: {:.4f} ageAcc: {:.4f} TshirtAcc: {:.4f} jacketAcc: {:.4f} skirtAcc: {:.4f} trousersAcc: {:.4f} Time: {:.4f}s'. \
            format(gender_epoch_loss, age_epoch_loss, Tshirt_epoch_loss,  jacket_epoch_loss, skirt_epoch_loss,trousers_epoch_loss,
                                        gender_epoch_acc, age_epoch_acc,Tshirt_epoch_acc, jacket_epoch_acc, skirt_epoch_acc, trousers_epoch_acc,time.time() - begin_time))
    begin_time = time.time()
            
    with open('loss_acc_pic/epoch/PETA_RAP_ResNet50_test_loss_acc.txt', 'a') as f:
        f.write(str(gender_epoch_loss.cpu().numpy()) + " " + str(age_epoch_loss.cpu().numpy()) + " " +
                str(Tshirt_epoch_loss.cpu().numpy()) + " " + str(jacket_epoch_loss.cpu().numpy()) + " " +
                str(skirt_epoch_loss.cpu().numpy()) + " " + str(trousers_epoch_loss.cpu().numpy()) + " " +

                str(gender_epoch_acc.cpu().numpy()) + " " + str(age_epoch_acc.cpu().numpy())+ " " +
                str(Tshirt_epoch_acc.cpu().numpy()) + " " + str(jacket_epoch_acc.cpu().numpy())+ " " +
                str(skirt_epoch_acc.cpu().numpy()) + " " + str(trousers_epoch_acc.cpu().numpy()))
        f.write('\n')
    f.close()
    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    global gender_best_acc, age_best_acc
    
    if gender_epoch_acc > gender_best_acc:
        gender_best_acc = gender_epoch_acc
        torch.save(model.state_dict(), 'pre_train/' + 'PETA_RAP_ResNet50_params' + '.pkl')
    if age_epoch_acc > age_best_acc:
        age_best_acc = age_epoch_acc




gender_best_acc = 0
age_best_acc = 0
if __name__ == "__main__":
    
    try:
        print("Strat training")
        
        for epoch in range(1, args.epochs + 1):
            model_scheduler.step()
            train(epoch)
            test(epoch)
    except KeyboardInterrupt:
        torch.save(model.state_dict(), 'pre_train/' + 'PETA_RAP_ResNet50_params' + '.pkl')
        
    print('Best TEST genderAcc: {:4f} ageAcc: {:4f}'.format(gender_best_acc, age_best_acc))
