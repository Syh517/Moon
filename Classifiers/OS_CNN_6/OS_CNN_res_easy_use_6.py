import os
from sklearn.metrics import accuracy_score
from os.path import dirname
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from .OS_CNN_Structure_build import generate_layer_parameter_list
from .log_manager import eval_condition, eval_model_2, save_to_log
from .FCN_Kernel_size import FCN

import math

class OS_CNN_easy_use():
    
    def __init__(self,Result_log_folder, 
                 dataset_name, 
                 device, 
                 start_kernel_size = 1,
                 Max_kernel_size = 89, 
                 paramenter_number_of_layer_list = [8*128, 5*128*256 + 2*256*128], 
                 max_epoch=2000,
                 batch_size=16,
                 print_result_every_x_epoch = 5,
                 n_OS_layer = 3,
                 lr = None
                ):
        
        super(OS_CNN_easy_use, self).__init__()
        
        if not os.path.exists(Result_log_folder +dataset_name+'/'):
            os.makedirs(Result_log_folder +dataset_name+'/')
        Initial_model_path = Result_log_folder +dataset_name+'/'+dataset_name+'initial_model'
        model_save_path = Result_log_folder +dataset_name+'/'+dataset_name+'Best_model'
        

        self.Result_log_folder = Result_log_folder
        self.dataset_name = dataset_name        
        self.model_save_path = model_save_path
        self.Initial_model_path = Initial_model_path
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        self.start_kernel_size = start_kernel_size
        self.Max_kernel_size = Max_kernel_size
        self.paramenter_number_of_layer_list = paramenter_number_of_layer_list
        self.max_epoch = max_epoch
        self.batch_size = batch_size
        self.print_result_every_x_epoch = print_result_every_x_epoch
        self.n_OS_layer = n_OS_layer
        
        if lr == None:
            self.lr = 0.001
        else:
            self.lr = lr
        self.OS_CNN = None



    def fit(self, X1_train, X2_train, y_train, X1_val, X2_val, y_val):

        print('code is running on ',self.device)

        for i in range(len(y_train)):
            if y_train[i] == 0:
                self.X1 = X1_train[i]
                self.X2 = X2_train[i]
                break

        self.X1 = np.array(self.X1).reshape(1, X1_train.shape[1])
        self.X2 = np.array(self.X2).reshape(1, X2_train.shape[1])

        # covert numpy to pytorch tensor and put into gpu
        X1_train = torch.from_numpy(X1_train)
        X1_train.requires_grad = False
        X1_train = X1_train.to(self.device)

        X2_train = torch.from_numpy(X2_train)
        X2_train.requires_grad = False
        X2_train = X2_train.to(self.device)

        y_train = torch.from_numpy(y_train).to(self.device)

        X1_test = torch.from_numpy(X1_val)
        X1_test.requires_grad = False
        X1_test = X1_test.to(self.device)

        X2_test = torch.from_numpy(X2_val)
        X2_test.requires_grad = False
        X2_test = X2_test.to(self.device)

        y_test = torch.from_numpy(y_val).to(self.device)

        # add channel dimension to time series data
        if len(X1_train.shape) == 2:
            X1_train = X1_train.unsqueeze_(1)
            X1_test = X1_test.unsqueeze_(1)

        if len(X2_train.shape) == 2:
            X2_train = X2_train.unsqueeze_(1)
            X2_test = X2_test.unsqueeze_(1)

        n_class = max(y_train) + 1


        torch_OS_CNN = FCN(int(X1_train.shape[1]), n_class.item(), int(X1_train.shape[-1])).to(self.device)

        # save_initial_weight
        torch.save(torch_OS_CNN.state_dict(), self.Initial_model_path)

        # loss, optimizer, scheduler
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(torch_OS_CNN.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=50, min_lr=0.0001)

        # build dataloader

        train_dataset = TensorDataset(X1_train, X2_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=max(int(min(X1_train.shape[0] / 10, self.batch_size)), 2),
                                  shuffle=True)
        test_dataset = TensorDataset(X1_test, X2_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=max(int(min(X2_train.shape[0] / 10, self.batch_size)), 2),
                                 shuffle=False)

        torch_OS_CNN.train()

        for i in range(self.max_epoch):
            print("epoch:" + str(i))
            for sample in train_loader:
                optimizer.zero_grad()
                y_predict = torch_OS_CNN(sample[0], sample[1])
                output = criterion(y_predict, sample[2])
                output.backward()
                optimizer.step()
            scheduler.step(output)

            if eval_condition(i, self.print_result_every_x_epoch):
                for param_group in optimizer.param_groups:
                    print('epoch =', i, 'lr = ', param_group['lr'])
                torch_OS_CNN.eval()
                metric_train = eval_model_2(torch_OS_CNN, train_loader)
                metric_test = eval_model_2(torch_OS_CNN, test_loader)
                torch_OS_CNN.train()

                print('train_Precision:', metric_train['Precision'], '\t train_Recall:', metric_train['Recall'],
                      '\t train_F1:', metric_train['F1'], '\t train_AUC:', metric_train['AUC'])
                print('test_Precision:', metric_test['Precision'], '\t test_Recall:', metric_test['Recall'],
                      '\t test_F1:', metric_test['F1'], '\t test_AUC:', metric_test['AUC'])
                print('loss:', output.item())
                sentence = 'train_F1=\t' + str(metric_train['F1']) + '\t test_F1=\t' + str(metric_test['F1'])

                # print('log saved at:')
                # save_to_log(sentence, self.Result_log_folder, self.dataset_name)
                # torch.save(torch_OS_CNN.state_dict(), self.model_save_path)

        # torch.save(torch_OS_CNN.state_dict(), self.model_save_path)
        self.OS_CNN = torch_OS_CNN

        
        
    def predict(self, X1_test, X2_test):

        X1_test = torch.from_numpy(X1_test)  # 将NumPy数组转换为PyTorch张量
        X1_test.requires_grad = False
        X1_test = X1_test.to(self.device)

        X2_test = torch.from_numpy(X2_test)  # 将NumPy数组转换为PyTorch张量
        X2_test.requires_grad = False
        X2_test = X2_test.to(self.device)

        if len(X1_test.shape) == 2:
            X1_test = X1_test.unsqueeze_(1)

        if len(X2_test.shape) == 2:
            X2_test = X2_test.unsqueeze_(1)
        
        test_dataset = TensorDataset(X1_test, X2_test)
        test_loader = DataLoader(test_dataset, batch_size=max(int(min(X1_test.shape[0] / 10, self.batch_size)),2), shuffle=False)
        
        self.OS_CNN.eval()
        
        predict_list = np.array([])
        for sample in test_loader:
            y_predict = self.OS_CNN(sample[0], sample[1])
            y_predict = y_predict.detach().cpu().numpy()
            y_predict = np.argmax(y_predict, axis=1)
            predict_list = np.concatenate((predict_list, y_predict), axis=0)

        return predict_list

    def predict_ts(self, X1):
        n = X1.shape[0]
        X2 = np.tile(self.X2, (n, 1))

        X1_test = torch.from_numpy(X1)  # 将NumPy数组转换为PyTorch张量
        X1_test.requires_grad = False
        X1_test = X1_test.to(self.device)

        X2_test = torch.from_numpy(X2)  # 将NumPy数组转换为PyTorch张量
        X2_test.requires_grad = False
        X2_test = X2_test.to(self.device)

        if len(X1_test.shape) == 2:
            X1_test = X1_test.unsqueeze_(1)

        if len(X2_test.shape) == 2:
            X2_test = X2_test.unsqueeze_(1)

        test_dataset = TensorDataset(X1_test, X2_test)
        test_loader = DataLoader(test_dataset, batch_size=max(int(min(X1_test.shape[0] / 10, self.batch_size)), 2),
                                 shuffle=False)

        self.OS_CNN.eval()

        predict_list = []
        for sample in test_loader:
            y_predict = self.OS_CNN(sample[0], sample[1])
            y_predict = y_predict.detach().cpu().numpy()
            predict_list.append(y_predict)

        y_predict = np.concatenate(predict_list)

        ymin = np.min(y_predict)
        ymax = np.max(y_predict)
        halfrange = math.ceil(max(abs(ymin), abs(ymax)))

        y_prob = y_predict + halfrange
        for yi in y_prob:
            sum = yi[0] + yi[1]
            yi[0] = yi[0] / sum
            yi[1] = yi[1] / sum

        return y_prob

    def predict_img(self, X2):
        n = X2.shape[0]
        X1 = np.tile(self.X1, (n, 1))

        X1_test = torch.from_numpy(X1)  # 将NumPy数组转换为PyTorch张量
        X1_test.requires_grad = False
        X1_test = X1_test.to(self.device)

        X2_test = torch.from_numpy(X2)  # 将NumPy数组转换为PyTorch张量
        X2_test.requires_grad = False
        X2_test = X2_test.to(self.device)

        if len(X1_test.shape) == 2:
            X1_test = X1_test.unsqueeze_(1)

        if len(X2_test.shape) == 2:
            X2_test = X2_test.unsqueeze_(1)

        test_dataset = TensorDataset(X1_test, X2_test)
        test_loader = DataLoader(test_dataset, batch_size=max(int(min(X2_test.shape[0] / 10, self.batch_size)), 2),
                                 shuffle=False)

        self.OS_CNN.eval()

        predict_list = []
        for sample in test_loader:
            y_predict = self.OS_CNN(sample[0], sample[1])
            y_predict = y_predict.detach().cpu().numpy()
            predict_list.append(y_predict)

        y_predict = np.concatenate(predict_list)

        ymin = np.min(y_predict)
        ymax = np.max(y_predict)
        halfrange = math.ceil(max(abs(ymin), abs(ymax)))

        y_prob = y_predict + halfrange
        for yi in y_prob:
            sum = yi[0] + yi[1]
            yi[0] = yi[0] / sum
            yi[1] = yi[1] / sum

        return y_prob




        
        
        
        
        
        
        
        