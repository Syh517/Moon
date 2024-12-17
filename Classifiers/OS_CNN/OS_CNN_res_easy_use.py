import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from .OS_CNN_Structure_build import generate_layer_parameter_list
from .log_manager import eval_condition, eval_model, save_to_log
from .OS_CNN_res import OS_CNN_res as OS_CNN

import math

class OS_CNN_easy_use():
    
    def __init__(self,
                 device, 
                 start_kernel_size = 1,
                 Max_kernel_size = 89, 
                 paramenter_number_of_layer_list = [8*128, 5*128*256 + 2*256*128], 
                 max_epoch = 2000, 
                 batch_size=16,
                 print_result_every_x_epoch = 5,
                 n_OS_layer = 3,
                 lr = None
                ):
        
        super(OS_CNN_easy_use, self).__init__()
        
        
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
        
        
    def fit(self, X_train, y_train, X_val, y_val):

        print('code is running on ',self.device)

        # covert numpy to pytorch tensor and put into gpu
        X_train = torch.from_numpy(X_train)
        X_train.requires_grad = False
        X_train = X_train.to(self.device)
        y_train = torch.from_numpy(y_train).to(self.device)
        
        
        X_test = torch.from_numpy(X_val)
        X_test.requires_grad = False
        X_test = X_test.to(self.device)
        y_test = torch.from_numpy(y_val).to(self.device)
        
        
        # add channel dimension to time series data
        if len(X_train.shape) == 2:
            X_train = X_train.unsqueeze_(1)
            X_test = X_test.unsqueeze_(1)

        n_class = max(y_train) + 1
        receptive_field_shape= min(int(X_train.shape[-1]/4),self.Max_kernel_size)
        if receptive_field_shape==0:
            receptive_field_shape =self.start_kernel_size+1
        
        # generate parameter list
        layer_parameter_list = generate_layer_parameter_list(self.start_kernel_size,
                                                             receptive_field_shape,
                                                             self.paramenter_number_of_layer_list,
                                                             in_channel = int(X_train.shape[1]))
        
        
        torch_OS_CNN = OS_CNN(layer_parameter_list, n_class.item(), self.n_OS_layer,False).to(self.device)
        self.model=torch_OS_CNN

        
        
        # loss, optimizer, scheduler
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(torch_OS_CNN.parameters(),lr=self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=50, min_lr=0.0001)
        
        # build dataloader
        
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=max(int(min(X_train.shape[0] / 10, self.batch_size)),2), shuffle=True)
        test_dataset = TensorDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=max(int(min(X_train.shape[0] / 10, self.batch_size)),2), shuffle=False)
        
        
        torch_OS_CNN.train()   
        
        for i in range(self.max_epoch):
            print("epoch:" + str(i))
            for sample in train_loader:
                optimizer.zero_grad()
                y_predict = torch_OS_CNN(sample[0])
                output = criterion(y_predict, sample[1])
                output.backward()
                optimizer.step()
            scheduler.step(output)
            
            if eval_condition(i,self.print_result_every_x_epoch):
                for param_group in optimizer.param_groups:
                    print('epoch =',i, 'lr = ', param_group['lr'])
                torch_OS_CNN.eval()
                metric_train = eval_model(torch_OS_CNN, train_loader)
                metric_test = eval_model(torch_OS_CNN, test_loader)
                torch_OS_CNN.train()

                print('train_Precision:', metric_train['Precision'], '\t train_Recall:', metric_train['Recall'],
                      '\t train_F1:', metric_train['F1'], '\t train_Accuracy:', metric_train['Accuracy'], '\t train_F1pa:', metric_train['F1pa'],
                       '\t train_PA_K_F1:', metric_train['PA_K_F1'], '\t train_AU_PR:', metric_train['AU_PR'])
                print('test_Precision:', metric_test['Precision'], '\t test_Recall:', metric_test['Recall'],
                      '\t test_F1:', metric_test['F1'], '\t test_Accuracy:', metric_test['Accuracy'], '\t test_F1pa:', metric_test['F1pa'],
                       '\t test_PA_K_F1:', metric_test['PA_K_F1'], '\t test_AU_PR:', metric_test['AU_PR'])
                print('loss:', output.item())

        self.OS_CNN = torch_OS_CNN

        
    def predict(self, X_test):
        
        X_test = torch.from_numpy(X_test)
        X_test.requires_grad = False
        X_test = X_test.to(self.device)
        
        if len(X_test.shape) == 2:
            X_test = X_test.unsqueeze_(1)
        
        test_dataset = TensorDataset(X_test)
        test_loader = DataLoader(test_dataset, batch_size=max(int(min(X_test.shape[0] / 10, self.batch_size)),2), shuffle=False)
        
        self.OS_CNN.eval()
        
        # predict_list = np.array([])
        predict_list=[]
        for sample in test_loader:
            y_predict = self.OS_CNN(sample[0])
            y_predict = y_predict.detach().cpu().numpy()
            # y_predict = np.argmax(y_predict, axis=1)
            # predict_list = np.concatenate((predict_list, y_predict), axis=0)
            predict_list.append(y_predict)

        y_predict=np.concatenate(predict_list)

        ymin = np.min(y_predict)
        ymax = np.max(y_predict)
        halfrange = math.ceil(max(abs(ymin), abs(ymax)))

        y_prob = y_predict + halfrange
        for yi in y_prob:
            sum = yi[0] + yi[1]
            yi[0] = yi[0] / sum
            yi[1] = yi[1] / sum

        return y_prob


        
        
        
        
        
        
        