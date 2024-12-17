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
from .CNN import CNN_classifier as OS_CNN


class OS_CNN_easy_use():

    def __init__(self, Result_log_folder,
                 dataset_name,
                 device,
                 start_kernel_size=1,
                 Max_kernel_size=33,
                 paramenter_number_of_layer_list=[8 * 128, 5 * 128 * 256 + 2 * 256 * 128],
                 max_epoch=2000,
                 batch_size=16,
                 print_result_every_x_epoch=10,
                 lr=0.001
                 ):

        super(OS_CNN_easy_use, self).__init__()

        if not os.path.exists(Result_log_folder + dataset_name + '/'):
            os.makedirs(Result_log_folder + dataset_name + '/')
        Initial_model_path = Result_log_folder + dataset_name + '/' + dataset_name + 'initial_model'
        model_save_path = Result_log_folder + dataset_name + '/' + dataset_name + 'Best_model'

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
        self.lr = lr
        self.OS_CNN = None

    def fit(self, X_train, y_train, X_val, y_val):

        print('code is running on ', self.device)

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

        input_shape = X_train.shape[-1]
        n_class = max(y_train) + 1
        receptive_field_shape = self.Max_kernel_size  # 感受野固定


        torch_OS_CNN = OS_CNN().to(self.device)
        torch_OS_CNN = torch_OS_CNN.float()

        # save_initial_weight
        torch.save(torch_OS_CNN.state_dict(), self.Initial_model_path)

        # loss, optimizer, scheduler
        criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
        optimizer = optim.Adam(torch_OS_CNN.parameters(), lr=self.lr)  # 确定优化器种类，待优化参数和学习率
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=50, min_lr=0.0001)

        # build dataloader
        train_dataset = TensorDataset(X_train, y_train)  # 对数据进行打包
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size,
                                  shuffle=True)  # shuffle参数指定是否在每个周期开始时随机打乱数据
        test_dataset = TensorDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size,
                                 shuffle=False)

        torch_OS_CNN.train()  # 开启train模式

        for i in range(self.max_epoch):  # 开始训练
            print("epoch:" + str(i))
            for sample in train_loader:  # 遍历每条train时序数据
                optimizer.zero_grad()  # 清空模型参数的梯度，以确保每次迭代的梯度计算都是基于当前小批量数据的，而不会受之前迭代的影响。这是为了避免在优化过程中梯度的不正确累积。
                y_predict = torch_OS_CNN(sample[0])  # 得到模型预测的label
                output = criterion(y_predict, sample[1])  # 交叉熵损失函数值
                output.backward()  # 反向传播并输出梯度值
                optimizer.step()  # 在每个mini-batch中更新模型
            scheduler.step(output)  # 在每个epoch中优化学习率

            if eval_condition(i, self.print_result_every_x_epoch):  # 每50步查看一下模型准确率
                for param_group in optimizer.param_groups:
                    print('epoch =', i, 'lr = ', param_group['lr'])
                torch_OS_CNN.eval()  # 开启eval模式。在eval模式下，会固定网络层参数值，使得输入数据不会影响模型
                metric_train = eval_model(torch_OS_CNN, train_loader)
                metric_test = eval_model(torch_OS_CNN, test_loader)
                torch_OS_CNN.train()  # 继续开启train模式

                print('train_Precision:', metric_train['Precision'], '\t train_Recall:', metric_train['Recall'],
                      '\t train_F1:', metric_train['F1'], '\t train_Accuracy:', metric_train['Accuracy'])
                print('test_Precision:', metric_test['Precision'], '\t test_Recall:', metric_test['Recall'],
                      '\t test_F1:', metric_test['F1'], '\t test_Accuracy:', metric_test['Accuracy'])
                print('loss:', output.item())
                sentence = 'train_F1=\t' + str(metric_train['F1']) + '\t test_F1=\t' + str(metric_test['F1'])

                print('log saved at:')
                save_to_log(sentence, self.Result_log_folder, self.dataset_name)  # 记入文件
                torch.save(torch_OS_CNN.state_dict(), self.model_save_path)  # 保存模型

        torch.save(torch_OS_CNN.state_dict(), self.model_save_path)  # 保存模型
        self.OS_CNN = torch_OS_CNN

    def predict(self, X_test):

        X_test = torch.from_numpy(X_test)  # 将NumPy数组转换为PyTorch张量
        X_test.requires_grad = False
        X_test = X_test.to(self.device)

        if len(X_test.shape) == 2:
            X_test = X_test.unsqueeze_(1)

        test_dataset = TensorDataset(X_test)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        self.OS_CNN.eval()

        predict_list = np.array([])
        for sample in test_loader:
            y_predict = self.OS_CNN(sample[0])
            y_predict = y_predict.detach().cpu().numpy()  # 阻断反传、移至cpu 返回值是cpu上的Tensor、返回值为numpy.array()
            y_predict = np.argmax(y_predict, axis=1)  # 找每行向量里面最大值的索引，怀疑原来的y_predict是概率
            predict_list = np.concatenate((predict_list, y_predict), axis=0)

        return predict_list







