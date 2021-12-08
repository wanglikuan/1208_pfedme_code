import torch
import os

from FLAlgorithms.users.usernewperavg import newUserAVG
from FLAlgorithms.servers.serverbase import Server
from utils.model_utils import read_data, read_user_data
import numpy as np

# Implementation for FedAvg Server

class newPerAvg(Server):
    def __init__(self, device, dataset,algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters,
                 local_epochs, optimizer, num_users, times):
        super().__init__(device, dataset,algorithm, model[0], batch_size, learning_rate, beta, lamda, num_glob_iters,
                         local_epochs, optimizer, num_users, times)

        # Initialize data for all  users
        data = read_data(dataset)
        #total_users = len(data[0])
        total_users = 20
        for i in range(total_users):
            id, train , test = read_user_data(i, data, dataset)
            user = newUserAVG(device, id, train, test, model, batch_size, learning_rate,beta,lamda, local_epochs, optimizer)
            self.users.append(user)
            self.total_train_samples += user.train_samples
            
        print("Number of users / total users:",num_users, " / " ,total_users)
        print("Finished creating FedAvg server.")

    def send_grads(self):
        assert (self.users is not None and len(self.users) > 0)
        grads = []
        for param in self.model.parameters():
            if param.grad is None:
                grads.append(torch.zeros_like(param.data))
            else:
                grads.append(param.grad)
        for user in self.users:
            user.set_grads(grads)

    def train(self):
        loss = []
        for glob_iter in range(self.num_glob_iters):
            print("-------------Round number: ",glob_iter, " -------------")
            #loss_ = 0
            self.send_parameters()

            self.selected_users = self.select_users(glob_iter,self.num_users)
            for user in self.selected_users:
                user.fine_tuning_train(2) #fine_tuning_epochs = 2
            # Evaluate model each interation
            self.evaluate()

            for user in self.selected_users:
                user.train(self.local_epochs) #* user.train_samples
            self.aggregate_parameters()
            #loss_ /= self.total_train_samples
            #loss.append(loss_)
            #print(loss_)
        #print(loss)
        self.save_results()
        self.save_model()
