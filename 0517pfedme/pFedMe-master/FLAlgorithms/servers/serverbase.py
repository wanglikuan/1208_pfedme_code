import torch, copy
from torch import nn
import os
import numpy as np
import h5py
from utils.model_utils import Metrics
from FLAlgorithms.trainmodel.models import *
import copy

class Server:
    def __init__(self, device, dataset,algorithm, model, batch_size, learning_rate ,beta, lamda,
                 num_glob_iters, local_epochs, optimizer,num_users, times):

        # Set up the main attributes
        self.device = device
        self.dataset = dataset
        self.num_glob_iters = num_glob_iters
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.total_train_samples = 0
        self.model = copy.deepcopy(model)
        self.users = []
        self.selected_users = []
        self.model_list = []
        self.num_users = num_users
        self.beta = beta
        self.lamda = lamda
        self.algorithm = algorithm
        self.rs_train_acc, self.rs_train_loss, self.rs_glob_acc,self.rs_train_acc_per, self.rs_train_loss_per, self.rs_glob_acc_per = [], [], [], [], [], []
        self.times = times
        #
        # for _ in range(self.num_users):
        #     self.model_list.append(self.model)
        #
        #self.model_list = []
        #for _ in range(self.num_users):
        #    self.model_list.append(DNN().to(torch.device("cuda:0")))
        self.model_list = [copy.deepcopy(self.model) for _ in range(self.num_users)]
        self.model_list_teacher = [copy.deepcopy(self.model) for _ in range(self.num_users)] #用于计算weight，存放teacher参数
        #
        self.weight_alpha = torch.ones(self.num_users, self.num_users, requires_grad=True)
        #weight_alpha = weight_alpha.float()/(self.N_parties-1)        
        self.weight_alpha = self.weight_alpha.float()/(self.num_users)
        self.weight_alpha = self.weight_alpha.cuda()  
        #
        self.weight_mean = self.weight_alpha.clone().detach()
        #
        self.np_weight_alpha = self.weight_alpha.cpu().clone().detach().numpy()
        # Initialize the server's grads to zeros
        #for param in self.model.parameters():
        #    param.data = torch.zeros_like(param.data)
        #    param.grad = torch.zeros_like(param.data)
        #self.send_parameters()
    def calculate_param_loss(self, loss_input, loss_target): #输入为.parameters()
        loss_fn = torch.nn.MSELoss(reduce=True, size_average=False)
        loss_tmp = [0.0 for _ in range(10)]
        loss = 0.0
        count_layer = 0
        # for idx, input_param, target_param in enumerate(zip(loss_input, loss_target)):
        #     loss_tmp[idx] = loss_fn(input_param, target_param)
        #     count_layer += 1
        # for idx in range(count_layer):
        #     loss += loss_tmp[idx]
        for input_param, target_param in zip(loss_input, loss_target):
            #input_data = input_param.data.requires_grad_()
            #target_data = target_param.data.requires_grad_()
            loss += loss_fn(input_param.data, target_param.data)
            #print('loss_tmp:',loss_fn(input_param.data, target_param.data))
        return loss #loss_tmp

    def calculate_weight(self):
        loss_fn = torch.nn.MSELoss(reduce=True, size_average=False)
        #拷贝模型参数#
        model_copy = copy.deepcopy(self.model_list)
        #复制模型参数至一个list=>由tensor变量组成#
        model_param_list=[]     ###模型参数列表###对应raw_logits###
        teacher_param_list=[]   ###teacher参数列表###对应teacher_logits###
        layer_num = [] #layer_num[0]为单个model中的层数 从1开始计数
        #初始化模型参数#清零teacher参数
        model_param_list_count = 0

        for idx_user, user in enumerate(self.selected_users):
            for param in user.get_parameters():
                model_param_list.append(param.data.clone().detach())
                model_param_list[model_param_list_count].requires_grad_()
                #print('model_param_list[model_param_list_count]', model_param_list[model_param_list_count])
                ######创建空的teacher参数列表######
                teacher_param_list.append(param.data.clone().detach())
                teacher_param_list[model_param_list_count] = torch.zeros_like(teacher_param_list[model_param_list_count])
                teacher_param_list[model_param_list_count].requires_grad_()
                #################################
                model_param_list_count += 1
            #print('layers_num_in_single_model:',model_param_list_count,'(from 1 to count)')
            layer_num.append(model_param_list_count)                

        # for idx_model, _ in enumerate(model_copy):
        #     for param in model_copy[idx_model].parameters():
        #         model_param_list.append(param.data.clone().detach())
        #         model_param_list[model_param_list_count].requires_grad_()
        #         print('model_param_list[model_param_list_count]', model_param_list[model_param_list_count])
        #         ######创建空的teacher参数列表######
        #         teacher_param_list.append(param.data.clone().detach())
        #         teacher_param_list[model_param_list_count] = torch.zeros_like(teacher_param_list[model_param_list_count])
        #         teacher_param_list[model_param_list_count].requires_grad_()
        #         #################################
        #         model_param_list_count += 1
        #     print('layers_num_in_single_model:',model_param_list_count,'(from 1 to count)')
        #     layer_num.append(model_param_list_count)

        # print('model_param_list[0].requires_grad',model_param_list[0].requires_grad) 
        # model_param_list[0].requires_grad_()
        # print('model_param_list[5].requires_grad',model_param_list[5].requires_grad)  
        # print('teacher_param_list[5].requires_grad',teacher_param_list[5].requires_grad) 
      
        #计算模型参数加权和#
        for teacher_idx in range(self.num_users): #teacher编号
            for layers_idx in range(layer_num[0]): #layers编号
                for raw_idx in range(self.num_users): #raw param编号
                    teacher_param_list[teacher_idx*layer_num[0]+layers_idx] = torch.add(teacher_param_list[teacher_idx*layer_num[0]+layers_idx], \
                        model_param_list[raw_idx*layer_num[0]+layers_idx].clone().detach() * self.weight_alpha[teacher_idx][raw_idx])
                    #print('now:', model_param_list[raw_idx*layer_num[0]+layers_idx])
                    #print('+1:', model_param_list[(raw_idx+1)*layer_num[0]+layers_idx])
                    #print('-----------------------------------------------------')
                    #print('self.weight_alpha[{}][{}]:'.format(teacher_idx, raw_idx),self.weight_alpha[teacher_idx][raw_idx])
            ####计算teacher与student的loss
            self.weight_alpha.retain_grad()
            loss = 0.0
            loss_penalty = 0.0
            for layers_idx_finish in range(layer_num[0]):
                #print('teacher_idx*layer_num[0]+{}:'.format(layers_idx_finish),teacher_idx*layer_num[0]+layers_idx_finish)
                loss_tmp = loss_fn(model_param_list[teacher_idx*layer_num[0]+layers_idx_finish], \
                    teacher_param_list[teacher_idx*layer_num[0]+layers_idx_finish])
                loss += loss_tmp
            for raw_idx in range(self.num_users):
                loss_penalty += loss_fn(self.weight_alpha[teacher_idx], self.weight_mean[teacher_idx])
            print('model:', teacher_idx, self.weight_alpha[teacher_idx], self.weight_mean[teacher_idx])
            print('loss_penalty:', loss_penalty)     
            print('loss:', loss)
            loss += loss_penalty*0.1
            loss.backward(retain_graph=True)

            #更新权重#
            with torch.no_grad():
                #weight[self_idx] = weight[self_idx] - weight[self_idx].grad * 0.001  #更新权重
                #print('weight.grad:', self.weight_alpha.grad) 
                #print('weight.requires_grad:', self.weight_alpha.requires_grad)                
                gradabs = torch.abs(self.weight_alpha.grad)
                gradsum = torch.sum(gradabs)
                gradavg = gradsum.item() / (self.num_users)
                grad_lr = 1.0
                for _ in range(10): #0.1
                    if gradavg > 0.01:
                        gradavg = gradavg*1.0/5
                        grad_lr = grad_lr/5                
                    if gradavg < 0.01:
                        gradavg = gradavg*1.0*5
                        grad_lr = grad_lr*5
                print('weight.grad:', self.weight_alpha.grad)        
                print('teacher_idx:',teacher_idx, 'grad_lr:', grad_lr)
                self.weight_alpha.sub_(self.weight_alpha.grad*grad_lr)
                #weight.sub_(weight.grad*50)
                self.weight_alpha.grad.zero_()   
                
        #更新权重np#   
        for self_idx in range(self.num_users): #对某一model，计算其softmax后的weight
            with torch.no_grad():
                self.weight_alpha[self_idx] = nn.functional.softmax(self.weight_alpha[self_idx]*5.0)
        self.np_weight_alpha = self.weight_alpha.cpu().clone().detach().numpy()

    def aggregate_grads(self):
        assert (self.users is not None and len(self.users) > 0)
        for param in self.model.parameters():
            param.grad = torch.zeros_like(param.data)
        for user in self.users:
            self.add_grad(user, user.train_samples / self.total_train_samples)

    def add_grad(self, user, ratio):
        user_grad = user.get_grads()
        for idx, param in enumerate(self.model.parameters()):
            param.grad = param.grad + user_grad[idx].clone() * ratio
            #print('ratio:', ratio)

    def send_parameters(self):
        assert (self.users is not None and len(self.users) > 0)
        for idx, user in enumerate(self.users):
            #user.set_parameters(self.model)            
            user.set_parameters(self.model_list[idx])

    def add_parameters_idx(self, user, ratio, idx_server, idx_user):#令model_list[idx]一次加上一个user的加权参数
        model = self.model.parameters()
        for server_param, user_param in zip(self.model_list[idx_server].parameters(), user.get_parameters()):
            server_param.data = server_param.data + user_param.data.clone() * self.np_weight_alpha[idx_server][idx_user]
        #print('ratio in add_parameters:', self.np_weight_alpha[idx_server][idx_user])

    def add_parameters(self, user, ratio):
        model = self.model.parameters()
        for server_param, user_param in zip(self.model.parameters(), user.get_parameters()):
            server_param.data = server_param.data + user_param.data.clone() * ratio
        #print('ratio in add_parameters:', ratio)

    def aggregate_parameters(self):
        assert (self.users is not None and len(self.users) > 0)
        for param in self.model.parameters():
            param.data = torch.zeros_like(param.data)
        for idx_server_model, _ in enumerate(self.model_list):
            for param in self.model_list[idx_server_model].parameters():
                param.data = torch.zeros_like(param.data)
        total_train = 0
        #if(self.num_users = self.to)
        for user in self.selected_users:
            total_train += user.train_samples
        # for user in self.selected_users:
        #     self.add_parameters(user, user.train_samples / total_train)
        for idx_server, user in enumerate(self.selected_users):
            print('np_weight_alpha[{}]'.format(idx_server), self.np_weight_alpha[idx_server])
            for idx_user, user_give_para in enumerate(self.selected_users):
                self.add_parameters_idx(user_give_para, user_give_para.train_samples / total_train, idx_server, idx_user)

    def save_model(self):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.model, os.path.join(model_path, "server" + ".pt"))

    def load_model(self):
        model_path = os.path.join("models", self.dataset, "server" + ".pt")
        assert (os.path.exists(model_path))
        self.model = torch.load(model_path)

    def model_exists(self):
        return os.path.exists(os.path.join("models", self.dataset, "server" + ".pt"))
    
    def select_users(self, round, num_users):
        '''selects num_clients clients weighted by number of samples from possible_clients
        Args:
            num_clients: number of clients to select; default 20
                note that within function, num_clients is set to
                min(num_clients, len(possible_clients))
        
        Return:
            list of selected clients objects
        '''
        if(num_users == len(self.users)):
            print("All users are selected")
            return self.users

        num_users = min(num_users, len(self.users))
        #np.random.seed(round)
        return np.random.choice(self.users, num_users, replace=False) #, p=pk)

    # define function for persionalized agegatation.
    def persionalized_update_parameters(self,user, ratio):
        # only argegate the local_weight_update
        for server_param, user_param in zip(self.model.parameters(), user.local_weight_updated):
            server_param.data = server_param.data + user_param.data.clone() * ratio


    def persionalized_aggregate_parameters(self):
        assert (self.users is not None and len(self.users) > 0)

        # store previous parameters
        previous_param = copy.deepcopy(list(self.model.parameters()))
        for param in self.model.parameters():
            param.data = torch.zeros_like(param.data)
        total_train = 0
        #if(self.num_users = self.to)
        for user in self.selected_users:
            total_train += user.train_samples

        for user in self.selected_users:
            self.add_parameters(user, user.train_samples / total_train)
            #self.add_parameters(user, 1 / len(self.selected_users))

        # aaggregate avergage model with previous model using parameter beta 
        for pre_param, param in zip(previous_param, self.model.parameters()):
            param.data = (1 - self.beta)*pre_param.data + self.beta*param.data
            
    # Save loss, accurancy to h5 fiel
    def save_results(self):
        alg = self.dataset + "_" + self.algorithm
        alg = alg + "_" + str(self.learning_rate) + "_" + str(self.beta) + "_" + str(self.lamda) + "_" + str(self.num_users) + "u" + "_" + str(self.batch_size) + "b" + "_" + str(self.local_epochs)
        if(self.algorithm == "pFedMe" or self.algorithm == "pFedMe_p"):
            alg = alg + "_" + str(self.K) + "_" + str(self.personal_learning_rate)
        alg = alg + "_" + str(self.times)
        if (len(self.rs_glob_acc) != 0 &  len(self.rs_train_acc) & len(self.rs_train_loss)) :
            with h5py.File("./results/"+'{}.h5'.format(alg, self.local_epochs), 'w') as hf:
                hf.create_dataset('rs_glob_acc', data=self.rs_glob_acc)
                hf.create_dataset('rs_train_acc', data=self.rs_train_acc)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss)
                hf.close()
            #with open("./results/"+'{}.txt'.format(alg), 'a') as f:
                #f.write('{}\t{}\t{}\t{}\n'.format(self.local_epochs, self.rs_glob_acc, self.rs_train_acc, self.rs_train_loss))

        
        # store persionalized value
        alg = self.dataset + "_" + self.algorithm + "_p"
        alg = alg  + "_" + str(self.learning_rate) + "_" + str(self.beta) + "_" + str(self.lamda) + "_" + str(self.num_users) + "u" + "_" + str(self.batch_size) + "b"+ "_" + str(self.local_epochs)
        if(self.algorithm == "pFedMe" or self.algorithm == "pFedMe_p"):
            alg = alg + "_" + str(self.K) + "_" + str(self.personal_learning_rate)
        alg = alg + "_" + str(self.times)
        if (len(self.rs_glob_acc_per) != 0 &  len(self.rs_train_acc_per) & len(self.rs_train_loss_per)) :
            with h5py.File("./results/"+'{}.h5'.format(alg, self.local_epochs), 'w') as hf:
                hf.create_dataset('rs_glob_acc', data=self.rs_glob_acc_per)
                hf.create_dataset('rs_train_acc', data=self.rs_train_acc_per)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss_per)
                hf.close()
            #with open("./results/"+'{}.txt'.format(alg), 'a') as f:
                #f.write('{}\t{}\t{}\t{}\n'.format(self.local_epochs, self.rs_glob_acc_per, self.rs_train_acc_per, self.rs_train_loss_per))


    def test(self):
        '''tests self.latest_model on given clients
        '''
        num_samples = []
        tot_correct = []
        losses = []
        for c in self.users:
            ct, ns = c.test()
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
        ids = [c.id for c in self.users]

        return ids, num_samples, tot_correct

    def train_error_and_loss(self):
        num_samples = []
        tot_correct = []
        losses = []
        for c in self.users:
            ct, cl, ns = c.train_error_and_loss() 
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
            losses.append(cl*1.0)
        
        ids = [c.id for c in self.users]
        #groups = [c.group for c in self.clients]

        return ids, num_samples, tot_correct, losses

    def test_persionalized_model(self):
        '''tests self.latest_model on given clients
        '''
        num_samples = []
        tot_correct = []
        for c in self.users:
            ct, ns = c.test_persionalized_model()
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
        ids = [c.id for c in self.users]

        return ids, num_samples, tot_correct

    def train_error_and_loss_persionalized_model(self):
        num_samples = []
        tot_correct = []
        losses = []
        for c in self.users:
            ct, cl, ns = c.train_error_and_loss_persionalized_model() 
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
            losses.append(cl*1.0)
        
        ids = [c.id for c in self.users]
        #groups = [c.group for c in self.clients]

        return ids, num_samples, tot_correct, losses

    def evaluate(self):
        stats = self.test()  
        stats_train = self.train_error_and_loss()
        glob_acc = np.sum(stats[2])*1.0/np.sum(stats[1])
        train_acc = np.sum(stats_train[2])*1.0/np.sum(stats_train[1])
        # train_loss = np.dot(stats_train[3], stats_train[1])*1.0/np.sum(stats_train[1])
        train_loss = sum([x * y for (x, y) in zip(stats_train[1], stats_train[3])]).item() / np.sum(stats_train[1])
        self.rs_glob_acc.append(glob_acc)
        self.rs_train_acc.append(train_acc)
        self.rs_train_loss.append(train_loss)
        #print("stats_train[1]",stats_train[3][0])
        for x,y in zip(stats[2], stats[1]):
            print("client Accurancy: ", x*1.0/y)
        print("Average Global Accurancy: ", glob_acc)
        print("Average Global Trainning Accurancy: ", train_acc)
        print("Average Global Trainning Loss: ",train_loss)
        with open("./results/"+'{}.txt'.format('emnist_ours'), 'a') as f:
            f.write('{}\t{}\t{}\t{}\n'.format(self.local_epochs, glob_acc, train_acc, train_loss))


    def evaluate_personalized_model(self):
        stats = self.test_persionalized_model()  
        stats_train = self.train_error_and_loss_persionalized_model()
        glob_acc = np.sum(stats[2])*1.0/np.sum(stats[1])
        train_acc = np.sum(stats_train[2])*1.0/np.sum(stats_train[1])
        # train_loss = np.dot(stats_train[3], stats_train[1])*1.0/np.sum(stats_train[1])
        train_loss = sum([x * y for (x, y) in zip(stats_train[1], stats_train[3])]).item() / np.sum(stats_train[1])
        self.rs_glob_acc_per.append(glob_acc)
        self.rs_train_acc_per.append(train_acc)
        self.rs_train_loss_per.append(train_loss)
        #print("stats_train[1]",stats_train[3][0])
        print("Average Personal Accurancy: ", glob_acc)
        print("Average Personal Trainning Accurancy: ", train_acc)
        print("Average Personal Trainning Loss: ",train_loss)
        with open("./results/"+'{}.txt'.format('evaluate_personalized_model'), 'a') as f:
            f.write('{}\t{}\t{}\t{}\n'.format(self.local_epochs, glob_acc, train_acc, train_loss))


    def evaluate_one_step(self):
        for c in self.users:
            c.train_one_step()

        stats = self.test()  
        stats_train = self.train_error_and_loss()

        # set local model back to client for training process.
        for c in self.users:
            c.update_parameters(c.local_model)

        glob_acc = np.sum(stats[2])*1.0/np.sum(stats[1])
        train_acc = np.sum(stats_train[2])*1.0/np.sum(stats_train[1])
        # train_loss = np.dot(stats_train[3], stats_train[1])*1.0/np.sum(stats_train[1])
        train_loss = sum([x * y for (x, y) in zip(stats_train[1], stats_train[3])]).item() / np.sum(stats_train[1])
        self.rs_glob_acc_per.append(glob_acc)
        self.rs_train_acc_per.append(train_acc)
        self.rs_train_loss_per.append(train_loss)
        #print("stats_train[1]",stats_train[3][0])
        print("Average Personal Accurancy: ", glob_acc)
        print("Average Personal Trainning Accurancy: ", train_acc)
        print("Average Personal Trainning Loss: ",train_loss)
        with open("./results/"+'{}.txt'.format('evaluate_one_step'), 'a') as f:
            f.write('{}\t{}\t{}\t{}\n'.format(self.local_epochs, glob_acc, train_acc, train_loss))

