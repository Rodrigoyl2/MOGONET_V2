""" Training and testing of the model
"""
import os
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import torch
import torch.nn.functional as F
from models import init_model_dict, init_optim
from utils import one_hot_tensor, cal_sample_weight, gen_adj_mat_tensor, gen_test_adj_mat_tensor, cal_adj_mat_parameter, custom_accuracy_score, custom_weighted_f1_score, custom_macro_f1_score, add_gaussian_noise, calculate_train_accuracy, calculate_test_loss
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
import torch.nn as nn
cuda = True if torch.cuda.is_available() else False

def prepare_trte_data(data_folder, view_list, noise_std_dev=0.01, num_data_points = None):
    num_view = len(view_list)
    labels_tr = np.loadtxt(os.path.join(data_folder, "labels_tr.csv"), delimiter=',')
    labels_te = np.loadtxt(os.path.join(data_folder, "labels_te.csv"), delimiter=',')
    labels_tr = labels_tr.astype(int)
    labels_te = labels_te.astype(int)

    # Adjust for n data points if provided
    if num_data_points is not None:
        labels_tr = labels_tr[:num_data_points]
        labels_te = labels_te[:num_data_points]

    data_tr_list = []
    data_te_list = []
    for i in view_list:
        data_tr = np.loadtxt(os.path.join(data_folder, str(i) + "_tr.csv"), delimiter=',')
        data_te = np.loadtxt(os.path.join(data_folder, str(i) + "_te.csv"), delimiter=',')

        # Adjust for n data points if provided
        if num_data_points is not None:
            data_tr = data_tr[:num_data_points, :]
            data_te = data_te[:num_data_points, :]

        # Add Gaussian noise to training data
        data_tr_noisy = add_gaussian_noise(data_tr, std_dev=noise_std_dev)

        data_tr_list.append(data_tr_noisy)
        data_te_list.append(data_te)

    num_tr = data_tr_list[0].shape[0]
    num_te = data_te_list[0].shape[0]
    data_mat_list = []
    for i in range(num_view):
        data_mat_list.append(np.concatenate((data_tr_list[i], data_te_list[i]), axis=0))
    
    data_tensor_list = []
    for i in range(len(data_mat_list)):
        data_tensor_list.append(torch.FloatTensor(data_mat_list[i]))
        if cuda:
            data_tensor_list[i] = data_tensor_list[i].cuda()

    idx_dict = {"tr": list(range(num_tr)), "te": list(range(num_tr, (num_tr + num_te)))}
    data_train_list = []
    data_all_list = []
    for i in range(len(data_tensor_list)):
        data_train_list.append(data_tensor_list[i][idx_dict["tr"]].clone())
        data_all_list.append(torch.cat((data_tensor_list[i][idx_dict["tr"]].clone(), data_tensor_list[i][idx_dict["te"]].clone()), 0))

    labels = np.concatenate((labels_tr, labels_te))

    return data_train_list, data_all_list, idx_dict, labels


def gen_trte_adj_mat(data_tr_list, data_trte_list, trte_idx, adj_parameter):
    adj_metric = "cosine" # cosine distance
    adj_train_list = []
    adj_test_list = []
    for i in range(len(data_tr_list)):
        adj_parameter_adaptive = cal_adj_mat_parameter(adj_parameter, data_tr_list[i], adj_metric)
        adj_train_list.append(gen_adj_mat_tensor(data_tr_list[i], adj_parameter_adaptive, adj_metric))
        adj_test_list.append(gen_test_adj_mat_tensor(data_trte_list[i], trte_idx, adj_parameter_adaptive, adj_metric))
    
    return adj_train_list, adj_test_list


def train_epoch(data_list, adj_list, label, one_hot_label, sample_weight, model_dict, optim_dict, train_VCDN=True):
    loss_dict = {}
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    for m in model_dict:
        model_dict[m].train()    
    num_view = len(data_list)
    for i in range(num_view):
        optim_dict["C{:}".format(i+1)].zero_grad()
        ci_loss = 0
        ci = model_dict["C{:}".format(i+1)](model_dict["E{:}".format(i+1)](data_list[i],adj_list[i]))
        ci_loss = torch.mean(torch.mul(criterion(ci, label),sample_weight))
        ci_loss.backward()
        optim_dict["C{:}".format(i+1)].step()
        loss_dict["C{:}".format(i+1)] = ci_loss.detach().cpu().numpy().item()
    if train_VCDN and num_view >= 2:
        optim_dict["C"].zero_grad()
        c_loss = 0
        ci_list = []
        for i in range(num_view):
            ci_list.append(model_dict["C{:}".format(i+1)](model_dict["E{:}".format(i+1)](data_list[i],adj_list[i])))
        c = model_dict["C"](ci_list)    
        c_loss = torch.mean(torch.mul(criterion(c, label),sample_weight))
        c_loss.backward()
        optim_dict["C"].step()
        loss_dict["C"] = c_loss.detach().cpu().numpy().item()
    
    return loss_dict
    

def test_epoch(data_list, adj_list, te_idx, model_dict):
    for m in model_dict:
        model_dict[m].eval()
    num_view = len(data_list)
    ci_list = []
    for i in range(num_view):
        ci_list.append(model_dict["C{:}".format(i+1)](model_dict["E{:}".format(i+1)](data_list[i],adj_list[i])))
    if num_view >= 2:
        c = model_dict["C"](ci_list)    
    else:
        c = ci_list[0]
    c = c[te_idx,:]
    prob = F.softmax(c, dim=1).data.cpu().numpy()
    
    return prob


def train_test(data_folder, view_list, num_class,
               lr_e_pretrain, lr_e, lr_c, 
               num_epoch_pretrain, num_epoch, num_data_points = None):
    train_acc_values, train_loss_values = [], []
    test_acc_values, test_loss_values = [], []
    train_acc_values = []
    test_acc_values = []
    test_f1_values = []
    test_auc_values = []
    test_inverval = 50
    num_view = len(view_list)
    dim_hvcdn = pow(num_class,num_view)
    if data_folder == 'ROSMAP':
        adj_parameter = 2
        dim_he_list = [200,200,100]
    if data_folder == 'BRCA':
        adj_parameter = 10
        dim_he_list = [400,400,200]
    data_tr_list, data_trte_list, trte_idx, labels_trte = prepare_trte_data(data_folder, view_list, num_data_points)
    labels_tr_tensor = torch.LongTensor(labels_trte[trte_idx["tr"]])
    onehot_labels_tr_tensor = one_hot_tensor(labels_tr_tensor, num_class)
    sample_weight_tr = cal_sample_weight(labels_trte[trte_idx["tr"]], num_class)
    sample_weight_tr = torch.FloatTensor(sample_weight_tr)
    if cuda:
        labels_tr_tensor = labels_tr_tensor.cuda()
        onehot_labels_tr_tensor = onehot_labels_tr_tensor.cuda()
        sample_weight_tr = sample_weight_tr.cuda()
    adj_tr_list, adj_te_list = gen_trte_adj_mat(data_tr_list, data_trte_list, trte_idx, adj_parameter)
    dim_list = [x.shape[1] for x in data_tr_list]
    model_dict = init_model_dict(num_view, num_class, dim_list, dim_he_list, dim_hvcdn)
    for m in model_dict:
        if cuda:
            model_dict[m].cuda()
    
    print("\nPretrain GCNs...")
    optim_dict = init_optim(num_view, model_dict, lr_e_pretrain, lr_c)
    for epoch in range(num_epoch_pretrain):
        train_epoch(data_tr_list, adj_tr_list, labels_tr_tensor, 
                    onehot_labels_tr_tensor, sample_weight_tr, model_dict, optim_dict, train_VCDN=False)
    print("\nTraining...")
    optim_dict = init_optim(num_view, model_dict, lr_e, lr_c)
    # Training loop
    for epoch in range(num_epoch + 1):
        # Train and get loss
        train_loss = train_epoch(data_tr_list, adj_tr_list, labels_tr_tensor, 
                                 onehot_labels_tr_tensor, sample_weight_tr, model_dict, optim_dict)
        train_loss_values.append(train_loss)

        # Calculate training accuracy
        # You need to implement a function to calculate train accuracy
        train_accuracy = calculate_train_accuracy(data_tr_list, adj_tr_list, labels_tr_tensor, model_dict)
        train_acc_values.append(train_accuracy)

        # Test every 'test_interval' epochs
        if epoch % test_inverval == 0:
            te_prob = test_epoch(data_trte_list, adj_te_list, trte_idx["te"], model_dict)

            # Calculate test accuracy and loss
            test_accuracy = accuracy_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))
            test_acc_values.append(test_accuracy)
            criterion = nn.CrossEntropyLoss()
            te_idx = torch.LongTensor(trte_idx["te"])  
            test_loss = calculate_test_loss(data_trte_list, adj_te_list, labels_trte, model_dict, criterion, te_idx)
            test_loss_values.append(test_loss)
    
    # print("type:", type(train_loss_values))
    # print("type:", type(test_loss_values))
    # print("train_acc_values:", train_acc_values)
    # print("test_acc_values:", test_acc_values)
    # Before plotting, check and print the types and first few elements
    print("train_loss_values type:", type(train_loss_values))
    print("train_loss_values sample:", train_loss_values[:5])
    print("test_loss_values type:", type(test_loss_values))
    print("test_loss_values sample:", test_loss_values[:5])


    # Plotting
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_acc_values, label='Training Accuracy')
    plt.title('Accuracy over epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # # Plotting
    # plt.figure(figsize=(12, 5))
    # train_loss_C = [d['C'] for d in train_loss_values]
    # # Plot for aggregate loss C
    # plt.subplot(1, 2, 1)
    # plt.plot(train_loss_C, label='Training Loss C')
    # plt.title('Aggregate Loss C over epochs')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.legend()

    # # Extracting individual components
    # train_loss_C1 = [d['C1'] for d in train_loss_values]
    # train_loss_C2 = [d['C2'] for d in train_loss_values]
    # train_loss_C3 = [d['C3'] for d in train_loss_values]
    # train_loss_C = [d['C'] for d in train_loss_values]

    # # Plotting
    # plt.figure(figsize=(12, 10))

    # # Plot for C1
    # plt.subplot(2, 2, 1)
    # plt.plot(train_loss_C1, label='Training Loss C1')
    # plt.title('Loss C1 over epochs')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.legend()

    # # Plot for C2
    # plt.subplot(2, 2, 2)
    # plt.plot(train_loss_C2, label='Training Loss C2')
    # plt.title('Loss C2 over epochs')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.legend()

    # # Plot for C3
    # plt.subplot(2, 2, 3)
    # plt.plot(train_loss_C3, label='Training Loss C3')
    # plt.title('Loss C3 over epochs')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.legend()

    # # Plot for aggregate loss C
    # plt.subplot(2, 2, 4)
    # plt.plot(train_loss_C, label='Training Loss C')
    # plt.title('Aggregate Loss C over epochs')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.legend()

    # plt.show()
    accuracy = test_acc_values[-1] if test_acc_values else None
    # Return the final testing accuracy
    return model_dict, accuracy