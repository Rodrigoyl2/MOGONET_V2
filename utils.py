import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

cuda = True if torch.cuda.is_available() else False
# Custom accuracy score function
def custom_accuracy_score(y_true, y_pred):
    correct_predictions = np.sum(y_true == y_pred)
    total_predictions = len(y_true)
    accuracy = correct_predictions / total_predictions
    return accuracy

# Custom F1 score function
def custom_f1_score(y_true, y_pred):
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    predicted_positives = np.sum(y_pred == 1)
    actual_positives = np.sum(y_true == 1)

    precision = true_positives / (predicted_positives + 1e-8)
    recall = true_positives / (actual_positives + 1e-8)

    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    return f1

# Custom F1 score function
def custom_weighted_f1_score(y_true, y_pred):
    # Calculate F1 score for each class
    unique_classes = np.unique(y_true)
    f1_scores = []

    for cls in unique_classes:
        true_positives = np.sum((y_true == cls) & (y_pred == cls))
        predicted_positives = np.sum(y_pred == cls)
        actual_positives = np.sum(y_true == cls)

        precision = true_positives / (predicted_positives + 1e-8)
        recall = true_positives / (actual_positives + 1e-8)

        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        f1_scores.append(f1)

    # Calculate weighted average of F1 scores
    class_counts = [np.sum(y_true == cls) for cls in unique_classes]
    weighted_f1 = np.average(f1_scores, weights=class_counts)

    return weighted_f1

# Custom macro F1 score function
def custom_macro_f1_score(y_true, y_pred):
    # Calculate F1 score for each class
    unique_classes = np.unique(y_true)
    f1_scores = []

    for cls in unique_classes:
        true_positives = np.sum((y_true == cls) & (y_pred == cls))
        predicted_positives = np.sum(y_pred == cls)
        actual_positives = np.sum(y_true == cls)

        precision = true_positives / (predicted_positives + 1e-8)
        recall = true_positives / (actual_positives + 1e-8)

        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        f1_scores.append(f1)

    # Calculate macro F1 score by averaging F1 scores across classes
    macro_f1 = np.mean(f1_scores)

    return macro_f1

# Custom ROC AUC score function
def custom_roc_auc_score(y_true, y_prob):
    positive_class_probabilities = y_prob[:, 1]  # Probability of positive class

    # Sort true labels and probabilities in descending order
    sorted_indices = np.argsort(positive_class_probabilities, kind='heapsort')[::-1]
    y_true_sorted = y_true[sorted_indices]
    positive_class_prob_sorted = positive_class_probabilities[sorted_indices]

    n_positive = np.sum(y_true == 1)
    n_negative = np.sum(y_true == 0)

    tpr = [0]  # True Positive Rate
    fpr = [0]  # False Positive Rate
    auc = 0

    for i in range(len(y_true_sorted)):
        if y_true_sorted[i] == 1:
            tpr.append(tpr[-1] + 1 / n_positive)
            fpr.append(fpr[-1])
        else:
            tpr.append(tpr[-1])
            fpr.append(fpr[-1] + 1 / n_negative)
            auc += (fpr[-1] - fpr[-2]) * tpr[-1]

    return auc

def cal_sample_weight(labels, num_class, use_sample_weight=True):
    if not use_sample_weight:
        return np.ones(len(labels)) / len(labels)
    count = np.zeros(num_class)
    for i in range(num_class):
        count[i] = np.sum(labels==i)
    sample_weight = np.zeros(labels.shape)
    for i in range(num_class):
        sample_weight[np.where(labels==i)[0]] = count[i]/np.sum(count)
    
    return sample_weight


def one_hot_tensor(y, num_dim):
    y_onehot = torch.zeros(y.shape[0], num_dim)
    y_onehot.scatter_(1, y.view(-1,1), 1)
    
    return y_onehot


def cosine_distance_torch(x1, x2=None, eps=1e-8):
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    return 1 - torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)

# old version
# def to_sparse(x):
#     x_typename = torch.typename(x).split('.')[-1]
#     sparse_tensortype = getattr(torch.sparse, x_typename)
#     indices = torch.nonzero(x)
#     if len(indices.shape) == 0:  # if all elements are zeros
#         return sparse_tensortype(*x.shape)
#     indices = indices.t()
#     values = x[tuple(indices[i] for i in range(indices.shape[0]))]
#     return sparse_tensortype(indices, values, x.size())
# new version
def to_sparse(x):
    indices = torch.nonzero(x)
    if len(indices.shape) == 0:  # if all elements are zeros
        return torch.sparse_coo_tensor(indices, [], x.size())
    indices = indices.t()
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]
    return torch.sparse_coo_tensor(indices, values, x.size())

def cal_adj_mat_parameter(edge_per_node, data, metric="cosine"):
    assert metric == "cosine", "Only cosine distance implemented"
    dist = cosine_distance_torch(data, data)
    parameter = torch.sort(dist.reshape(-1,)).values[edge_per_node*data.shape[0]]
    return np.ndarray.item(parameter.data.cpu().numpy())


def graph_from_dist_tensor(dist, parameter, self_dist=True):
    if self_dist:
        assert dist.shape[0]==dist.shape[1], "Input is not pairwise dist matrix"
    g = (dist <= parameter).float()
    if self_dist:
        diag_idx = np.diag_indices(g.shape[0])
        g[diag_idx[0], diag_idx[1]] = 0
        
    return g


def gen_adj_mat_tensor(data, parameter, metric="cosine"):
    assert metric == "cosine", "Only cosine distance implemented"
    dist = cosine_distance_torch(data, data)
    g = graph_from_dist_tensor(dist, parameter, self_dist=True)
    if metric == "cosine":
        adj = 1-dist
    else:
        raise NotImplementedError
    adj = adj*g 
    adj_T = adj.transpose(0,1)
    I = torch.eye(adj.shape[0])
    if cuda:
        I = I.cuda()
    adj = adj + adj_T*(adj_T > adj).float() - adj*(adj_T > adj).float()
    adj = F.normalize(adj + I, p=1)
    adj = to_sparse(adj)
    
    return adj


def gen_test_adj_mat_tensor(data, trte_idx, parameter, metric="cosine"):
    assert metric == "cosine", "Only cosine distance implemented"
    adj = torch.zeros((data.shape[0], data.shape[0]))
    if cuda:
        adj = adj.cuda()
    num_tr = len(trte_idx["tr"])
    
    dist_tr2te = cosine_distance_torch(data[trte_idx["tr"]], data[trte_idx["te"]])
    g_tr2te = graph_from_dist_tensor(dist_tr2te, parameter, self_dist=False)
    if metric == "cosine":
        adj[:num_tr,num_tr:] = 1-dist_tr2te
    else:
        raise NotImplementedError
    adj[:num_tr,num_tr:] = adj[:num_tr,num_tr:]*g_tr2te
    
    dist_te2tr = cosine_distance_torch(data[trte_idx["te"]], data[trte_idx["tr"]])
    g_te2tr = graph_from_dist_tensor(dist_te2tr, parameter, self_dist=False)
    if metric == "cosine":
        adj[num_tr:,:num_tr] = 1-dist_te2tr
    else:
        raise NotImplementedError
    adj[num_tr:,:num_tr] = adj[num_tr:,:num_tr]*g_te2tr # retain selected edges
    
    adj_T = adj.transpose(0,1)
    I = torch.eye(adj.shape[0])
    if cuda:
        I = I.cuda()
    adj = adj + adj_T*(adj_T > adj).float() - adj*(adj_T > adj).float()
    adj = F.normalize(adj + I, p=1)
    adj = to_sparse(adj)
    
    return adj


def save_model_dict(folder, model_dict):
    if not os.path.exists(folder):
        os.makedirs(folder)
    for module in model_dict:
        torch.save(model_dict[module].state_dict(), os.path.join(folder, module+".pth"))
            
    
def load_model_dict(folder, model_dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for module in model_dict:
        module_path = os.path.join(folder, module + ".pth")
        if os.path.exists(module_path):
            state_dict = torch.load(module_path, map_location=device)
            try:
                model_dict[module].load_state_dict(state_dict)
            except RuntimeError as e:
                print(f"RuntimeError loading module {module}: {e}")
                print(f"Loading module {module} with strict=False")
                model_dict[module].load_state_dict(state_dict, strict=False)  # Ignore missing keys
            model_dict[module].to(device)
            print(f"Module {module} loaded!")
        else:
            print(f"WARNING: Module {module} from model_dict is not loaded!")

    return model_dict


def add_gaussian_noise(data, mean=0, std_dev=.01):
    # Debugging: Print the parameters to check their values
    print(f"Data shape: {data.shape}, Mean: {mean}, Std Dev: {std_dev}")
    
    # Set a default standard deviation if std_dev is None
    std_dev = std_dev if std_dev is not None else .01

    # Generate Gaussian noise
    gaussian_noise = np.random.normal(mean, std_dev, data.shape)
    
    # Add the noise to the original data
    data_noisy = data + gaussian_noise
    return data_noisy

def calculate_train_accuracy(data_list, adj_list, true_labels, model_dict):
    # Set the model to evaluation mode
    for m in model_dict:
        model_dict[m].eval()

    num_view = len(data_list)
    ci_list = []
    
    # Get predictions from each GCN_E + Classifier_1 combination
    for i in range(num_view):
        ci = model_dict["C{:}".format(i+1)](model_dict["E{:}".format(i+1)](data_list[i], adj_list[i]))
        ci_list.append(ci)

    # Combine the predictions using VCDN if there are multiple views
    if num_view >= 2:
        predictions = model_dict["C"](ci_list)
    else:
        predictions = ci_list[0]

    # Convert predictions to class labels
    _, predicted_labels = torch.max(predictions, 1)

    # Calculate accuracy
    correct = (predicted_labels == true_labels).sum().item()
    total = true_labels.size(0)
    accuracy = correct / total

    return accuracy

def calculate_test_loss(data_list, adj_list, true_labels, model_dict, criterion, te_idx):
    # Set the model to evaluation mode
    for m in model_dict:
        model_dict[m].eval()

    num_view = len(data_list)
    ci_list = []

    # Get predictions from each GCN_E + Classifier_1 combination
    for i in range(num_view):
        ci = model_dict["C{:}".format(i+1)](model_dict["E{:}".format(i+1)](data_list[i], adj_list[i]))
        ci_list.append(ci)

    # Combine the predictions using VCDN if there are multiple views
    if num_view >= 2:
        predictions = model_dict["C"](ci_list)
    else:
        predictions = ci_list[0]

    # Select the test subset of the predictions
    test_predictions = predictions[te_idx]
    true_labels_tensor = torch.tensor(true_labels[te_idx], dtype=torch.long)

    # Now use true_labels_tensor in the criterion
    loss = criterion(test_predictions, true_labels_tensor)

    return loss.item()
