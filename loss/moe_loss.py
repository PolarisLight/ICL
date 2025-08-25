import torch
import torch.nn.functional as F
import torch.nn as nn

def soft_entropy(input, target, reduction='mean'):
    """ Cross entropy that accepts soft targets
    Args:
         pred: predictions for neural network
         targets: targets, can be soft
         size_average: if false, sum is returned instead of mean

    Examples::

        input = torch.FloatTensor([[1.1, 2.8, 1.3], [1.1, 2.1, 4.8]])
        input = torch.autograd.Variable(out, requires_grad=True)

        target = torch.FloatTensor([[0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])
        target = torch.autograd.Variable(y1)
        loss = soft_entropy(input, target)
        loss.backward()
    """
    logsoftmax = nn.LogSoftmax(dim=1)
    res = -target * logsoftmax(input)
    if reduction == 'mean':
        return torch.mean(torch.sum(res, dim=1))
    elif reduction == 'sum':
        return torch.sum(torch.sum(res, dim=1))
    else:
        return

def mix_outputs(outputs, labels, balance=False, label_dis=None):
    label_dis = torch.tensor(
        label_dis, dtype=torch.float, requires_grad=False).cuda()
    label_dis = label_dis.unsqueeze(0).expand(labels.shape[0], -1)
    loss = 0.0
    loss_distillation = 0.0
    loss_parallel = 0.0
    if balance == True:
        moe_ce = []
        for i in range(len(outputs)):
            loss += soft_entropy(outputs[i] + label_dis.log(), labels)
    else:
        #===================new added==========================
        moe_ce = []
        #======================================================
        for i in range(len(outputs)):
            # base ce
            loss += soft_entropy(outputs[i], labels)
            #===================new added==========================
            moe_ce.append(soft_entropy(outputs[i], labels))
            #======================================================
            # distillation loss
            for j in range(len(outputs)):
                if i != j:
                    loss_distillation += F.kl_div(F.log_softmax(outputs[i]),
                                    F.softmax(outputs[j]))
        # #===================new added==========================
        # fuse_outputs_parallel
        reciprocal_sum = 0.0
        for ce in moe_ce:
            reciprocal_sum += 1 / (ce + 1e-8) 
        fused_output = 1/(reciprocal_sum) #
        
        loss_parallel = fused_output * len(outputs)
        #======================================================
    avg_output = sum(outputs) / len(outputs)
    return loss, avg_output, loss_distillation, loss_parallel




def feature_diversity_loss(features, lambda_reg=0.01):

    features = torch.stack(features, dim=1)  # (batch_size, num_experts, feature_dim)
    batch_size, num_experts, feature_dim = features.shape

    feature_pairwise_distances = torch.cdist(features.view(batch_size * num_experts, feature_dim),
                                             features.view(batch_size * num_experts, feature_dim))

    mask = torch.eye(batch_size * num_experts).to(features.device)
    feature_pairwise_distances = feature_pairwise_distances.masked_select(mask == 0).view(batch_size * num_experts, -1)

    loss = feature_pairwise_distances.mean()
    
    return lambda_reg * loss
