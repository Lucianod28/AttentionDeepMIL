from __future__ import print_function
import math

import numpy as np

import argparse
import torch
import torch.utils.data as data_utils
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from dataloader import MnistBags
from breast_cancer_dataloader import BreastCancerBags
from model import Attention, GatedAttention, CancerAttention

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST bags Example')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default=0.0005, metavar='LR',
                    help='learning rate (default: 0.0005)')
parser.add_argument('--reg', type=float, default=1e-3, metavar='R',
                    help='weight decay')
parser.add_argument('--target_number', type=int, default=9, metavar='T',
                    help='bags have a positive labels if they contain at least one 9')
parser.add_argument('--mean_bag_length', type=int, default=10, metavar='ML',
                    help='average bag length')
parser.add_argument('--var_bag_length', type=int, default=2, metavar='VL',
                    help='variance of bag length')
parser.add_argument('--num_bags_train', type=int, default=500, metavar='NTrain',
                    help='number of bags in training set')
parser.add_argument('--num_bags_test', type=int, default=50, metavar='NTest',
                    help='number of bags in test set')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--cancer', action='store_true', default=False,
                    help='Use the breast cancer dataset')
parser.add_argument('--model', type=str, default='attention', help='Choose b/w attention and gated_attention')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    print('\nGPU is ON!')

# hyperparameters
percentage_labeled = .2
# max_unsupervised_weight = (30 * (500* percentage_labeled)) / (500 - 50)
max_unsupervised_weight = 3
epoch_with_max_rampup = 60 # for rampup function
alpha = 0.6 # for weighting function


def ramp_up_function(epoch, epoch_with_max_rampup=80):
    """ Ramps the value of the weight and learning rate according to the epoch
        according to the paper
    Arguments:
        {int} epoch
        {int} epoch where the rampup function gets its maximum value
    Returns:
        {float} -- rampup value
    """

    if epoch == 0:
        return 0
    elif epoch < epoch_with_max_rampup:
        p = max(0.0, float(epoch)) / float(epoch_with_max_rampup)
        p = 1.0 - p
        return math.exp(-p*p*5.0)
    else:
        return 1.0


def train(train_loader, epoch, Z, z_tilde):
    model.train()
    supervised_loss = 0.
    running_temporal_ensembling_loss = 0.
    train_error = 0.
    Y_probs = torch.zeros(len(train_loader))
    # reset gradients
    optimizer.zero_grad()
    total_loss = 0.
    for batch_idx, (data, label) in enumerate(train_loader):
        # print(batch_idx)
        labeled = batch_idx % 100 < 100 * percentage_labeled
        # labeled = True

        bag_label = label[0]
        if args.cuda:
            data, bag_label = data.cuda(), bag_label.cuda()
        data = Variable(data)
        bag_label = Variable(bag_label)

        # calculate loss and metrics
        unsupervised_weight = ramp_up_function(epoch, epoch_with_max_rampup) * max_unsupervised_weight

        loss, _, Y_prob, neg_log_likelihood, temporal_ensembling_loss = model.calculate_objective(data, bag_label, z_tilde[batch_idx], unsupervised_weight, labeled)
        total_loss += loss

        Y_probs[batch_idx] = Y_prob.detach() # save for temporal ensembling later
        # Z[batch_idx] = alpha * Z[batch_idx] + (1 - alpha) * Y_prob
        # z_tilde[batch_idx] = Z[batch_idx] / (1 - alpha ** epoch)

        if neg_log_likelihood != 0:
            supervised_loss += neg_log_likelihood.item()
        if temporal_ensembling_loss != 0:
            running_temporal_ensembling_loss += temporal_ensembling_loss.item()
        error, _ = model.calculate_classification_error(data, bag_label)
        train_error += error
        # backward pass
        # loss.backward()
    total_loss /= len(train_loader)
    total_loss.backward()
    # step
    optimizer.step()

    # update temporal ensembling variables
    Z = alpha * Z + (1 - alpha) * Y_probs
    z_tilde = Z / (1 - alpha ** epoch)

    # calculate loss and error for epoch
    supervised_loss /= len(train_loader) * percentage_labeled
    running_temporal_ensembling_loss /= len(train_loader)
    train_error /= len(train_loader)

    #print('Epoch: {}, Loss: {:.4f}, Train error: {:.4f}'.format(epoch, train_loss.cpu().numpy()[0], train_error))
    return supervised_loss, running_temporal_ensembling_loss, train_error, Z, z_tilde

def train_only_supervised(train_loader, epoch):
    model.train()
    train_loss = 0.
    train_error = 0.
    for batch_idx, (data, label) in enumerate(train_loader):
        bag_label = label[0]
        if args.cuda:
            data, bag_label = data.cuda(), bag_label.cuda()
        data, bag_label = Variable(data), Variable(bag_label)

        # reset gradients
        optimizer.zero_grad()
        # calculate loss and metrics
        loss, _ = model.calculate_neg_log_objective(data, bag_label)
        train_loss += loss.data[0]
        error, _ = model.calculate_classification_error(data, bag_label)
        train_error += error
        # backward pass
        loss.backward()
        # step
        optimizer.step()

    # calculate loss and error for epoch
    train_loss /= len(train_loader)
    train_error /= len(train_loader)

    # print('Epoch: {}, Loss: {:.4f}, Train error: {:.4f}'.format(epoch, train_loss.cpu().numpy()[0], train_error))
    return train_loss, 0, train_error

def test(test_loader):
    model.eval()
    test_loss = 0.
    test_error = 0.
    for batch_idx, (data, label) in enumerate(test_loader):
        bag_label = label[0]
        instance_labels = label[1]
        if args.cuda:
            data, bag_label = data.cuda(), bag_label.cuda()
        data, bag_label = Variable(data), Variable(bag_label)
        loss, attention_weights, _, _, _ = model.calculate_objective(data, bag_label)
        test_loss += loss.data[0]
        error, predicted_label = model.calculate_classification_error(data, bag_label)
        test_error += error

        if batch_idx < 5:  # plot bag labels and instance labels for first 5 bags
            bag_level = (bag_label.cpu().data.numpy()[0], int(predicted_label.cpu().data.numpy()[0][0]))
            instance_level = list(zip(instance_labels.numpy()[0].tolist(),
                                 np.round(attention_weights.cpu().data.numpy()[0], decimals=3).tolist()))

            print('\nTrue Bag Label, Predicted Bag Label: {}\n'
                  'True Instance Labels, Attention Weights: {}'.format(bag_level, instance_level))

    test_error /= len(test_loader)
    test_loss /= len(test_loader)

    print('\nTest Set, Loss: {:.4f}, Test error: {:.4f}'.format(test_loss.cpu().numpy()[0], test_error))
    return test_error, test_loss


if __name__ == "__main__":
    print('Load Train and Test Set')
    print('%d epochs, %d train bags, %d test bags, %d mean bag length, %d variance bag length' % (args.epochs, args.num_bags_train, args.num_bags_test, args.mean_bag_length, args.var_bag_length))
    loader_kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    if args.cancer:
        train_loader = data_utils.DataLoader(BreastCancerBags(transforms.ToTensor(), True))
        test_loader = data_utils.DataLoader(BreastCancerBags(transforms.ToTensor(), False))
    else:
        train_loader = data_utils.DataLoader(MnistBags(target_number=args.target_number,
                                                    mean_bag_length=args.mean_bag_length,
                                                    var_bag_length=args.var_bag_length,
                                                    num_bag=args.num_bags_train,
                                                    seed=args.seed,
                                                    train=True),
                                            batch_size=1,
                                            shuffle=False,
                                            **loader_kwargs)
        test_loader = data_utils.DataLoader(MnistBags(target_number=args.target_number,
                                                    mean_bag_length=args.mean_bag_length,
                                                    var_bag_length=args.var_bag_length,
                                                    num_bag=args.num_bags_test,
                                                    seed=args.seed,
                                                    train=False),
                                            batch_size=1,
                                            shuffle=False,
                                            **loader_kwargs)

    print('Init Model')
    if args.model=='attention':
        if args.cancer:
            model = CancerAttention()
        else:
            model = Attention()
    elif args.model=='gated_attention':
        assert not args.cancer
        model = GatedAttention()
    if args.cuda:
        model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.reg)
    print('Start Training')
    writer = SummaryWriter()
    # Temporal ensembling variables
    Z = torch.zeros(len(train_loader))
    z_tilde = torch.zeros(len(train_loader))
    print('max unsupervised weight is:', max_unsupervised_weight)
    for epoch in range(1, args.epochs + 1):
        supervised_loss, temporal_ensembling_loss, train_error, Z, z_tilde = train(train_loader, epoch, Z, z_tilde)
        # supervised_loss, temporal_ensembling_loss, train_error = train_only_supervised(train_loader, epoch)
        writer.add_scalar("logs/supervised_loss", supervised_loss, epoch)
        writer.add_scalar("logs/temporal_ensembling_loss", temporal_ensembling_loss, epoch)
        writer.add_scalar("logs/train_error", train_error, epoch)

        print('Start Testing')
        test_error, test_loss = test(test_loader)
        writer.add_scalar("logs/test_loss", test_loss, epoch)
        writer.add_scalar("logs/test_error", test_error, epoch)


