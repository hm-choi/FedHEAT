import copy

import torch
__all__ = ['evaluate']


def evaluate(args, model, testloader, device) -> float:
    '''
    Return: accuracy of global test data
    '''
    eval_device = device if not args.multiprocessing else 'cuda:' + args.main_gpu
    eval_model = copy.deepcopy(model)
    eval_model.eval()
    eval_model.to(eval_device)
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(eval_device), data[1].to(eval_device)
            outputs = eval_model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = 100 * correct / float(total)
    print('Accuracy of the network on the 10000 test images: %f %%' % (
            100 * correct / float(total)))
    eval_model.to('cpu')
    return acc
