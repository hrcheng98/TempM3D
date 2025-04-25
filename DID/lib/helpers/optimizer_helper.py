import torch.optim as optim


def build_optimizer(cfg_optimizer, model):
    weights, biases = [], []
    for name, param in model.named_parameters():
        if 'bias' in name:
            biases += [param]
        else:
            weights += [param]

    parameters = [{'params': biases, 'weight_decay': 0},
                  {'params': weights, 'weight_decay': cfg_optimizer['weight_decay']}]

    if cfg_optimizer['type'] == 'adam':
        optimizer = optim.Adam(parameters, lr=cfg_optimizer['lr'])
        # for name, k in model.named_parameters():
        #     if name.split('.')[0] == 'backbone': k.requires_grad = False
        # optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg_optimizer['lr'])

        # optimizer = optim.AdamW(parameters, lr=cfg_optimizer['lr'], betas=(0.95, 0.99), weight_decay=1e-6)
        # optimizer = optim.AdamW(parameters, lr=cfg_optimizer['lr'])
    elif cfg_optimizer['type'] == 'sgd':
        optimizer = optim.SGD(parameters, lr=cfg_optimizer['lr'], momentum=0.9)
    else:
        raise NotImplementedError("%s optimizer is not supported" % cfg_optimizer['type'])

    return optimizer