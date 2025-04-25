import os
import torch


def get_checkpoint_state(model=None, optimizer=None, epoch=None):
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        if isinstance(model, torch.nn.DataParallel):
            model_state = model.module.state_dict()
        else:
            model_state = model.state_dict()
    else:
        model_state = None

    return {'epoch': epoch, 'model_state': model_state, 'optimizer_state': optim_state}


def save_checkpoint(state, filename, logger):
    logger.info("==> Saving to checkpoint '{}'".format(filename))
    filename = '{}.pth'.format(filename)
    torch.save(state, filename)


def load_checkpoint(model, optimizer, filename, logger, map_location):
    if os.path.isfile(filename):
        logger.info("==> Loading from checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename, map_location=map_location)
        epoch = checkpoint.get('epoch', -1)
        if model is not None and checkpoint['model_state'] is not None:
            model.load_state_dict(checkpoint['model_state'])
            # st = checkpoint['model_state']
            # for name in list(st.keys()):
            #     if name.split('.')[0] != 'backbone':
            #         st.pop(name)
            # model.load_state_dict(st, strict=False)
            # model.load_state_dict(checkpoint['model_state'], strict=False)
        if optimizer is not None and checkpoint['optimizer_state'] is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(map_location)
        # epoch = 5

        logger.info("==> Done")
    else:
        raise FileNotFoundError

    # '''Do not use opt para'''
    # epoch = 0
    # logger.info("==> Loading from checkpoint '{}'".format(filename))
    # checkpoint = torch.load(filename, map_location=map_location)
    # model.load_state_dict(checkpoint['model_state'])

    return epoch