from torch import optim


def get_optim_and_scheduler(network, epochs, lr, train_all=True):
    if train_all:
        params = network.parameters()
    else:
        params = network.get_params(lr)
    optimizer = optim.SGD(params, weight_decay=2e-5, momentum=.9, nesterov=False, lr=lr)
    step_size = int(epochs * 0.6)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
    print("Step size: %d" % step_size)
    return optimizer, scheduler
