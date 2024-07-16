import torch


def mse_objective(mu, var, target):

    if mu.numel() > 1 or target.numel() > 1:
        mu = mu.squeeze()
        target = target.squeeze()

    return torch.nn.functional.mse_loss(mu, target)


def uncertainty_objective(mu, var, target):
    return var.mean()


def composed_objective(mu, var, target, objectives):
    loss = 0
    for objective, alpha in objectives.items():
        loss += alpha * objective(mu, var, target)
    return loss
