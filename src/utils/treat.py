import torch
import numpy as np

from src.utils.loss import root_mean_square_error


def create_identical_batch(instance, n=8):
    id_batch = []
    for entry in instance:
        id_batch.append(torch.stack(n*[entry]))
    return id_batch


def test_create_identical_batch(batch):

    id_batch = create_identical_batch(batch, n=128)

    assert len(id_batch) == len(batch)
    assert len(id_batch[0]) == len(batch[0])

    for entry in id_batch:
        assert entry.std(0).sum() == 0

    for e_idx, entry in enumerate(id_batch):
        instance = entry[0]
        for i_idx, _ in enumerate(entry):
            assert (id_batch[e_idx][i_idx] == instance).all()


def model_step(model, X, T):
    Y_hat, *_ = model(X, T)
    mu, var = torch.chunk(Y_hat, 2,-1)
    return mu, torch.sigmoid(var)


def compute_inferred_uncertainty(model, X, T):       
    return model_step(model, X, T)


def compute_sampled_uncertainty(model, X, T):
    X, T = create_identical_batch((X, T))
    mu, _ = model_step(model, X, T)
    return mu.mean(1), mu.var(1)


def input_to_numpy(f):
    def wrapper(*args, **kwargs):
        args_converted = []
        for arg in args:
            if isinstance(arg, torch.Tensor):
                args_converted.append(arg.detach().squeeze().numpy())
            else:
                args_converted.append(arg)

        kwargs_converted = {}
        for k, arg in kwargs.items():
            if isinstance(arg, torch.Tensor):
                kwargs_converted[k] = arg.detach().squeeze().numpy()
            else:
                kwargs_converted[k] = arg
        return f(*args_converted, **kwargs_converted)
    return wrapper


def detach(f):
    def wrapper(*args, **kwargs):
        detached = []
        for out in f(*args, **kwargs):

            if isinstance(out, dict):
                detached = {}
                for k, v in out.items():
                    if isinstance(v, torch.Tensor):
                        detached[k] = v.detach()
                    else:
                        detached[k] = v

            elif isinstance(out, torch.Tensor):
                detached.append(out.detach())
            else:
                detached.append(out)
        return detached
    return wrapper


def freeze_parameters(model):
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def initialize_randn(treatments):
    return torch.nn.Parameter(torch.randn_like(treatments))


def apply_treatment_constraint(compute_uncertainty, constraint_fn):
    def model_with_treatment_constraint(instance, treatments):
        treatments = constraint_fn(treatments)
        return compute_uncertainty(**instance, treatments=treatments)
    return model_with_treatment_constraint


@detach
def select_treatment(
    model, 
    instance, 
    target, 
    objective, 
    constraints, 
    optimizer_cls, 
    optimization_loop, 
    n_iter=100, 
    **optimizer_kwargs
    ):
    # freeze model parameters and wrap compute uncertainty with constraint
    compute_uncertainty = freeze_parameters(model).compute_uncertainty
    compute_uncertainty = apply_treatment_constraint(
        compute_uncertainty=compute_uncertainty,
        constraint_fn=constraints
        )
    
    # initialize treatments
    treatments = initialize_randn(instance['treatments'])

    # remove redundant instance information
    instance.pop('treatments')
    instance.pop('initial_state')

    # initialize optimizer
    optimizer = optimizer_cls([treatments], **optimizer_kwargs)

    # run the optimization loop
    out = optimization_loop(
        objective=objective,
        compute_uncertainty=compute_uncertainty,
        optimizer=optimizer,
        instance=instance,
        treatments=treatments,
        target=target,
        n_iter=n_iter
        )
    return out


@detach
def sgdlike_loop(
        objective, 
        compute_uncertainty, 
        optimizer, 
        instance, 
        treatments, 
        target, 
        n_iter
        ):
    losses = []
    for _ in range(n_iter):
        # compute_uncertainty_ = copy.deepcopy(compute_uncertainty)
        mu, var, treatments, loss = sgdlike_step(
            objective=objective,
            compute_uncertainty=compute_uncertainty,
            optimizer=optimizer,
            instance=instance,
            treatments=treatments,
            target=target,
            )
        losses.append(loss)
    return mu, var, treatments, losses


def sgdlike_step(
        objective,
        compute_uncertainty,
        optimizer,
        instance,
        treatments,
        target
        ):

    optimizer.zero_grad()
    mu, var = compute_uncertainty(instance, treatments)
    loss = objective(mu, var, target)
    loss.backward()
    optimizer.step()

    return mu.detach(), var.detach(), treatments, loss.item()


def lbfgslike_loop(
        objective,
        compute_uncertainty,
        optimizer,
        instance,
        treatments,
        target,
        n_iter
        ):
    return lbfgslike_step(
        objective=objective,
        compute_uncertainty=compute_uncertainty,
        optimizer=optimizer,
        instance=instance,
        treatments=treatments,
        target=target
        )


@detach
def lbfgslike_step(
        objective,
        compute_uncertainty,
        optimizer,
        instance,
        treatments,
        target
        ):
    
    losses = []
    mu = None
    var = None

    def closure():
        nonlocal mu
        nonlocal var
        optimizer.zero_grad()
        mu, var = compute_uncertainty(instance, treatments)
        loss = objective(mu, var, target)
        loss.backward()
        losses.append(loss.item())
        return loss

    optimizer.step(closure)
    return mu, var, treatments, losses


def lr_sweep(values, treatment_selection):
    outcomes = {}
    for lr in values:
        *_, losses = treatment_selection(lr=lr)
        outcomes[lr] = losses[-1]
    return outcomes


@input_to_numpy
def simulate_treatments(
    treatments,
    initial_state,
    t_horizon,
    simulate_outcome
):

    treatments_as_func = lambda tx,dose:treatments[int(tx)]
    outcome = simulate_outcome(
        initial_state=initial_state,
        treatment_dose=1.0,
        t=np.arange(t_horizon),
        intervention=treatments_as_func
        )
    return outcome


@input_to_numpy
def evaluate_treatment_selection(
    treatments,
    target,
    initial_state,
    t_horizon,
    simulate_outcome
):
    outcome = simulate_treatments(
        treatments=treatments,
        initial_state=initial_state,
        t_horizon=t_horizon,
        simulate_outcome=simulate_outcome
    )

    return root_mean_square_error(outcome, target)
