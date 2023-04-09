import torch
import gpytorch
import tqdm

def training(model, X_train, y_train, n_epochs=200, lr=0.3, loss_threshold=0.00001, fix_noise_variance=None, verbose=True):
    '''
    Training function for GPs. 

    Parameters
    ----------
    model : gpytorch.models.ExactGP
        GP model to be trained.
    X_train : torch.tensor
        Training data.
    y_train : torch.tensor
        Training labels.
    n_epochs : int, optional
        Number of epochs to train for. The default is 200.
    lr : float, optional
        Learning rate. The default is 0.3.
    loss_threshold : float, optional
        Threshold for loss. The default is 0.00001.
    fix_noise_variance : float, optional
        If not None, fix the noise variance to this value. The default is None.
    verbose : bool, optional
        If True, print loss at each epoch. The default is True.
    
    Returns
    -------
    ls : list
        List of losses at each epoch.
    mll : gpytorch.mlls.ExactMarginalLogLikelihood
        Marginal log likelihood.
    '''
    model.train()
    model.likelihood.train()
    
    # Set the initial weight of each kernel component to 1 / n_comp. If the model is not a mixture model, n_comp = 1.
    try:
        n_comp = len([m for m in model.covar_module.data_covar_module.kernels])
        for i in range(n_comp):
            model.covar_module.data_covar_module.kernels[i].outputscale = (1 / n_comp)
    except AttributeError:
        n_comp = 1

    # Use the adam optimizer
    if fix_noise_variance is not None:
        model.likelihood.noise = fix_noise_variance
        training_parameters = [p for name, p in model.named_parameters()
                               if not name.startswith('likelihood')]
    else:
        training_parameters = model.parameters()
        
    optimizer = torch.optim.Adam(training_parameters, lr=lr)

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
    model = model.double()
    counter = 0
    ls = list()
    with tqdm.trange(n_epochs, disable=not verbose) as bar:
        for i in bar:
    
            optimizer.zero_grad()
            
            output = model(X_train.double())
            loss = -mll(output, y_train)
            if (hasattr(model.covar_module, 'data_covar_module')):
                if (hasattr(model.covar_module.data_covar_module, 'kernels')):
                    with torch.no_grad():
                        for j in range(n_comp):
                            model.covar_module.data_covar_module.kernels[j].outputscale =  \
                            model.covar_module.data_covar_module.kernels[j].outputscale /  \
                            sum([model.covar_module.data_covar_module.kernels[i].outputscale for i in range(n_comp)])
            else:
                pass
            loss.backward()
            ls.append(loss.item())
            optimizer.step()
            if (i > 0):
                if abs(ls[counter - 1] - ls[i]) < loss_threshold:
                    break
            counter = counter + 1
                        
            # display progress bar
            postfix = dict(Loss=f"{loss.item():.3f}",
                           noise=f"{model.likelihood.noise.item():.3}")
            
            if (hasattr(model.covar_module, 'base_kernel') and
                hasattr(model.covar_module.base_kernel, 'lengthscale')):
                lengthscale = model.covar_module.base_kernel.lengthscale
                if lengthscale is not None:
                    lengthscale = lengthscale.squeeze(0).detach().cpu().numpy()
            else:
                lengthscale = model.covar_module.lengthscale

            if lengthscale is not None:
                if len(lengthscale) > 1:
                    lengthscale_repr = [f"{l:.3f}" for l in lengthscale]
                    postfix['lengthscale'] = f"{lengthscale_repr}"
                else:
                    postfix['lengthscale'] = f"{lengthscale[0]:.3f}"
                
            bar.set_postfix(postfix)
            
    return ls, mll