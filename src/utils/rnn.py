import torch

def permute_out_rnn_style(f):
    def wrapper(*args, **kwargs):
        

        outputs = f(*args, **kwargs)
        outputs_permuted = outputs
       
        if isinstance(outputs, torch.Tensor) and outputs.ndim == 3:
            outputs_permuted = outputs.permute(0, 2, 1)
        
        if isinstance(outputs, tuple):
            outputs_permuted = []
            for output in outputs:
                if isinstance(output, torch.Tensor) and output.ndim == 3:
                    outputs_permuted.append(output.permute(0, 2, 1))

        return outputs

    return wrapper

def permute_in_rnn_style(f):
    def wrapper(*args, **kwargs):
        
        args_permuted = []
        for arg in args:
            if isinstance(arg, torch.Tensor) and arg.ndim == 3:
                args_permuted.append(arg.permute(0, 2, 1))
            else:
                args_permuted.append(arg)
        
        kwargs_permuted = {}
        for kw, arg in kwargs.items():
            if isinstance(arg, torch.Tensor) and arg.ndim == 3:
                kwargs_permuted.update({kw:arg.permute(0, 2, 1)})
            else:
                kwargs_permuted.append({kw:arg})

        outputs = f(*args_permuted, **kwargs_permuted)
        return outputs
    
    return wrapper

def permute_rnn_style(f):
    def wrapper(*args, **kwargs):
        
        args_permuted = []
        for arg in args:
            if isinstance(arg, torch.Tensor) and arg.ndim == 3:
                args_permuted.append(arg.permute(0, 2, 1))
            else:
                args_permuted.append(arg)
        
        kwargs_permuted = {}
        for kw, arg in kwargs.items():
            if isinstance(arg, torch.Tensor) and arg.ndim == 3:
                kwargs_permuted.update({kw:arg.permute(0, 2, 1)})
            else:
                kwargs_permuted.update({kw:arg})

        outputs = f(*args_permuted, **kwargs_permuted)
        outputs_permuted = outputs
       
        if isinstance(outputs, torch.Tensor) and outputs.ndim == 3:
            outputs_permuted = outputs.permute(0, 2, 1)
        
        if isinstance(outputs, tuple):
            outputs_permuted = []
            for output in outputs:
                if isinstance(output, torch.Tensor) and output.ndim == 3:
                    outputs_permuted.append(output.permute(0, 2, 1))
        
        return outputs_permuted
    
    return wrapper