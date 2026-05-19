from time import time


def print_time(func):
    def wrapper(*args, **kwargs):
        start = time()

        ret = func(*args, **kwargs)

        time_spent = time() - start

        print(f"{func.__name__} spent {time_spent:.3f} s")

        return ret

    return wrapper

def model_setup_and_record_attributes(model):
    dtype_record = {}
    requires_grad_record = {}
    # for n, p in model.state_dict().items():
    for n, p in model.named_parameters():
        dtype_record[n] = p.data.dtype
        # p.data = p.data.type(torch.bfloat16)

    # set requires_grad to be true for getting model's derivatives
    for n, p in model.named_parameters():
        requires_grad_record[n] = p.requires_grad
        p.requires_grad = True

    device = next(iter(model.parameters())).device

    return dtype_record, requires_grad_record, device

def model_reset(model, dtype_record, requires_grad_record, device):
    # set to original requires grad
    for n, p in model.named_parameters():
        p.requires_grad = requires_grad_record[n]

    for n, p in model.named_parameters():
        p.data = p.data.type(dtype_record[n])

    model.to(device)
