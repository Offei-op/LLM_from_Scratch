import torch
def calc_loss_batch(input_batch,target_batch,model,device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0,1),target_batch.flatten())

    return loss

def calc_loss_loader(dl,model,device,num_batches=None):
    if len(dl) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(dl)
    else:
        num_batches = min(num_batches,len(dl))
    
    i = 0
    dl = iter(dl)
    loss=0

    for i,(x,y) in enumerate(dl):
        if i < num_batches:
            loss+= calc_loss_batch(x,y,model,device).item()
        else:
            break
        
    return loss/num_batches