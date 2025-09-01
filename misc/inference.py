import numpy as np
import torch
from torch import nn

    
class Inference:
    def __init__(self, model, dataloader, device='cuda', tta=False):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.tta = tta
    
    def run(self):
        model = self.model.eval()
        model = model.to(self.device)
        preds = []
        ys = []
        with torch.no_grad():
            for x, y, *_ in self.dataloader:
                x = x.to(self.device)
                orig_shape = x.shape
                if self.tta:
                    x = x.reshape(-1, *orig_shape[2:])
                pred = model(x)
                if self.tta:
                    pred = pred.reshape(-1, orig_shape[1], *pred.shape[1:])
                    pred = pred.mean(1).cpu()
                else:
                    pred = pred.cpu()
                pred = nn.functional.normalize(pred)
                preds.append(pred)
                ys.append(y)
        preds = torch.cat(preds, 0).numpy()
        ys = torch.cat(ys).numpy().astype(np.float32)
        
        return preds, ys
    