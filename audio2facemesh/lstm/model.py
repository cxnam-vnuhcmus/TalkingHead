import torch
import torch.nn as nn
import numpy as np
import os

class Audio2FeatureModel(nn.Module):
    def __init__(self, opt):
        super(Audio2FeatureModel, self).__init__()
        self.opt = opt
        # define networks
        self.downsample = nn.Sequential(
                nn.Linear(in_features=opt.input_ndim, out_features=512),
                nn.BatchNorm1d(512),
                nn.LeakyReLU(0.2),
                nn.Linear(512, 512),
                )
        self.LSTM = nn.LSTM(input_size=512,
                            hidden_size=256,
                            num_layers=3,
                            dropout=0,
                            bidirectional=False,
                            batch_first=True)
        self.fc = nn.Sequential(
                nn.Linear(in_features=256, out_features=512),
                nn.BatchNorm1d(512),
                nn.LeakyReLU(0.2),
                nn.Linear(512, 512),
                nn.BatchNorm1d(512),
                nn.LeakyReLU(0.2),
                nn.Linear(512, opt.output_ndim))

        self.init_model()
        self.num_params()
        self.register_buffer("step", torch.zeros(1, dtype=torch.long))
                    
    
    def forward(self, x):
        self.step += 1
        bs, item_len, ndim = x.shape
        x = x.reshape(bs, -1)
        x = self.downsample(x)
        x = x.unsqueeze(1)
        output, (hn, cn) = self.LSTM(x)
        pred = self.fc(output.squeeze(1))
        return pred

    def init_model(self):
        for p in self.parameters():
            if p.dim() > 1: nn.init.xavier_uniform_(p)

    def get_step(self):
        return self.step.data.item()

    def reset_step(self):
        # assignment to parameters or buffers is overloaded, updates internal dict entry
        self.step = self.step.data.new_tensor(1)

    def log(self, path, msg):
        with open(path, "a") as f:
            print(msg, file=f)

    def load(self, path, optimizer=None):
        # Use device of model params as location for loaded state
        device = next(self.parameters()).device

        if torch.cuda.is_available():
            checkpoint = torch.load(str(path), map_location="cuda:"+str(torch.cuda.current_device()))
        else:
            checkpoint = torch.load(str(path), map_location="cpu")
        
        self.step = checkpoint["step"] - 1
        
        self.load_state_dict(checkpoint["model_state"])

        if "optimizer_state" in checkpoint and optimizer is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            
        fname = os.path.basename(path)
        step = self.step.cpu().numpy()[0]
        print(f"Load pretrained model {fname} | Step: {step}")

    def save(self, path, optimizer=None):
        if optimizer is not None:
            torch.save({
                "step": self.step + 1,
                "model_state": self.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            }, str(path))
        else:
            torch.save({
                "step": self.step + 1,
                "model_state": self.state_dict(),
            }, str(path))


    def num_params(self, print_out=True):
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
        if print_out:
            print("Trainable Parameters: %.3fM" % parameters)
        return parameters










