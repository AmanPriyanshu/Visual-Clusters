from torch import nn
import torch

torch.manual_seed(2020)

class Network(nn.Module):
	def __init__(self, emb_dim=128):
		super(Network, self).__init__()
		self.conv = nn.Sequential(
			nn.Conv2d(1, 32, 5),
			nn.PReLU(),
			nn.MaxPool2d(2, stride=2),
			nn.Dropout(0.3),
			nn.Conv2d(32, 64, 5),
			nn.PReLU(),
			nn.MaxPool2d(2, stride=2),
			nn.Dropout(0.3)
		)
		
		self.fc = nn.Sequential(
			nn.Linear(64*4*4, 512),
			nn.PReLU(),
			nn.Linear(512, emb_dim)
		)
		
	def forward(self, x):
		x = self.conv(x)
		x = x.view(-1, 64*4*4)
		x = self.fc(x)
		# x = nn.functional.normalize(x)
		return x

def init_weights(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight)