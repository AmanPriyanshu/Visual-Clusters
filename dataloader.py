from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from torchvision import transforms
import torch

torch.manual_seed(2020)
np.random.seed(2020)

class MNIST(Dataset):
	def __init__(self, df, train=True, transform=None):
		self.is_train = train
		self.transform = transform
		self.to_pil = transforms.ToPILImage()
					
		self.images = df.iloc[:, 1:].values.astype(np.uint8)
		self.labels = df.iloc[:, 0].values
		self.index = df.index.values
		
	def __len__(self):
		return len(self.images)
	
	def __getitem__(self, item):
		anchor_img = self.images[item].reshape(28, 28, 1)
		
		if self.is_train:
			anchor_label = self.labels[item]

			positive_list = self.index[self.index!=item][self.labels[self.index!=item]==anchor_label]

			positive_item = np.random.choice(positive_list)
			positive_img = self.images[positive_item].reshape(28, 28, 1)
			
			negative_list = self.index[self.index!=item][self.labels[self.index!=item]!=anchor_label]
			negative_item = np.random.choice(negative_list)
			negative_img = self.images[negative_item].reshape(28, 28, 1)
			
			if self.transform:
				anchor_img = self.transform(self.to_pil(anchor_img))
				positive_img = self.transform(self.to_pil(positive_img))
				negative_img = self.transform(self.to_pil(negative_img))
			
			return anchor_img, positive_img, negative_img, anchor_label
		
		else:
			anchor_label = self.labels[item]
			if self.transform:
				anchor_img = self.transform(self.to_pil(anchor_img))
			return anchor_img, anchor_label

def get_dataloader(batch_size=32):
	df = pd.read_csv("./data/MNIST.csv")
	train_df = df[:40000]
	test_df = df[40000:]
	train_ds = MNIST(train_df, train=True,transform=transforms.Compose([transforms.ToTensor()]))
	test_ds = MNIST(test_df, train=False, transform=transforms.Compose([transforms.ToTensor()]))
	train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
	test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=True)
	return train_loader, test_loader

if __name__ == '__main__':
	get_dataloader()