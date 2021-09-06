import torch
from SNN import Network, init_weights
from triplet_loss import TripletLoss
from tqdm import tqdm
from dataloader import MNIST
from dataloader import get_dataloader
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

c = 0
completed = 544

torch.manual_seed(2020)
np.random.seed(2020)

def test(epoch, model, device, test_loader):
	embeds = None
	embed_labels = None
	model.eval()
	bar = tqdm(test_loader, total=len(test_loader), disable=True)
	for idx, (anchor_img, anchor_label) in enumerate(bar):
		anchor_img = anchor_img.to(device)
		anchor_out = model(anchor_img)
		if embeds is None:
			embeds = anchor_out.cpu().detach().numpy()
			embed_labels = anchor_label.cpu().detach().numpy()
		else:
			embeds = np.concatenate((embeds, anchor_out.cpu().detach().numpy()))
			embed_labels = np.concatenate((embed_labels, anchor_label.cpu().detach().numpy()))
		bar.set_description("TESTING:- "+str({"epoch": epoch+1}))
		bar.refresh()
	bar.close()
	return embeds, embed_labels

def plot_images(embeds, labels, N=5):
	global c, completed
	if c>=completed:
		colors = ["blue", "green", "orange", "red", "purple", "brown", "pink", "gray", "olive", "cyan"]
		plt.cla()
		plt.clf()
		fig = plt.figure(figsize = (10, 7))
		ax = plt.axes(projection ="3d")
		#ax.set_axis_off()
		ax.grid(False)
		_, min_num = np.unique(labels, return_counts=True)
		min_num = np.min(min_num)
		for label in range(10):
			indexes = np.argwhere(labels==label).flatten()
			labelled_embeds = embeds[indexes[:min_num]]
			ax.scatter3D(labelled_embeds.T[0], labelled_embeds.T[1], labelled_embeds.T[2], color=colors[label])
			ax.view_init(-150, (c+1)%360-180)
		plt.savefig("./images/img"+"0"*(N-len(str(c)))+str(c)+".png")
		fig.clear()
		plt.close('all')
	c+=1

def train(embedding_dims=3, epochs=10):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	model = Network(embedding_dims)
	model.apply(init_weights)
	model = torch.jit.script(model).to(device)

	optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
	criterion = torch.jit.script(TripletLoss())

	train_loader, test_loader = get_dataloader()

	for epoch in range(epochs):
		running_loss = []
		model.train()
		bar = tqdm(train_loader, total=len(train_loader))
		for idx, (anchor_img, positive_img, negative_img, anchor_label) in enumerate(bar):
			anchor_img = anchor_img.to(device)
			positive_img = positive_img.to(device)
			negative_img = negative_img.to(device)
			optimizer.zero_grad()
			anchor_out = model(anchor_img)
			positive_out = model(positive_img)
			negative_out = model(negative_img)
			loss = criterion(anchor_out, positive_out, negative_out)
			loss.backward()
			optimizer.step()
			running_loss.append(loss.cpu().detach().item())
			bar.set_description("TRAINING:- "+str({"epoch": epoch+1, "loss": round(sum(running_loss)/len(running_loss), 4)}))
			bar.refresh()
			test_embeddings, test_labels = test(epoch, model, device, test_loader)
			plot_images(test_embeddings, test_labels)
		bar.close()
	torch.save({"model_state_dict": model.state_dict(), "optimzier_state_dict": optimizer.state_dict()}, "trained_model.pth")

if __name__ == '__main__':
	train()