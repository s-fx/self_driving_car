import os 
import glob
import cv2
import torchvision
from torch.utils.data import Dataset
import torch

class SegDataset(Dataset):
	def __init__(self, img_dir, mask_dir, transform=None):
		self.transform=transform
		self.img = sorted(glob.glob(os.path.join(img_dir, "*.png")))
		self.mask = sorted(glob.glob(os.path.join(mask_dir, "*.png")))

	def __len__(self):
		return len(self.img)

	def __getitem__(self, item):
		if torch.is_tensor(item):
			item = item.tolist()
			
			"""
			Default size 3x1164x874
			"""
			
		img = cv2.imread(self.img[item])
		mask = cv2.imread(self.mask[item], 0)		# --  0=>grayscale 
			
		if self.transform:
			norm = torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
			img = norm(self.transform(img))	
			mask = self.transform(mask).squeeze(0)
			mask[mask < 0.255] = 4.0
			mask[mask < 0.35] = 3.0
			mask[mask < 0.443] = 2.0
			mask[mask < 0.54] = 1.0	
			mask[mask < 0.7] = 0.0
		return img, mask
