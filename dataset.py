# -*- coding:UTF-8 -*-
from PIL import Image
from torch.utils.data import Dataset


def default_loader(path):
    try:
        img = Image.open(path)
        return img.convert('RGB')
    except:
        print("Cannot read image: {}".format(path))

class PETAData(Dataset):
	
	def __init__(self, txt_path, dataset='', data_transforms=None, loader=default_loader):
		super(PETAData, self).__init__()
		with open(txt_path) as input_file:
			lines = input_file.readlines()
			
			self.img_name = [line.split(",")[0] for line in lines]
			self.img_gender_label = [int(line.split(",")[1]) for line in lines]
			self.img_age_label = [int(line.split(",")[2]) for line in lines]
			self.img_Tshirt_label = [int(line.split(",")[3]) for line in lines]
			self.img_jacket_label = [int(line.split(",")[4]) for line in lines]
			self.img_skirt_label = [int(line.split(",")[5]) for line in lines]
			self.img_trousers_label = [int(line.split(",")[6]) for line in lines]
		
		self.data_transforms = data_transforms
		self.dataset = dataset
		self.loader = loader
	
	def __len__(self):
		return len(self.img_name)
	
	def __getitem__(self, index):
		img_name = self.img_name[index]
		img_gender_label = self.img_gender_label[index]
		img_age_label = self.img_age_label[index]
		img_Tshirt_label = self.img_Tshirt_label[index]
		img_jacket_label = self.img_jacket_label[index]
		img_skirt_label = self.img_skirt_label[index]
		img_trousers_label = self.img_trousers_label[index]
		
		img = self.loader(img_name)  
		
		if self.data_transforms is not None:
			try:
				img = self.data_transforms[self.dataset](img)
			except BaseException:
				print("Cannot transform image: {}".format(img))
		return img, img_gender_label, img_age_label, img_Tshirt_label, img_jacket_label, img_skirt_label, img_trousers_label
