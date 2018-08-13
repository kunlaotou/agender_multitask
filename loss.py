from __future__ import print_function

import torch.nn as nn
import torch.nn.functional as F


class AGLoss(nn.Module):
	def __init__(self):
		super(AGLoss, self).__init__()
	
	def forward(self, gender_preds, gender_targets, age_preds, age_targets,Tshirt_preds,Tshirt_targets,jacket_preds,img_jacket_target,skirt_preds,img_skirt_target,trousers_preds,img_trousers_target):
		
		age_loss = F.cross_entropy(gender_preds, gender_targets)
		gender_loss = F.cross_entropy(age_preds, age_targets)

		
		Tshirt_loss = F.cross_entropy(Tshirt_preds, Tshirt_targets)

		jacket_loss = F.cross_entropy(jacket_preds, img_jacket_target)
		skirt_loss = F.cross_entropy(skirt_preds, img_skirt_target)
		trousers_loss = F.cross_entropy(trousers_preds, img_trousers_target)
		
		return gender_loss, age_loss, Tshirt_loss, jacket_loss, skirt_loss, trousers_loss
