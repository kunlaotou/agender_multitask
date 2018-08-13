# -*- coding:UTF-8 -*
import torch
from torch import nn
from torch.autograd import Variable
from torchvision import models



from dot import make_dot

def tensor_hook(grad):
	print('tensor hook')
	print('grad:', grad)
	return grad


class MyResNet50(nn.Module):
	def __init__(self):
		super(MyResNet50,self).__init__()
		self.model = models.resnet50(pretrained=True)
		self.model = nn.Sequential(*list(self.model.children())[:-1])
		self.fc_gender = nn.Linear(2048, 2)
		self.fc_age = nn.Linear(2048, 4)
		self.fc_Tshirt = nn.Linear(2048,2)
		self.fc_jacket = nn.Linear(2048,2)
		self.fc_skirt = nn.Linear(2048,2)
		self.fc_trousers = nn.Linear(2048,2)
		
	def forward(self, x):
		
		x = self.model(x)
		x = x.view(x.size(0), -1)
		gender = self.fc_gender(x)
		age = self.fc_age(x)
		img_Tshirt_label = self.fc_Tshirt(x)
		img_jacket_label = self.fc_jacket(x)
		img_skirt_label = self.fc_skirt(x)
		img_trousers_label = self.fc_trousers(x)
		
		return gender, age, img_Tshirt_label, img_jacket_label, img_skirt_label, img_trousers_label
	
	def my_hook(self, module, grad_input, grad_output):
		# print('doing my_hook')
		# print('original grad:', grad_input)
		# print('original outgrad:', grad_output)
		# grad_input = grad_input[0]*self.input   # 这里把hook函数内对grad_input的操作进行了注释，
		# grad_input = tuple([grad_input])        # 返回的grad_input必须是tuple，所以我们进行了tuple包装。
		# print('now grad:', grad_input)
		return grad_input

if __name__ == '__main__':
	model = MyResNet50().cuda()
	input = Variable(torch.randn([1,3,224,224])).cuda()
	model.register_backward_hook(model.my_hook)
	# input.register_hook(tensor_hook)
	gender,age = model(input)
	print('input.grad:', input.grad)
	for param in model.parameters():
		print('{}:grad->{}'.format(param, param.grad))
	# make_dot(gender, params=dict(model.named_parameters())).render('prelu_resnet50_gender', view=True)
