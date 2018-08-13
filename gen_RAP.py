# # -*- coding:UTF-8 -*-
import random
def gen_RAP_8a(RAP_IMAGE_PATH, RAP_LABEL_PATH ):
	image_name = []
	with open(RAP_IMAGE_PATH) as image:
		allimage = image.readlines()
		for line in allimage:
			line = line.strip('\n')
			line = "/home/tangwenhua/datasets/RAP_dataset/RAP_dataset/"+line
			image_name.append(line)

	label_g_a = []	
	Tshirt = []
	jacket = []
	upperSuit = []
	lowerSkirt = []
	lowerTrousers = []
	with open(RAP_label_path) as label:
		alllabel = label.readlines()
		for line in alllabel:

			temp = line.split(',')
			gender = temp[0]
			if gender == '2' :
				Tshirt.append(2)
				jacket.append(2)
				upperSuit.append(2)
				lowerSkirt.append(2)
				lowerTrousers.append(2)
				label_g_a.append([gender, age])
				continue
			
			# age_list = [int(temp[1]), int(temp[2]), int(temp[3])]
			# for index in range(len(age_list)):
			# 	if age_list[index] == '1':
			# 		age = str(index)	

			if temp[1] == '1':
				age = 0
			elif temp[2] == '1':
				age = 1
			elif temp[3] == '1':
				age = 2
			label_g_a.append([gender, age])
			
			# if temp[18] == '1' or temp[23] == '1':
			# 	Tshirt.append(1)
			# if temp[18] != '1' and temp[23] != '1':
			# 	Tshirt.append(0)
			if temp[18] == '1':
				Tshirt.append(1)
			else:
				Tshirt.append(0)

			if temp[20] == '1':
				jacket.append(1)
			else:
				jacket.append(0)


			if temp[21] == '1':
				upperSuit.append(1)
			else:
				upperSuit.append(0)

			# if temp[25] == '1' or temp[26] == '1' or temp[27] == '1':
			# 	lowerSkirt.append(1)
			# if temp[25] != '1' and temp[26] != '1' and temp[27] != '1':
			# 	lowerSkirt.append(0)
			if temp[26] == '1':
				lowerSkirt.append(1)
			else:
				lowerSkirt.append(0)

			# if temp[24] == '1' or temp[28] == '1':
			# 	lowerTrousers.append(1)
			# if temp[24] != '1' and temp[28] != '1':
			# 	lowerTrousers.append(0)
			if temp[28] == '1':
				lowerTrousers.append(1)
			else:
				lowerTrousers.append(0)
			
			
	result = []
	for i in range(len(image_name)):		
		temp = image_name[i] + " " + str(label_g_a[i][0]) + " " + str(label_g_a[i][1]) + " " + str(Tshirt[i])+ " " + str(jacket[i])+ " " + str(upperSuit[i])+ " " + str(lowerSkirt[i])+ " " + str(lowerTrousers[i])
		if temp.split(' ')[1] == '2':
			continue
		
		result.append(temp)

	random.shuffle(result)
	threshold = 0.8
	length = len(result)

	# with open('data_list/RAP_train.txt','w') as f:
	# 	for i in range(len(result)):
	# 		f.write(line[i])
	# 		f.write('\n')
	test = open('data_list/RAP_test.txt', 'w')
	wf = open('data_list/RAP_train.txt', 'w')
	for i in range(len(result)):
		item = result[i].strip()
		print(item.split(","))

		if i <= length * threshold:
			wf.write('%s,%s,%s,%s,%s,%s,%s,%s\n' % (item.split(" ")[0], item.split(" ")[1], item.split(" ")[2], item.split(" ")[3], item.split(" ")[4], item.split(" ")[5], item.split(" ")[6], item.split(" ")[7]))
		else:
			test.write('%s,%s,%s,%s,%s,%s,%s,%s\n' % (item.split(" ")[0], item.split(" ")[1], item.split(" ")[2], item.split(" ")[3], item.split(" ")[4], item.split(" ")[5], item.split(" ")[6], item.split(" ")[7]))
	wf.close()
	test.close()


if __name__ == '__main__':
	RAP_imagename_path = '/home/tangwenhua/datasets/RAP_dataset/RAP_annotation/imagesname.txt'
	RAP_label_path = '/home/tangwenhua/datasets/RAP_dataset/RAP_annotation/label.csv'
	gen_RAP_8a(RAP_imagename_path, RAP_label_path)
