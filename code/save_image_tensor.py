import json
import torch
import clip
from PIL import Image
import collections
import operator
from collections import OrderedDict
import collections
from tqdm import tqdm
import os
import numpy as np

# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

#* do not run this file unless want to recalculate everything in document_full
# assert False

# '''

save_dir = "/mnt/edward/data2/emeraldliu/datas/model/clip/tensors/"
doc_lst = json.load(open("/mnt/edward/data2/emeraldliu/datas/json/candidate_1m.json"))
dim = len(doc_lst)

device = "cuda:2" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
print("loading success")

vn_dir = "/data2/emeraldliu/visualnews-datasets/origin/"
tara_dir = "/mnt/edward/data2/weixifeng/MMR/dataset/"

image_tensors = []

skip = 0

for data in tqdm(doc_lst):
	img_name = data["image"]
	if img_name[0] == '.':
		image_path = vn_dir+img_name
	else:
		image_path = tara_dir+img_name
	try:
		image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)		
		with torch.no_grad():
			image_feature = model.encode_image(image)
			image_feature /= image_feature.norm(dim=-1, keepdim=True).to(device)
	except:
		skip += 1
		print("skipped")
		image_feature = torch.empty(1,512).to(device)
	# print(type(image_feature),image_feature.shape)
	image_tensors.append(image_feature)

print("skip count",skip)

image_tensors = torch.stack(image_tensors)
image_tensors = torch.squeeze(image_tensors)

print("Image tensor",image_tensors.shape)
# save_dir = "/mnt/edward/data2/emeraldliu/datas/model/clip/tensors/"
image_tensors = image_tensors.reshape(dim,1,512)
var_path = os.path.join(save_dir+"/images.npy")
np_var = image_tensors.data.cpu().numpy()
np.save(var_path,np_var)
images_load = np.load(save_dir+"/images.npy")
image_np = np.asarray(images_load)
image_tensors_load = torch.from_numpy(image_np)
image_tensors_load = torch.squeeze(image_tensors_load)
image_tensors_load = image_tensors_load.to(device)
print("Image tensor",image_tensors_load.shape)