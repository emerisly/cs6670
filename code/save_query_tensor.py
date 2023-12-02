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

save_dir = "/mnt/edward/data2/emeraldliu/datas/computer_vision/cs6670/code/data/tensor/"

file = "dev"

data_json = json.load(open("/mnt/edward/data2/emeraldliu/datas/computer_vision/cs6670/code/data/"+file+".json"))

device = "cuda:2" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
print("loading success")

dim = len(data_json)
query_tensors = []
for data in tqdm(data_json):
	query = clip.tokenize([data["query"]]).to(device)
	with torch.no_grad():
		query_feature = model.encode_text(query)
		query_feature /= query_feature.norm(dim=-1, keepdim=True)

	# print(type(query_feature),query_feature.shape)
	query_tensors.append(query_feature)


query_tensors = torch.stack(query_tensors)
query_tensors = torch.squeeze(query_tensors)

print("query tensor",query_tensors.shape)

query_tensors = query_tensors.reshape(dim,1,512)
var_path = os.path.join(save_dir+"/queries_"+file+".npy")
np_var = query_tensors.data.cpu().numpy()
np.save(var_path,np_var)

# save_dir = "/mnt/edward/data2/emeraldliu/datas/computer_vision/cs6670/code/data/tensor/"
# query_tensors = np.load(save_dir+"/test_queries.npy")
# query_np = np.asarray(query_tensors)
# query_tensors_load = torch.from_numpy(query_np)
# query_tensors_load = torch.squeeze(query_tensors_load)
# query_tensors_load = query_tensors_load.to(device)
# print("query tensor",query_tensors_load.shape)
# print(query_tensors_load[0])