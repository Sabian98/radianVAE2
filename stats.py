import torch
from models_vae import Generator, Discriminator, EncoderVAE
import numpy as np
from scipy.stats.stats import pearsonr,wasserstein_distance
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from  molecules_code.classes_16 import Generator as mGen16
from  molecules_code.classes_64 import Generator as mGen64
from  molecules_code.classes_128 import Generator as mGen128
import pickle
import networkx as nx
import math
# from molecules_code import classes_16
# from classes_16 import Generator




class measures:
	def __init__(self,data_path,enc_path,length,epoch,tp,model_type):
		self.contact=0
		self.g_conv_dim=[128, 256, 512]
		self.z_dim = 8
		self.ngpu=2
		self.epoch=epoch
		self.type=tp
		self.model_type=model_type
		self.vertexes=length
		self.bond_num_types=2
		self.atom_num_types=20
		self.dropout_rate=0.
		self.batch_size=64
		self.enc_path=enc_path
		self.data_path=data_path
		self.post_method="usual"
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.load_model()

		#for molecules
		self.nz=100
		self.fixed_noise = torch.randn(self.batch_size, self.nz, 1, 1, device=self.device)

	def load_model(self):
		self.data = np.load(self.data_path,allow_pickle='TRUE').item()
		print(len(self.data.values()))
		self.num_steps=len(self.data.values())//self.batch_size
		if self.model_type=="GraphVAE":

			self.vae_stats()

		if self.model_type=="GraphRNN":
			self.rnnLoader(self.vertexes,self.epoch)

	def load_original(self):

		original_arr=[]
		for item in self.data.values():
			original_arr.append(item[0])
		return original_arr


	def vae_stats(self):

		# print(self.num_steps)
		self.decoder = Generator(self.g_conv_dim, self.z_dim, self.vertexes, self.bond_num_types,
                                 self.atom_num_types, self.dropout_rate).to(self.device)

		self.decoder.load_state_dict(torch.load(self.enc_path, map_location=lambda storage, loc: storage))

		original_arr=self.load_original()

		if self.type=="long":
			original_distances=self.long_range(original_arr,self.contact)
		elif self.type=="short":
			original_distances=self.short_range(original_arr,self.contact)
		
		
		
		generated_distances=[]
		
		for _ in range(self.num_steps):
			z = self.sample_z(self.batch_size)
			z = torch.from_numpy(z).to(self.device).float()
			edges_logits, nodes_logits = self.decoder(z)
			edges, nodes = self.get_gen_mols(nodes_logits, edges_logits, self.post_method)
			edges=edges.cpu().numpy()

			if self.type=="long":
				f_distances=self.long_range(edges,self.contact)
			
				generated_distances.extend(f_distances)
			elif self.type=="short":

				f_distances=self.short_range(edges,self.contact)
			
				generated_distances.extend(f_distances)
			else:
				generated_distances.extend(self.backbone(edges,0))


		


		if (len(generated_distances)>len(original_distances)):
			generated_distances=generated_distances[:len(original_distances)]
		else:
			original_distances=original_distances[:len(generated_distances)]

		self.histo(original_distances,generated_distances)

		
	def Bhattacharyya(self,x,y):

		plt.figure(figsize=(8,4), dpi=80) 
		cnt_x = plt.hist(x, bins=20,width=0.5)
		cnt_y = plt.hist(y, bins=20)
		x_=cnt_x[0]/len(x)   # No. of points in bin divided by total no. of samples.
		y_=cnt_y[0]/len(y)    
		BC=np.sum(np.sqrt(x_*y_))
		plt.close()
		return -np.log(BC)

	def stats(self,original_distances,generated_distances):
		print("stats for epoch "+ str(self.epoch)+ "and "+str(self.vertexes)+" is")
		# print("PCC is \n")
		# print(str(pearsonr(original_distances,generated_distances))+"\n")
		print("EMD is \n")
		print(str(wasserstein_distance(original_distances,generated_distances))+"\n")
		print("BD is \n")
		print(str(self.Bhattacharyya(original_distances,generated_distances))+"\n")


	@staticmethod
	def postprocess_logits(inputs, method, temperature=1.):
		def listify(x):
 			return x if type(x) == list or type(x) == tuple else [x]

		def delistify(x):
			return x if len(x) > 1 else x[0]

		if method == 'soft_gumbel':
			softmax = [F.gumbel_softmax(e_logits.contiguous().view(-1, e_logits.size(-1))
                                        / temperature, hard=False).view(e_logits.size())
                       for e_logits in listify(inputs)]
		elif method == 'hard_gumbel':
			softmax = [F.gumbel_softmax(e_logits.contiguous().view(-1, e_logits.size(-1))
                                        / temperature, hard=True).view(e_logits.size())
                       for e_logits in listify(inputs)]
		else:
			softmax = [F.softmax(e_logits / temperature, -1)
                       for e_logits in listify(inputs)]

		return [delistify(e) for e in (softmax)]

	def short_range(self,arr,th):
		dist=[]
		for f_cont in arr:
			count=0
			for index in range(len(f_cont)):
				arr2=f_cont[index][index+2:index+5]
				
				count+=arr2[np.where(arr2 == th)].size
			dist.append(count/len(f_cont))
		return dist

	def long_range(self,arr,th):
		dist=[]
		for f_cont in arr:
			count=0
			for index in range(len(f_cont)):
				arr2=f_cont[index][index+2:]
				
				count+=arr2[np.where(arr2 == th)].size
			dist.append(count/len(f_cont))
		return dist

	def molecules_short_range(self,edges,th):

		fake_dist1=[]
		for f_cont in edges:
			count1=0
			for index in range(len(f_cont)):
				a=f_cont[index][index+2:index+5]

				ref_tens2=torch.full((len(a),1),th,dtype=torch.long)
				b=ref_tens2.detach().numpy().squeeze()

				count1+=np.sum(b>a)
			fake_dist1.append(count1)
		return fake_dist1

	def molecules_long_range(self,edges,th):

		fake_dist1=[]
		for f_cont in edges:
			count1=0
			for index in range(len(f_cont)):
				a=f_cont[index][index+2:]

				ref_tens2=torch.full((len(a),1),th,dtype=torch.long)
				b=ref_tens2.detach().numpy().squeeze()

				count1+=np.sum(b>a)
			fake_dist1.append(count1)
		return fake_dist1

	def rnnLoader(self,vertices,epoch):
		matrices=pickle.load(open("/scratch/trahman2/graphRNN/outputs/RNN"+str(vertices)+"_"+str(epoch)+".txt",'rb'))
		final_arr=[]
		for elem in matrices:
			final_arr.append(nx.to_numpy_array(elem))

		if self.type=="long" or self.type=="short":

			original_arr=self.load_original()

		if self.type=="long":
			original_distances=self.long_range(original_arr,self.contact)
			dist=self.long_range(final_arr,self.contact)
		elif self.type=="short":
			original_distances=self.short_range(original_arr,self.contact)
			dist=self.short_range(final_arr,self.contact)
		else:
			dist=self.backbone(final_arr,0)

		if self.type=="long" or self.type=="short":
		# self.stats(original_distances,dist)
			if (len(dist)>len(original_distances)):
				dist=dist[:len(original_distances)]
			else:
				original_distances=original_distances[:len(dist)]

		# print(np.mean(dist))
		self.histo(original_distances,dist)
		# np.savetxt("/scratch/trahman2/original_distances.txt",original_distances)
		# np.savetxt("/scratch/trahman2/dist.txt",dist)
			
	def  molecules_loader(self,length,epoch):
		mol=[]
		if length==16:
			mgen=mGen16
		elif length==64:
			mgen=mGen64
		else:
			mgen=mGen128
		
		stri="FL"+str(length)+"/Models for FL"+str(length)+"/wgan_FL"+str(length)+"_epochs"+str(epoch)

		netG=mgen(self.ngpu).to(self.device)
		netG.load_state_dict(torch.load("/scratch/trahman2/molecules_saved_models/"+stri+".pth",map_location=torch.device('cpu')))
		for _ in range(self.num_steps):
			edges = netG(self.fixed_noise).squeeze().cpu().detach().numpy()
			# print(edges.shape)
			# f_distances=self.molecules_short_range(edges,8)
			if self.type=="long":

				f_distances=self.molecules_long_range(edges,8)
			elif self.type=="short":
				f_distances=self.molecules_short_range(edges,8)
			else:
				f_distances=self.backbone(edges,8)
			mol.extend(f_distances)

		

		if self.type!="backbone":
			original_arr=[]
			for item in self.data.values():
				original_arr.append(item[0])
		# original_distances=self.short_range(original_arr,self.contact)
			if self.type=="long":
				original_distances=self.long_range(original_arr,self.contact)
			if self.type=="short":
				original_distances=self.short_range(original_arr,self.contact)



			if (len(mol)>len(original_distances)):
				mol=mol[:len(original_distances)]
			else:
				original_distances=original_distances[:len(mol)]

			self.histo(original_distances,mol)
		# print(mol_short)
		# print(np.mean(mol))

	def backbone(self,arr,th):
		res=[]
		for elem in arr:
			count=0
			for index in range(len(elem)-1):
				if self.model_type=="wgan":
					if elem[index][index+1]<=th:
						count+=1
				else:
					if elem[index][index+1]==th:
						count+=1

			score=count/(len(elem)-1)

			if math.isnan(score)==False :
				res.append(score)
			else:
				res.append(0)
		return res



	def get_gen_mols(self, n_hat, e_hat, method):
		(edges_hard, nodes_hard) = self.postprocess_logits((e_hat, n_hat), method)
		edges_hard, nodes_hard = torch.max(edges_hard, -1)[1], torch.max(nodes_hard, -1)[1]

		return edges_hard, nodes_hard

	def sample_z(self, batch_size):
			return np.random.normal(0, 1, size=(batch_size, self.z_dim))

	def histo(self,real,fake):
		
		# bins = np.linspace(0, 50,1)
		# plt.hist(fake, bins, alpha=0.5, label='Generated',color='r')
		# plt.hist(real, bins, alpha=0.5, label='Input',color='b')
		# plt.legend()
		num_bins=100
		n, bins, patches = plt.hist(fake, num_bins, facecolor='red', alpha=0.5, label='Generated')
		n2, bins2, patches2 = plt.hist(real, num_bins, facecolor='blue', alpha=0.5, label='Input')
		plt.legend()

		# xlim([0, 6]);
		plt.savefig("/scratch/trahman2/loss_figs/"+str(self.model_type)+"_"+str(self.type)+"_"+str(self.vertexes)+"_"+str(self.epoch)+".png"
			,dpi=300,bbox_inches="tight",pad_inches = 0)
		plt.close()


	# def get_sample(self):

len_arr=[200]
epochs=[50]
models=["graphRNN"]
ms_type="short"

for model_type in models:

	for length in len_arr:

		for epoch in epochs:

			ms=measures("/scratch/trahman2/variable_length.npy","/scratch/trahman2/saved_models/models_"+
				str(length)+"/"+str(epoch)+"-decoder.ckpt",length,epoch,ms_type,model_type)
			if model_type=="wgan":
				ms.molecules_loader(length,epoch)











