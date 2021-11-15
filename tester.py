import numpy as np
'''
def test(path):

	# data = np.load(path,allow_pickle='TRUE').item()
	rad_data=np.loadtxt(path)
	if (len(rad_data))>=16:

		final_arr=[]
		for elem in rad_data[:16]:
		
		
			qrt=[]
			for e in elem:

				if e>=0 and e<=90:
					qrt.append(0)
				elif e>90 and e<=180:
					qrt.append(1)
				elif e>=-90 and e<0:
					qrt.append(2)
				elif e>=-180 and e<=-90:
					qrt.append(3)
			ind=(qrt[0]*4)+qrt[1]
			final_arr.append(ind)
		return final_arr
		

	
# def hot_encode(a):
# 	a = np.array(a)
# 	b = np.zeros((a.size, 16))
# 	b[np.arange(a.size),a] = 1
# 	return b

file=open("/scratch/trahman2/list.txt")
fr=file.readlines()


n_arr=[]
for elem in fr:
	name=elem.strip("\n")
	n_arr.append(name)
	

dict1=np.load("/scratch/trahman2/my_file_16.npy",allow_pickle='TRUE').item()
l=list(dict1.keys())
l_arr=[]
for elem in l:
	l_arr.append(str(elem))

hot_dict={}

for elem in n_arr:
	if elem[:4] in l_arr:

		hot_coded=np.array(test("/scratch/trahman2/maps/dihedral_coords/"+elem))

		if (hot_coded.size)==16:

			hot_dict[elem[:4]]=hot_coded

np.save("/scratch/trahman2/dihedral_16.npy",hot_dict)

'''

dict1 = np.load("/scratch/trahman2/my_file_16.npy",allow_pickle='TRUE').item()
dict2 = np.load("/scratch/trahman2/dihedral_16.npy",allow_pickle='TRUE').item()



for key in dict1:
	if key in dict2:
		if (dict2[key].size!=16):
			print (key)
		dict1[key].append(dict2[key])

arr=[]
for key in dict1:
	if (len(dict1[key]))!=3:
		arr.append(key)
	
[dict1.pop(key) for key in arr]

np.save("/scratch/trahman2/modified_16.npy",dict1)



