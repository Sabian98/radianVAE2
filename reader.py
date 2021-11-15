
import matplotlib.pyplot as plt
import numpy as np

def plotter(final_arr,range_name,fl,tp):
	
	plt.figure(dpi=300)
	# print(final_arr[0])
	plt.plot(['10','20','30','50'],final_arr[0],'go-',linewidth=3,label='GraphVAE')
	plt.plot(['10','20','30','50'],final_arr[1],'ro-',linewidth=3,label='WGAN')
	plt.plot(['10','20','30','50'],final_arr[2],'bo-',linewidth=3,label='GraphRNN')
	plt.xlabel("Epochs")
	plt.ylabel(tp+"(Input, Generated)")
	plt.legend()
	plt.savefig("Figure 6/"+range_name+"Range"+str(fl)+str(tp)+".png",bbox_inches="tight",pad_inches = 0)
	plt.close()



fl=128
tp="BD"
rng="Short"
arr1=[0.018,
0.017,
0.018,
0.17


]
arr2=[0.56,
0.21,
0.07,
0.14


]
arr3=[
0.09,


0.07,


0.07,


0.07




]



plotter([arr1,arr2,arr3],rng,fl,tp)
'''
model_type="graphVAE"
type="long"
vertexes=200
epoch=50
x = np.loadtxt("dist.txt")
y=np.loadtxt("original_distances.txt")
num_bins = 100
n, bins, patches = plt.hist(x, num_bins, facecolor='blue', alpha=0.5, label='Generated')
n2, bins2, patches2 = plt.hist(y, num_bins, facecolor='red', alpha=0.5, label='Input')
plt.legend()
plt.savefig("variable histo/"+str(model_type)+"_"+str(type)+"_"+str(vertexes)+"_"+str(epoch)+".png",dpi=300,bbox_inches="tight",pad_inches = 0)
'''





