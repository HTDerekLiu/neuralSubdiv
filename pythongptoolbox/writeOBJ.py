import numpy as np

def writeOBJ(fileName,V,F):
	f = open(fileName, 'w')
	for ii in range(V.shape[0]):
		string = 'v ' + str(V[ii,0]) + ' ' + str(V[ii,1]) + ' ' + str(V[ii,2]) + '\n'
		f.write(string)
	Ftemp = F + 1
	for ii in range(F.shape[0]):
		string = 'f ' + str(Ftemp[ii,0]) + ' ' + str(Ftemp[ii,1]) + ' ' + str(Ftemp[ii,2]) + '\n'
		f.write(string)
	f.close()