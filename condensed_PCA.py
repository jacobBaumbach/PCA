#NOTE FOR GALVANIZE: I have converted the original program into a class.  All the methods are
#the same and produce the same output it is just now in the form of a class

#This class performs the PCA on a pandas dataframe
#The goal of this class was to try to minimize the number of lines of code without
#explicitly using a built in PCA function like the one in sci-kit learn.
#The procedure is essentially identical to myOwnPCA class
#The dataframe is first demeaned, then the covariance matrix is calculated, then the 
#dataframe is converted to a numpy matrix so I can use numpy's built in eig function,
#which returns the eigenvalues and eigenvectors for the covariance matrix.
#The eigenvalues and corresponding eigenvectors are sorted in descending order and then
#returned.
#===================================================================================
import pandas as pd
import numpy as np

class condPCA:

	def __init__(self,d):
		self.d=d


	#demeans the dataframe
	def demean(self):
		ky=self.d.keys().tolist()
		for i in ky:
			self.d[i]=self.d[i].map(lambda x: x-self.d[i].mean())
		return self.d
	#sorts the eigenvalues and corresponding eigenvectors in descending order
	def eigsrt(self,evl,evec):
		indx=evl.argsort()[::-1]
		vl=evl[indx]
		vec=evec[:,indx]
		return vl,vec
	#main method
	def PCAcond(self):
		dd=self.demean()#demean data
		cv=dd.cov()#covariance matrix
		mat=cv.as_matrix()#convert dataframe to numpy matrix
		eigvl,eigvec=np.linalg.eig(mat)#calculate eigenvalues and eigenvectors
		eigvlsrt, eigvecsrt=self.eigsrt(eigvl,eigvec)#sort eigenvalues and corresponding eigenvectors
		return eigvlsrt,eigvecsrt

