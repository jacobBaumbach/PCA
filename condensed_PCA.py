#This program performs the PCA on a pandas dataframe
#The goal of this program was to try to minimize the number of lines of code without
#explicitly using a built in PCA function like the one in sci-kit learn.
#The procedure is essentially identical to PCA_func_JWB_version
#The dataframe is first demeaned, then the covariance matrix is calculated, then the 
#dataframe is converted to a numpy matrix so I can use numpy's built in eig function,
#which returns the eigenvalues and eigenvectors for the covariance matrix.
#The eigenvalues and corresponding eigenvectors are sorted in descending order and then
#returned.

import pandas as pd
import numpy as np

#demeans the dataframe
def demean(df):
	ky=df.keys().tolist()
	for i in ky:
		df[i]=df[i].map(lambda x: x-df[i].mean())
	return df
#sorts the eigenvalues and corresponding eigenvectors in descending order
def eigsrt(evl,evec):
	indx=evl.argsort()[::-1]
	vl=evl[indx]
	vec=evec[:,indx]
	return vl,vec
#main func
def PCAcond(df):
	d=demean(df)#demean data
	cv=d.cov()#covariance matrix
	mat=cv.as_matrix()#convert dataframe to numpy matrix
	eigvl,eigvec=np.linalg.eig(mat)#calculate eigenvalues and eigenvectors
	eigvlsrt, eigvecsrt=eigsrt(eigvl,eigvec)#sort eigenvalues and corresponding eigenvectors
	return eigvlsrt,eigvecsrt

