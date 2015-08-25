#this is a collection of functions used to generate the Principal Components of 
#a given set of data.  I have tried to program each step using strictly functions
#I created in an attempt to prove my understanding of PCA

#This group of functions are utilized by the main function
#PCA which accepts the set of data(a ppandas dataframe) and an error tolerance for the QR algorithm.*
#The PCA function generates sorted eigenvalues, accompanying eigenvectors and a 
#dictionary displaying the relative imortance of each principal component.**
#This function first converts the dataframe to a dictionary,then demeans the data, then calculates the covariance matrix,
#then the eigenvalues and eigenvectors are generated for the covariance matrix
#via the QR algorithm, which utilizes the Householder Transformation, and then
#finally sorts the eigenvalues and their corresponding eigenvectors in descending
#order

#*The lower the tolerance(bounded by zero) the closer the generated eigenvalues
#and vectors will be to the true ones but the smaller the tolerance the longer the
#compile time will be.  It is recommended that 0.1 is used as the tolerance.
#** The eigenvectors are given in a dictionary where the key for a given eigenvector
#represents the position in the list of eigenvalues where the corresponding eigenvalue
#can be found
#=================================================================================
import math
import copy
import pandas as pd
#==================================================================================
#converts the dataframe to a dictionary
def pd2dict(df):
	dct=df.to_dict('list')
	ky=dct.keys()
	fnldict={i:df[ky[i]] for i in range(0,len(ky))}
	return fnldict

#===============================================================================
#This block of functions is primarily used to demean the data and then calculate
#the covariance matrix

#perfoms the dot product between two vectors
def dp(X,Y):
	return sum([float(x)*float(y) for (x,y) in zip(X,Y)])
#takes average of a list
def avg(X):
	return (float(sum(X))/len(X))
#subracts the mean of each element of a list
def dmean(d):
	mn=avg(d)
	x=[i-mn for i in d]
	return x
#subtracts the mean of each column from each element in the column
def dmeanMAT(d):
	col = len(d)
	dDmean={i:[0.0 for j in range(0,len(d[0]))] for i in range(0,col)}
	for i in range(0,col):
		dDmean[i]=dmean(d[i])
	return dDmean
#calculates the covariance between two lists
def indcov(X,Y):
	XDmean=dmean(X)
	YDmean=dmean(Y)
	cv=(1/(float(len(X))-1.0))*dp(XDmean,YDmean)
	return cv
#calculates the covariances between each column of the matrix of data d
def covMat(d):
	col=len(d)
	cv={i:[0.0 for j in range(0,col)] for i in range(0,col)}
	k=0
	for i in range(0, col):
		for j in range(k,col):
			cv[i][j]=indcov(d[i],d[j])
			cv[j][i]=cv[i][j]
		k+=1
	return cv
#======================================================================================
#This block is used to perfom QR factorization via Householder Transormation which produces an
#orthonormal matrix.  This orthnomrmal matrix is used in the QR algorithm to
#obtain the eigenvalues and eigenvectors of the covariance matix

#classic sign function
def sgn(x):
	if x<0:
		return -1.0
	elif x==0:
		return 0.0
	else:
		return 1.0

#calculate the norm of a vector
def vectDist(X):
	return math.sqrt(sum([float(x)*float(x) for x in X]))

#normalizes a vector
def norm(X):
	denom=vectDist(X)
	nrm=[x/denom for x in X]
	return nrm

#mulplies a vector by a scalar
def sclvec(a,b):
	c=[a*x for x in b]
	return c

#adds two vectors together
def ad2vec(a,b):
	c=[a[i]+b[i] for i in range(0,len(b))]
	return c

#subtracts two matricies
def matSub(a,b):
	ky=a.keys()
	c={i:[a[i][j]-b[i][j] for j in range(0,len(a[ky[0]]))] for i in ky}
	return c

#multiplies a*b (in that order) where a is a matrix and b is a vector
def matvec(a,b):
	ky=a.keys()
	c={i:[0.0 for j in range(0,len(a[ky[0]]))] for i in ky}
	for i in range(0,len(a[ky[0]])):
		cnt=0
		for j in ky:
			c[j][i]+=a[j][i]*b[cnt]
			cnt+=1
	return c

#multiplies a column vector by a row vector to produce a matrix
def vec2mat(a,b):
	c={i:[0.0 for j in range(0,len(a))] for i in range(0,len(b))}
	for i in range(0,len(b)):
		for j in range(0,len(a)):
			c[i][j]=a[i]*b[j]
	return c

#subracts a reflection from a subset of the identity matrix
def pCreate(v,strt,stp):
	c={i:[0.0 for j in range(0,stp)] for i in range(0,stp)}
	for i in range(0,stp):
		c[i][i]=1
	for i in range(strt,stp):
		for j in range(strt,stp):
			c[i][j]=c[i][j]-2*v[i-strt][j-strt]
	return c

#multiplies two matricies X*Y, in that order, since the matricies are less than 50 rows and 
#50 columns I decided to use the nested loop instead of Strassen's algorithm
def matMult(X,Y):#0 list row lists, 1 list col list
	c={j:[0.0 for i in range(0,len(X[0]))] for j in range(0,len(Y))}
	for i in range(0,len(X[0])):
		for j in range(0,len(Y)):
			for k in range(0,len(X[0])):
				c[j][i]+=X[k][i]*Y[j][k]
	return c

#perfoms the Householder transformation uses reflections on the inputed matrix to create an 
#orthonormal matrix Q.  This function could also be used to produce R, but my QR algorithm
#uses the fact that R=(Q^-1)*A making calculating R pointless for this function
def HT2(d):
	col=len(d)
	row=len(d[0])
	r={i:[d[i][j]for j in range(0,row)] for i in range(0,col)}
	Q={i:[0.0 for j in range(0,row)] for i in range(0,col)}
	for i in range(0,col):
		Q[i][i]=1.0

	for i in range(0,col):
		x=[r[i][j] for j in range(i,row)]
		vd=vectDist(x)
		s=sgn(x[0])
		e=[0.0 for j in range(i,row)]
		e[0]=1
		v=ad2vec(sclvec(-s*vd,e),sclvec(-1,x))
		if vectDist(v)>0.0:
			v=norm(v)
			V=vec2mat(v,v)
			P=pCreate(V,i,col)
			Q=matMult(Q,P)
	return Q
#=======================================================================================
#This block is used to perform the QR algorithm which produces the eigenvalues and eigenvectors
#for the covariance matrix

#tranposes a matrix
def tranps(X):
	Y={j:[0.0 for i in range(0,len(X[0]))] for j in range(0, len(X))}
	for i in range(0,len(X)):
		for j in range(0,len(X[0])):
			Y[i][j]=X[j][i]
	return Y

#finds the diagonal of a matrix
def diag(X):
	Y=[X[i][i] for i in range(0,len(X))]
	return Y

#calculates the "error" generated by each step of the QR algorithm by subtracting the 
#two diagonals(the diagonal will be the eventaul eigenvalues) and calculating the norm
#of the difference
def generror(d1,d2):
	Y=[i-j for (i,j) in zip(d1,d2)]
	return vectDist(Y)

#the QR algorithm is used to generate the eigenvalues and eigenvectors of the covariance matrix
#the QR factorizations uses Householder Transformation and the algorithm also relies on
#R=(Q^-1)*A as well as the property that for orthonormal matricies that Q^-1=Q^T
def QRalgo(d,tau):
	dd={i:[d[i][j] for j in range(len(d))]for i in range(len(d))}
	Q={i:[0.0 for j in range(len(d))]for i in range(len(d))}
	diag1=diag(d)
	err=tau+1.0
	cnt=0
	while err>tau:
		Q=HT2(dd)
		dd=matMult(matMult(tranps(Q),dd),Q)
		diag2=diag(dd)
		err=generror(diag2,diag1)
		diag1=diag2
		if cnt==0:
			QQ=Q	
		else:
			QQ=matMult(QQ,Q)
		cnt+=1
	return diag1, QQ

#==========================================================================================
#sorts output and generate summary dictionary

#sorts the eigenvalues and corresponding eigenvectors in descending order
def srt(e,ev):
	srtkey=sorted(range(len(e)),key= lambda k: e[k], reverse=True)
	e.sort(reverse=True)
	evsrt=[[] for i in range(0,len(e))]
	for i in srtkey:
		evsrt[i]=ev[srtkey[i]]
	return e, evsrt

#priduces a summary dictionary for eigenvalues
def sumdict(e):
	fnl={"eigenvalues":[], "proportion":[],"cummulative":[]}
	denom=sum(e)
	cumm=0.0
	for i in range(0,len(e)):
		fnl["eigenvalues"].append(e[i])
		fnl["proportion"].append(e[i]/denom)
		cumm+=fnl["proportion"][i]
		fnl["cummulative"].append(cumm)
	return fnl


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#main function

def PCA(df,tau):
	d=pd2dict(df)
	ddmean=dmeanMAT(d)#demean the data
	cv=covMat(ddmean)#calculate covariance matrix
	cveig, cveigvec=QRalgo(cv,tau)#calculate eigen values and vectors
	cveigSort,cveigvecSort=srt(cveig, cveigvec)#sort eigenvalues in descending order and their corresponding eigenvectors
	summary=sumdict(cveigSort)#produces summary dictionary
	return cveigSort, cveigvecSort, summary