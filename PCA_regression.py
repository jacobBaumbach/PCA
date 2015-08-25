#This program brings all the other programs together.  The goal of this program is
#to run a regression with average stay(how many seconds someone stays on a website)
#on one PCA factor.  I chose the first PCA factor because it accounted for 0.9997344548956167% of the "variation" in the data
#I acquired this number from the summary dictionary outputted by the PCA function in PCA_func_JWB_version
#The program generates the first PCA factor three times.  Once using my explicit function,
#a second time using my condensed PCA function and a final time using the PCA function in sklearn
#the goal of this whole project has not necessarily been to generate an insightful regression, but
#rather validate my two PCA functions and utilize some common libraries and techniques used in data
#science.  I believe I accomplished this since the three regressions using separate PCA factors from each
#function come out identical. Therefore validating both of the PCA functions I created.

import PCA_func_JWB_version
import condensed_PCA
import pandas as pd 
import sqlite3 as lite
import numpy as np 
import statsmodels.api as sm
from sklearn import decomposition
#===================================================================================
#multiplies a*b (in that order) where a is a vector and b is a matrix
def matvec(a,b):
	d=b.to_dict('list')
	ky=d.keys()
	c=[0.0 for j in range(0,len(d[ky[0]]))] 
	for i in range(0,len(d[ky[0]])):
		for j in range(0,len(ky)):
			c[i]+=a[j]*d[ky[j]][i]
	return c
#===================================================================================
#connect database and place the SQL data into a dataframe
con=lite.connect('/Users/jacobbaumbach/Desktop/PCA/competedata.db')
cur=con.cursor()
df=pd.read_sql("SELECT * from competedata", con, index_col='url')

#===================================================================================
#partition data into dependent var, avgstay and the independent variables
dep='avgstay'
depend=df[dep]#our dependent variable
df.drop(dep, axis=1, inplace=True)#deletes dependent variable so our PCA is performed on only independent variables
#===================================================================================
eigvl,eigvec,summary=PCA_func_JWB_version.PCA(df,0.1)#calculates PCA using the explicit version of PCA I created

#===================================================================================
eigvlCond,eigvecCond=condensed_PCA.PCAcond(df)#calculates the PCA using the condensed version of PCA

#===================================================================================
#calculates the PCA using sklearns built in function
dfmat=df.as_matrix()
pca=decomposition.PCA(n_components=1)
indSKLrn=pca.fit_transform(dfmat)


#===================================================================================
#acquire the factors from the non sklearn PCA output
ind=matvec(eigvec[0],df)
indCond=matvec(eigvecCond[0],df)

#===================================================================================
#prep the data for the regression
y=np.matrix(depend).transpose()
x1=np.matrix(ind).transpose()
x1cond=np.matrix(indCond).transpose()
x1SKLrn=np.matrix(indSKLrn)

X1=sm.add_constant(x1)
X1COND=sm.add_constant(x1cond)
X1SKLRN=sm.add_constant(x1SKLrn)

#===================================================================================
#This section runs the regressions.  As mentioned in the preamble the regressions aren't that
#insightful, but have identical results, validating my two PCA functions

#My long form PCA func 
model=sm.OLS(y,X1)
f=model.fit()
print f.summary()

#My short form PCA func
model1=sm.OLS(y,X1COND)
ff=model1.fit()
print ff.summary()

#sklearn PCA
model2=sm.OLS(y,X1SKLRN)
fff=model2.fit()
print fff.summary()

