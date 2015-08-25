#This program imports a list of popular US websites, cleans the imported list and then pulls
#data for 400 websites(only 400 due to api data restrictions) and places the data in a 
#SQLite database.  This data will eventually have PCA performed on it and then be used in
#a regression

import requests
import pandas as pd
import sqlite3 as lite
#========================================================================================
#import list websites from text file
with open("/Users/jacobbaumbach/Desktop/PCA/list.txt") as f:
    content=f.readlines()
#cleans list
content=[content[i].rstrip('\n').rstrip('\r') for i in range(0,len(content))]

apikey=#ENTER compete API Key here

#=====================================================================================
#data names
#uv=unique visitors
#vis=number of separate visits made to a domain by all unique visitors
#pv=The number of times a page has been loaded from a domain
#avgstay=The average number of seconds that a visit lasts
#vpp=The average number of times each unique visitor visits the domain
#ppv=The average number of pages displayed during a visit
#att=The percent of total minutes spent by all US users on the Internet that were spent on this domain
#reachd=The percent of all US users on the Internet that had at least one visit to this domain by day
#========================================================================================
#open database
con=lite.connect('competedata.db')
cur=con.cursor()

#loop through each website
for i in content[0:400]:
	url='https://apps.compete.com/sites/'+i+'/trended/search/?apikey='+apikey+'&metrics=uv,vis,pv,avgstay,vpp,ppv,att&latest=1'#generate url that'll be used in the api for i, the given website
	r=requests.get(url)#capture data in json fomat for the given website
	#check if first iteration of loop
	if i==content[0]:
		with con:
			cur.execute("DROP TABLE IF EXISTS competedata;")#delete table if already exists
			cur.execute("CREATE TABLE competedata(url TEXT PRIMARY KEY, uv REAL, vis REAL, pv REAL, avgstay REAL, vpp REAL, ppv REAL, att REAL);")#create table

	with con:
		#insert data into table
		cur.execute("INSERT INTO competedata(url,uv,vis,pv,avgstay,vpp,ppv,att) VALUES (?,?,?,?,?,?,?,?)",(i,float(r.json()['data']['trends']['uv'][0]['value']),float(r.json()['data']['trends']['vis'][0]['value']),float(r.json()['data']['trends']['pv'][0]['value']),float(r.json()['data']['trends']['avgstay'][0]['value']),float(r.json()['data']['trends']['vpp'][0]['value']),float(r.json()['data']['trends']['ppv'][0]['value']),float(r.json()['data']['trends']['att'][0]['value']),))#!!!!!!!!!!!!!!!!!!!)

con.close()#clost database



		