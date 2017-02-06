import numpy as np
import pandas as pd
import re

data = pd.read_csv('2015.txt', sep='|')
data = data[data['SR TYPE'] == 'Flooding']
data.to_csv('2015_lite.txt', sep='|')
exit()

for i in range(2011,2016):
	data = pd.read_csv(str(i)+'.txt', sep='|')
	data = data[data['SR TYPE'] == 'Flooding']
	data.to_csv(str(i)+'_lite.txt', sep='|')


# def readData(filename, sep='|'):
# 	res = []
# 	firstLine = True
# 	with open(filename) as f:
# 		while True:
# 			line = f.readline()
# 			if line is None:
# 				break
# 			if firstLine:
# 				headers = line.split(sep)
# 				for header in headers:
# 					col = [header]
# 					res.append(col)
# 				firstLine = False
# 			cols = line.split(sep)
# 			if (len(cols) != 29):
# 				continue
# 			if cols[15] == 'Flooding':
# 				for i in range(len(cols)):
# 					res[i].append(cols[i])
# 	f.close()
# 	return res

# data2014 = readData('2014.txt')
# print(len(data2014))
# print(len(data2014[0]))
