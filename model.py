import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.stats import gaussian_kde
from scipy.interpolate import griddata
from sklearn.linear_model import LinearRegression
import pandas
from getEvent import *
from PIL import Image

# ne = np.array(Image.open('1.tif'))
# nw = np.array(Image.open('2.tif'))
# se = np.array(Image.open('3.tif'))
# sw = np.array(Image.open('4.tif'))
# print(ne.shape)
# print(nw.shape)
# print(se.shape)
# print(sw.shape)
# print(sw[0,0])
#[ -1.70071534e+12   8.50358942e+11   8.50356394e+11   0.00000000e+00
#   8.30475438e-01]
#19.6381824491
f = pandas.read_csv("data/2015_lite.txt",'|')
f = f.as_matrix(columns=f.columns[[-10,-3,-2]])
f = f[f[:,1] != 'Unknown']
f = f[f[:,0] != ""]
f = f[~np.isnan(f[:,1:3].astype('float32')).any(axis=1)]
for i in range(f.shape[0]):
	f[i,0] = getDate(f[i,0])
f = f.astype('float32')

ft = pandas.read_csv("data/2014_lite.txt",'|')
ft = ft.as_matrix(columns=ft.columns[[-10,-3,-2]])
ft = ft[ft[:,1] != 'Unknown']
ft = ft[ft[:,0] != ""]
ft = ft[~np.isnan(ft[:,1:3].astype('float32')).any(axis=1)]
for i in range(ft.shape[0]):
	ft[i,0] = getDate(ft[i,0])
ft = ft.astype('float32')

maxLat = np.max(f[:,1])
minLat = np.min(f[:,1])
maxLon = np.max(f[:,2])
minLon = np.min(f[:,2])
print(maxLat)
print(minLat)
print(maxLon)
print(minLon)
exit()

gauges = [(-95.178333,29.517222),(-95.289167,29.674167),(-95.396944,29.775),(-95.093611,29.876111),(-95.141111,29.916111),(-95.258333,30.026389),(-95.124167,30.145278),(-95.436111,30.110278),(-95.645833,30.119722),(-95.428611,30.035556),(-95.598333,29.973333),(-95.808056,29.95),(-95.233056,29.836944),(-95.417778,29.956667),(-95.408333,29.76),(-95.523333,29.746667)]
locs = [120,340,520,720,750,760,790,1050,1070,1120,1160,1180,1620,1660,2240,2260]

radius = 0.12
num = 25

# rainfall = []
# for loc in locs:
# 	rainfall.append(np.genfromtxt(str(loc)+"_14.csv"))
# rainfall = np.asarray(rainfall)

# features = []
# values = []
# for i in range(31):
# 	print(i)
# 	X = np.linspace(minLat,maxLat,num)
# 	Y = np.linspace(minLon,maxLon,num)
# 	density = np.zeros((num,num))
# 	rdensity = np.zeros((num,num))
# 	pred = np.zeros((num,num))
# 	for xi in range(num):
# 		for yi in range(num):
# 			for row in ft:
# 				if (row[1] - X[xi])**2 + (row[2] - Y[yi])**2 < radius**2:
# 					if row[0] == 120 + i:
# 						density[xi,yi] += 1
# 			for j in range(len(locs)):
# 				if (gauges[j][1] - X[xi])**2 + (gauges[j][0] - Y[yi])**2 < radius**2:
# 					rdensity[xi,yi] += rainfall[j,i]
# 			cover = []
# 			if yi < num / 2 and xi < num / 2:
# 				cover = se[8112 / num * 2 * yi, 8112 / num * 2 * xi]
# 			elif yi >= num / 2 and xi < num / 2:
# 				cover = sw[4054 / num * 2 * (yi-num/2), 8112 / num * 2 * xi]
# 			elif yi < num / 2 and xi >= num / 2:
# 				cover = ne[8112 / num * 2 * yi, 8112 / num * 2 * (xi-num/2)]
# 			else:
# 				cover = nw[8112 / num * 2 * (yi-num/2), 8112 / num * 2 * (xi-num/2)]
# 			pred[xi,yi] = -1.70071534e+12*cover[0] +8.50358942e+11*cover[1] +8.50356394e+11* cover[2] - 8.30475438e-01 * rdensity[xi,yi] + 19.6381824491
# 	print(np.sum(density**2))
# 	print(np.sum((pred-density)**2)/num/num)

# exit()

rainfall = []
for loc in locs:
	rainfall.append(np.genfromtxt(str(loc)+".csv"))
rainfall = np.asarray(rainfall)

clf = LinearRegression()
features = []
values = []
for i in range(31):
	print(i)
	X = np.linspace(minLat,maxLat,num)
	Y = np.linspace(minLon,maxLon,num)
	density = np.zeros((num,num))
	rdensity = np.zeros((num,num))
	zero = True
	for xi in range(num):
		for yi in range(num):
			for row in f:
				if (row[1] - X[xi])**2 + (row[2] - Y[yi])**2 < radius**2:
					if row[0] == 120 + i:
						density[xi,yi] += 1
						zero = False
			for j in range(len(locs)):
				if (gauges[j][1] - X[xi])**2 + (gauges[j][0] - Y[yi])**2 < radius**2:
					rdensity[xi,yi] += rainfall[j,i]
					if rainfall[j,i] != 0:
						zero = False
			if density[xi,yi] != 0 and rdensity[xi,yi] != 0:
				cover = []
				if yi < num / 2 and xi < num / 2:
					cover = se[8112 / num * 2 * yi, 8112 / num * 2 * xi]
				elif yi >= num / 2 and xi < num / 2:
					cover = sw[4054 / num * 2 * (yi-num/2), 8112 / num * 2 * xi]
				elif yi < num / 2 and xi >= num / 2:
					cover = ne[8112 / num * 2 * yi, 8112 / num * 2 * (xi-num/2)]
				else:
					cover = nw[8112 / num * 2 * (yi-num/2), 8112 / num * 2 * (xi-num/2)]
				features.append(np.asarray([cover[0]/255.0,cover[1]/255.0,cover[2]/255.0,cover[3]/255.0,rdensity[xi,yi]]))
				values.append(density[xi,yi])
	if not zero:
		print("plot: "+str(i))
		plt.figure()
		plt.imshow(density,extent=(minLat,maxLat,minLon,maxLon),cmap=cm.gist_rainbow)
		plt.figure()
		plt.imshow(rdensity,extent=(minLat,maxLat,minLon,maxLon),cmap=cm.gist_rainbow)
print(features)
print("----------")
print(values)
clf.fit(np.asarray(features),np.asarray(values))
plt.show()
print(clf.coef_)
print(clf.intercept_)
exit()


X = np.linspace(minLat,maxLat,num)
Y = np.linspace(minLon,maxLon,num)
# density = [np.zeros((num,num)),np.zeros((num,num)),np.zeros((num,num)),np.zeros((num,num)),np.zeros((num,num)),np.zeros((num,num))]
density = np.zeros((num,num))
for xi in range(num):
	print(xi)
	for yi in range(num):
		for row in f:
			if (row[1] - X[xi])**2 + (row[2] - Y[yi])**2 < radius**2:
				if row[0] == 145:
					density[xi,yi] += 1
				# if row[0] < 60:
				# 	density[0][xi,yi] += 1
				# elif row[0] < 121:
				# 	density[1][xi,yi] += 1
				# elif row[0] < 182:
				# 	density[2][xi,yi] += 1
				# elif row[0] < 244:
				# 	density[3][xi,yi] += 1
				# elif row[0] < 305:
				# 	density[4][xi,yi] += 1
				# else:
				# 	density[5][xi,yi] += 1

# wrong = 0
# for row in ft:
# 	x = int((row[1] - minLat) / (maxLat - minLat) * num)
# 	if x < 0:
# 		x = 0
# 	if x >= num:
# 		x = num - 1
# 	y = int((row[2] - minLon) / (maxLon - minLon) * num)
# 	if y < 0:
# 		y = 0
# 	if y >= num:
# 		y = num - 1
# 	if row[0] < 60:
# 		if density[0][x,y] <= np.mean(np.mean(density[0])):
# 			wrong += 1
# 	elif row[0] < 121:
# 		if density[1][x,y] <= np.mean(np.mean(density[0])):
# 			wrong += 1
# 	elif row[0] < 182:
# 		if density[2][x,y] <= np.mean(np.mean(density[0])):
# 			wrong += 1
# 	elif row[0] < 244:
# 		if density[3][x,y] <= np.mean(np.mean(density[0])):
# 			wrong += 1
# 	elif row[0] < 305:
# 		if density[4][x,y] <= np.mean(np.mean(density[0])):
# 			wrong += 1
# 	else:
# 		if density[5][x,y] <= np.mean(np.mean(density[0])):
# 			wrong += 1

# print(1.0 * wrong / ft.shape[0])

print(X[0])
print(X[1])

# for i in range(6):
# 	plt.figure()
# 	plt.imshow(density[i],extent=(minLat,maxLat,minLon,maxLon),cmap=cm.gist_rainbow)
plt.figure()
plt.imshow(density,extent=(minLat,maxLat,minLon,maxLon),cmap=cm.gist_rainbow)
plt.show()