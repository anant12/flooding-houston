import numpy as np
from math import *

def lonlat2Dis(lat1,lat2,lon1,lon2):
	R = 6371000  #Mean earth radius
	t1 = radians(lat1)
	t2 = radians(lat2)
	dt = radians(lat2 - lat1)
	dl = radians(lon2 - lon1)
	a = sin(dt / 2) * sin(dt / 2) + cos(t1) * cos(t2) * sin(dl / 2) * sin(dl / 2)
	c = 2 * atan2(sqrt(a),sqrt(1-a))
	return R * c

def getDate(s,leap=False):
	if leap:
		months = [0,31,60,91,121,152,182,213,244,274,305,335]
	else:
		months = [0,31,29,90,120,151,181,212,243,273,304,334]
	return months[int(s[5:7])-1]+int(s[8:10])

def toDate(date,leap=False):
	if leap:
		months = [0,31,60,91,121,152,182,213,244,274,305,335]
	else:
		months = [0,31,29,90,120,151,181,212,243,273,304,334]
	for i in range(len(months)-1):
		if date <= months[i+1]:
			return str(i+1)+"."+str(date-months[i])
	return str(len(months))+"."+str(date-months[-1])

def getInfo(year):
	res = []
	with open(str(year)+"_lite.txt","r") as f:
		line = f.readline()
		line = f.readline()
		while line != "":
			infos = line.split("|")
			line = f.readline()
			date = infos[-10]
			lat = infos[-3]
			lon = infos[-2]
			if date == "" or lat == "" or lon == "" or lat == "Unknown" or lon == "Unknown":
				res.append([0,0,0])
				continue
			res.append([getDate(date,year % 4 == 0),float(lat),float(lon)])
	f.close()
	return res

def buildGraph(year,dateDiff,distDiff):
	info = getInfo(year)
	graph = {}
	for i in range(len(info)):
		graph[i] = []
		if info[i][0] == 0:
			continue
		for j in range(i + 1, len(info)):
			if info[j][0] == 0 or info[j][0] - info[i][0] > dateDiff:
				continue
			dist = lonlat2Dis(info[i][1],info[j][1],info[i][2],info[j][2])
			if dist > distDiff:
				continue
			graph[i].append(j)
	return info, graph

# info, g = buildGraph(2015,1,500)
# nextNonEmpty = 0
# while nextNonEmpty < len(g) and len(g[nextNonEmpty]) == 0:
# 	nextNonEmpty += 1
# connectedGraphs = []
# while nextNonEmpty < len(g):
# 	connected = [nextNonEmpty]
# 	bfs = [nextNonEmpty]
# 	while len(bfs) > 0:
# 		current = bfs.pop(0)
# 		for i in g[current]:
# 			if len(g[i]) == 0 or i in connected:
# 				continue
# 			connected.append(i)
# 			bfs.append(i)
# 	for i in connected:
# 		g[i] = []
# 	connectedGraphs.append(connected)
# 	nextNonEmpty += 1
# 	while nextNonEmpty < len(g) and len(g[nextNonEmpty]) == 0:
# 		nextNonEmpty += 1

# # f = open("2015_events.txt","w")
# minlat = ""
# maxlat = ""
# minlon = ""
# maxlon = ""
# f = open("2015_coordinates.txt","w")
# f.write("MinLat MaxLat MinLon MaxLon\n")
# for cg in connectedGraphs:
# 	if len(cg) < 5:
# 		continue
# 	dateStart = info[cg[0]][0]
# 	dateEnd = info[cg[-1]][0]
# 	minLat = float('inf')
# 	maxLat = float('-inf')
# 	minLon = float('inf')
# 	maxLon = float('-inf')
# 	for i in cg:
# 		if info[i][1] < minLat:
# 			minLat = info[i][1]
# 		if info[i][1] > maxLat:
# 			maxLat = info[i][1]
# 		if info[i][2] < minLon:
# 			minLon = info[i][2]
# 		if info[i][2] > maxLon:
# 			maxLon = info[i][2]
# 	minlat += str(minLat) + ","
# 	maxlat += str(maxLat) + ","
# 	minlon += str(minLon) + ","
# 	maxlon += str(maxLon) + ","
# 	f.write(str(minLat))
# 	f.write(" ")
# 	f.write(str(maxLat))
# 	f.write(" ")
# 	f.write(str(minLon))
# 	f.write(" ")
# 	f.write(str(maxLon))
# 	f.write("\n")
# print(minlat)
# print(maxlat)
# print(minlon)
# print(maxlon)
# f.close()
