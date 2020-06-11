import csv
import math as m

def getboxandalpha(file):
	#file="/home/shrey/shreythesis/data_tracking_label_2/training/label_02/0000.txt"
	f=open(file,"r")
	lines=f.readlines()
	result=[]
	for x in lines:
	    result.append(x.split(' ')[3])
	f.close()

	result=[]
	bbox=[]
	k=0
	with open(file, 'rb') as csvfile:

		spamreader = csv.reader(csvfile, delimiter=' ')
		for row in spamreader:
			#for element in row:
			if 'DontCare' in row[2]:
				continue
			else:
				result.append(row[0:10])
				#print result
				#print k
				#result[k].pop(0)
				result[k].pop(3)
				result[k].pop(3)
				#result[k].pop(1)
			k+=1
	#print result
	return result

def getspeeds(file):
	with open(file, 'rb') as csvfile:
		myCarCurrYaw=0.
		myCarCurrentX=0.
		myCarCurrentY=0.
		spamreader = csv.reader(csvfile, delimiter=' ')
		#myCarCurrYaw=[]
		k=0
		positions=[]
		mypos=[]
		for row in spamreader:
			#for element in row:
			if 'MyCar' in row[2]:
				myCarCurrYaw = float(row[5])
				myCarCurrentX = float(row[4])
				myCarCurrentY = float(row[3])
				#print (myCarCurrYaw,myCarCurrentX,myCarCurrentY)
				p=[row[2],row[0],myCarCurrentX,myCarCurrentY,myCarCurrYaw]
				mypos.append(p)

			else:
				#result.append(row[:])
				#print (myCarCurrYaw,myCarCurrentX,myCarCurrentY)
				deltaX = m.cos(myCarCurrYaw)* float(row[3])+ m.sin(myCarCurrYaw) * float(row[4])
				deltaY = m.sin(myCarCurrYaw)* float(row[3]) - m.cos(myCarCurrYaw) * float(row[4])
				positionsx = deltaX+myCarCurrentX
				positionsy = deltaY+myCarCurrentY
				positionsyaw = float(row[7])
				p=[row[2],row[0],positionsx,positionsy,positionsyaw]
				positions.append(p)
				
			#print k
			#result[k].pop(0)
			#result[k].pop(3)
			#result[k].pop(3)
			#result[k].pop(1)
		#k+=1
	speeds=[]
	for i,row in enumerate(positions):
		if i==0:
			#s=[row[0],row[1],0,0,0]
			#speeds.append(s)
			continue
		else:
			vx=(positions[i][2]-positions[i-1][2])*10
			vy=(positions[i][3]-positions[i-1][3])*10
			spee=m.sqrt((((positions[i][2]-positions[i-1][2])**2)+((positions[i][3]-positions[i-1][3])**2))*10)
			s=[row[0],row[1],vx,vy,spee]
			speeds.append(s)
	return speeds

#print(speeds)

'''
for k in range(len(result)):
	result[k].pop(0)
	result[k].pop(1)
	result[k].pop(1)
	result[k].pop(1)
	#result[k].pop(4)
	print result[k]
'''

'''
file="/home/shrey/shreythesis/data_tracking_label_2/training/label_02/0000.txt"
f=open(file,"r")
lines=f.readlines()
result=[]
for x in lines:
    result.append(x.split(' ')[3])
f.close()

result=[]
bbox=[]
k=0
with open(file, 'rb') as csvfile:

	spamreader = csv.reader(csvfile, delimiter=' ')
	for row in spamreader:
		#for element in row:
		if 'DontCare' in row[2]:
			continue
		else:
			result.append(row[0:10])
			#print result
			#print k
			#result[k].pop(0)
			result[k].pop(3)
			result[k].pop(3)
			#result[k].pop(1)
		k+=1
print result
return result
'
'''