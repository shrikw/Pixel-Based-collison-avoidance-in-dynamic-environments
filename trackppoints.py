'''
Shrey Iyengar
'''

# Python 2/3 compatibility
from __future__ import print_function
import glob
import numpy as np
import cv2 as cv
#from freenect import sync_get_depth as get_depth, sync_get_video as get_video
#import video
from common import anorm2, draw_str
from time import clock
from new import callibrate
global depth, frame
global lines1
from matplotlib import pyplot as plt
from math import sqrt
import cmath
import math
from itertools import combinations
from ransacpoints import select_feature_points
from ransacpoints import ransac1
from ransacpoints import calcepipole
from ransacpoints import validate_all_points
from ransacpoints import get_outliers
from ransacpoints import getsubsets
#from ransacpoints import euclidean_distance
from ransacpoints import get_neighbors 
from ransacpoints import k_nearest_neighbors
from CalculateTTI import calculate_TTI_cvalue
from readtxt import getboxandalpha
from readtxt import getspeeds
import csv

#global x0,y0,x1,y1
#color = tuple(np.random.randint(0,255,3).tolist())
global vect
import random

img_width = 1241
img_height = 376
FPS=10
lk_params = dict( winSize  = (15, 15),
                  maxLevel = 8,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict( maxCorners = 500,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 10 )
colorcode={0:(0, 0, 0),1:(0, 255, 0),2:(0, 0, 255),3:(255, 255, 255),4:(45, 35, 85),5:(45, 35, 85),6:(89, 30, 70),7:(70, 30, 89),8:(80, 110, 90),9:(110, 90, 80),10:(90, 80, 110),11:(40, 60, 20),12:(20, 40, 60),13:(60, 40, 20)}
#F = np.matrix('''
#				1. 0. 0. 1. 0. 0.;
#				0. 1. 0. 0. 1. 0.;
#				0. 0. 1. 0. 0. 1.;
#				0. 0. 0. 1. 0. 0.;
#				0. 0. 0. 0. 1. 0.;
#				0. 0. 0. 0. 0. 1.''')
def euclidean_distance(row1, row2):
    distance = 0.0
    #print()
    for i in range(2):
        #print("asdad")
        #print(i)
        distance += (row1[0][i] - row2[0][i])**2
    return sqrt(distance)

def bounding_box_naive(points):
    """returns a list containing the bottom left and the top right 
    points in the sequence
    Here, we use min and max four times over the collection of points
    """
    bot_left_x = min(point[0,0] for point in points)
    bot_left_y = min(point[0,1] for point in points)
    top_right_x = max(point[0,0] for point in points)
    top_right_y = max(point[0,1] for point in points)

    return [(int(bot_left_x), int(bot_left_y)), (int(top_right_x), int(top_right_y))]

def random_color():
    rgbl=[255,0,0]
    random.shuffle(rgbl)
    return tuple(rgbl)

F = np.matrix('''.000025''')
F_small=np.matrix('''0.000025''')
F_large=np.matrix('''0.0000250''')
F_HUGE=np.matrix('''0.000025''')

#abcdef=np.eye(F.shape[0])
#print (abcdef.shape)

#H = np.matrix('''
#				1. 0. 0. 0. 0. 0.;
#				0. 1. 0. 0. 0. 0.;
#				0. 0. 1. 0. 0. 0.''')
H = np.matrix('''1.''')
#motion = np.matrix('0. 0. 0. 0. 0. 0.').T
motion = np.matrix('0.').T
#Q = np.matrix(np.eye(6))
Q= np.matrix('1.')
R = 0.03**2
#global x_state

#P = np.matrix(np.eye(6))*1000
P = np.matrix('0.5')

#x_state = np.matrix('0. 0. 0. 0. 0. 0.').T
x_state = np.matrix('0.').T

def kalman_update_predict(x, P, measurement, R, motion, Q, F, H):
	xyz= np.matrix(measurement).T
	
	#print("F.shape")
	#print(F.shape)
	#print("measurement shape")
	#print (xyz.shape)
	#print("X shape")
	#print (x.shape)
	#print("h shape")
	#print (H.shape)
	#HSTARX= H*x
	#print ("HSTARX shape")
	#print (HSTARX.shape)
	y = np.matrix(measurement).T - H * x
	#print("Y shape")
	#print (y.shape)
	S = H * P * H.T + R  # residual convariance
	#print("s shape")
	#print (S.shape)
	#Sinv=np.linalg.pinv(S)
	#Sinv.dropna(inplace=True)
	#K = P * H.T * Sinv    # Kalman gain
	K = P * H.T * S.I    # Kalman gain
	#print("k shape")
	#print( K)
	x = x + K*y
	#print("X shape")
	#print (x)
	#abcdef=np.eye(F.shape[0])
	#print("abcdef")
	#print (abcdef.shape)
	#I = np.matrix(np.eye(F.shape[0])) # identity matrix
	I = np.matrix('1.') # identity matrix
	#print("I shape")
	#print (I.shape)
	P = (I - K*H)*P
	#print("P shape")
	#print (P.shape)

	# PREDICT x, P based on motion
	x = F*x + motion
	#print ("x after pred")
	#print(x)
	P = F*P*F.T + Q
	return x,P
	
def kalman_predict(	x_state,P_new,F,motion,Q):
	x_state = F*x_state + motion
	P_new = F*P_new*F.T + Q
	return x_state,P_new

"""
def kalman(x, P, measurement, R, motion, Q, F, H):
	
	'''
	Parameters:
	x: initial state
	P: initial uncertainty convariance matrix
	measurement: observed position (same shape as H*x)
	R: measurement noise (same shape as H)
	motion: external motion added to state vector x
	Q: motion noise (same shape as P)
	F: next state function: x_prime = F*x
	H: measurement function: position = H*x

	Return: the updated and predicted new values for (x, P)

	See also http://en.wikipedia.org/wiki/Kalman_filter

	This version of kalman can be applied to many different situations by
	appropriately defining F and H 
	'''
	# UPDATE x, P based on measurement m    
	# distance between measured and current position-belief
	y = np.matrix(measurement).T - H * x
	S = H * P * H.T + R  # residual convariance
	K = P * H.T * S.I    # Kalman gain
	x = x + K*y
	I = np.matrix(np.eye(F.shape[0])) # identity matrix
	P = (I - K*H)*P

	# PREDICT x, P based on motion
	x = F*x + motion
	P = F*P*F.T + Q

	return x, P
"""

'''
kalman = cv.KalmanFilter(6, 6, 0)
#kalman.transitionMatrix = 1* np.eye(6)
kalman.transitionMatrix = np.array([[1., 0., 0.,0.,0.,0.], 
									[0., 1., 0.,0.,0.,0.],
									[0., 0., 1.,0.,0.,0.],
									[0., 0., 0.,1.,0.,0.],
									[0., 0., 0.,0.,1.,0.],
									[0., 0., 0.,0.,0.,1.]],np.float32)  #F
kalman.measurementMatrix = np.array([[1., 0., 0.,0.,0.,0.], 
									[0., 1., 0.,0.,0.,0.],
									[0., 0., 1.,0.,0.,0.],
									[0., 0., 0.,1.,0.,0.],
									[0., 0., 0.,0.,1.,0.],
									[0., 0., 0.,0.,0.,1.]],np.float32)                              #H

kalman.processNoiseCov = 1e-5 * np.eye(6)
kalman.measurementNoiseCov = 1e-1 * np.ones((6, 6))
kalman.errorCovPost = 1. * np.ones((6, 6))
kalman.statePost = 0.1 * np.random.randn(6, 6)

print("Measurement Matrix")
print(kalman.measurementMatrix)
print("Transition Matrix")
print(kalman.transitionMatrix)
print("MeasurementNoiseCov")
print(kalman.measurementNoiseCov)
print("ProcessNoiseCov")
print(kalman.processNoiseCov)
print("ControlMatrix")
print(kalman.controlMatrix)
'''
#x_state = np.matrix('0. 0. 0. 0. 0. 0.').T
#P = np.matrix(np.eye(6))
#ret, mtx, dist, rvecs, tvecs = callibrate()
mtx=np.array([[718.85600000,   0.,       607.1928  ],
 	 [  0.0,         718.85600000, 185.2157],
	 [  0.0,           0.,           1.0        ]],np.float32)
	 
#focal_length = (mtx[0,0]+mtx[1,1])/2
focal_length=718.856
file='/home/shrey/shreythesis/data_tracking_label_2/training/label_02/0000.txt'
bboxandalpha=getboxandalpha(file)
imagesunsorted = glob.glob('/home/shrey/shreythesis/data_tracking_image_2/training/image_02/0000/*.png')
file2="/home/shrey/shreythesis/data_tracking_label_2/training/label_02/parsedLOC_0000.txt"
speds= getspeeds(file2)
images=sorted(imagesunsorted)
#focal_length=np.float32(focal_length)
#x_pred,p_pred = kalman_predict(x_state,P_new,F,motion,Q)
#x_new,p_new= kalman_update(x_pred, p_pred, np.matrix(' 0. 0. 0.'),R,F, H)
#print (mtx)
def calc_point(angle):
	return (np.around(img_width/2 + img_width/3*cos(angle), 0).astype(int), np.around(img_height/2 - img_width/3*sin(angle), 1).astype(int))
"""
def drawlines(img1,img2,lines,pts1,pts2):
	''' img1 - image on which we draw the epilines for the points in img2
		lines - corresponding epilines '''
	r,c = img1.shape
	img1 = cv.cvtColor(img1,cv.COLOR_GRAY2BGR)
	img2 = cv.cvtColor(img2,cv.COLOR_GRAY2BGR)
	x0=0
	y0=0
	x1=0
	y1=0
	for r,pt1,pt2 in zip(lines,pts1,pts2):
		color = tuple(np.random.randint(0,255,3).tolist())
		x0,y0 = map(int, [0, -r[2]/r[1] ])
		x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
		#img1 = cv.line(img1, (x0,y0), (x1,y1), color,1)
		#img1 = cv.circle(img1,tuple(pt1),5,color,-1)
		#img2 = cv.circle(img2,tuple(pt2),5,color,-1)
	return x0,y0,x1,y1
"""

class App:
    def __init__(self, video_src):
		self.track_len = 10
		self.detect_interval = 5
		#self.detect_intervaltwo = 150
		self.tracks = []
		self.tracks2 = []
		self.tracks_first = []
		#(depth,_), (rgb,_) = get_depth(), get_video()
		#self.cam = video.create_capture(video_src)
		self.cam = cv.VideoCapture(video_src)
		self.frame_idx = 0
		self.frame_count=0
		self.axisi=0
		#self.fgbg = cv.bgsegm.createBackgroundSubtractorMOG()

    def run(self):
        #while True:

        	
			#fig = plt.figure()
			#fig.canvas.draw()

			# convert canvas to image
			#imgfig = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
			#imgfig  = imgfig.reshape(fig.canvas.get_width_height()[::-1] + (3,))

			#fig.canvas.draw()
			#imgfig = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
			
			#fig.canvas.draw()
			

			for fname in images:
				frame = cv.imread(fname)
				boundingboxesssss={}
				alpha={}
				speeds={}
				fig = plt.figure()
				fig.canvas.draw()
				imgfig = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
				imgfig  = imgfig.reshape(fig.canvas.get_width_height()[::-1] + (3,))
				#fig = plt.figure()
				#fig.canvas.draw()

				# convert canvas to image
				#imgfig = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
				#imgfig  = imgfig.reshape(fig.canvas.get_width_height()[::-1] + (3,))

				#fig.canvas.draw()
				#imgfig = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
				'''
				for k,k1 in zip(range(len(bboxandalpha)), range(len(speds)) ):
					
					#print (bboxandalpha[k][2])
					#print(int(self.frame_idx)
					if int(bboxandalpha[k][0])==self.frame_idx:
						name=bboxandalpha[k][2]
						boundingboxesssss[name]=bboxandalpha[k][4:]
						alpha[name]=bboxandalpha[k][3]
					if int(speds[k1][1])==self.frame_idx:
						namesped=speds[k][2]
						speeds[name]=speds[k][1:]
						#alpha[name]=bboxandalpha[k][3]
						#print(boundingboxesssss)
				'''
				#print (speeds)
				#print("lalal")

					
				#print(alpha)
				#print('boundingboxesssss')
				#print(boundingboxesssss)
				#(depth,_), (frame,_) = get_depth(), get_video()
				#frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
				#_ret, frame = self.cam.read()
				#fgmask = self.fgbg.apply(frame)
				frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
				vis = frame.copy()

				#print("tracksbefore")
				#print (self.tracks)
				#print(self.frame_idx )
				#print(self.tracks)
				if len(self.tracks) > 0:

					img0, img1 = self.prev_gray, frame_gray
					r,c = img0.shape
					p0orig=self.tracks

					p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
					p1, _st, _err = cv.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params) #optical flow 
					#p0r, _st, _err = cv.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
					#E, mask_essential=cv.findEssentialMat(p0, p1, mtx, cv.RANSAC, 0.999, 1.0);
					#print ("E")
					#print (E)
					#principle_x0=mtx[0,2]
					#principle_y0=mtx[1,2]
					principle_x0=607.1928
					principle_y0=185.2157
					principle_point=(principle_x0,principle_y0)
					principle_point=np.float32(principle_point)
					#print(principle_point)
					
					p1_new = p1-principle_point
					
					#p1_new=p1
					p0_new= p0-principle_point
					#kes= p1_new-p0_new
					#print(kes.shape)

					
					#print("normals")
					#print(normals[0])
					#p0_new_test= p0-principle_point
					#p0r_new=p0r-principle_point
					#print("p0_new")

					#print(p0_new)
					#p0_new=p0
					#for rowsp0,rowsp1 in zip(p0_new,p1_new):
						#distance=(rowsp0[0,0]-rowsp1[0,0])**2+(rowsp0[0,1]-rowsp1[0,1])**2
						#print (distance)
					#print("p0..................")
					#print(p0_new)
					#print("p1..................")
					#print(p1_new)
					E, mask_essential=cv.findEssentialMat(p0_new, p1_new, mtx, cv.RANSAC, 0.999, 1.0);
					#print("EEE")
					#print(E)
					pose, deltaR, t, mask_pose =cv.recoverPose(E, p1_new, p0_new);
					#print("deltaR")
					#print(deltaR)
					
					temp = np.zeros((p1_new.shape[0],1,3))
					#print(temp.shape)
					#print(p1_new.shape)
					temp[:,:,:-1] = p1_new
					temp[:,:,2]=focal_length  #????????
					p1_new=temp
					temp = np.zeros((p0_new.shape[0],1,3))
					temp[:,:,:-1] = p0_new
					temp[:,:,2]=focal_length
					p0_new=temp
					p_avg = (p1_new+p0_new)/2
					p1_m=p1_new-p_avg
					p0_m=p0_new-p_avg
					#k= p1_new-p0_new
					#print(k.shape)
					'''
					#print(p1_m)
					dim1=p1_m.shape[0]
					#dim2=p1_new.shape[1]
					sigma= np.zeros((3,3))
					for i in range(0,dim1):
						sig=np.dot(p0_m[i].T,p1_m[i])
						#print('sigmaaaaaaa...')
						#print(sig)
						sigma+=sig
					#print("sig....................")
					#print(sigma)
					U,S,V=np.linalg.svd(sigma)
					deltaR=np.dot(U,V.T)
					print("deltaR")
					print(deltaR)
					
					
					
					'''
					#F, mask_fun = cv.findFundamentalMat(p0r,p1,cv.RANSAC)
					#print (F)
					#prediction
					#mtx_T=np.transpose(mtx)
					#mtx_T_inverse=np.linalg.inv(mtx_T)
					#mtx_inverse= np.linalg.inv(mtx)
					#Fund=np.linalg.multi_dot([mtx_T_inverse,E,mtx_inverse])
					
					
					#print("t")
					#print(t.shape)
					
					dst, jacobian= cv.Rodrigues(deltaR)
					
					#print("jacobian")
					#print(jacobian)
					#print ("dst")
					#print (dst)
					
					thx=dst[0]
					thy=dst[1]
					thz=dst[2]
					#measurement=np.matrix('0. 0. 0.')
					measurementX=np.matrix('0.')
					measurementY=np.matrix('0.')
					measurementZ=np.matrix('0.')
					#measurement=np.matrix('0. 0. 0.')
					#measurement=dst
					#print(measurement)
					theta = sqrt((thx*thx) + (thy*thy) + (thz*thz))
					#measurement=measurement/theta
					#normalized vector
					if theta!=0:
						#print("................theta..................")
						#print(theta)
						vect = dst/theta
					else:
						vect = dst
					vect_dot=vect*FPS
					thx_dot=vect_dot[0]
					thy_dot=vect_dot[1]
					thz_dot=vect_dot[2]
					measurementX[0]=thx_dot
					
					#measurement[0,1]=thy
					measurementY[0]=thy_dot
					#measurement[0,2]=thz
					measurementZ[0]=thz_dot
					#print('..............................dasdada',(x_new,y_new,z_new))
					#global x_state
					#print (x_state)
					if self.frame_idx==1: #kalman filter using direction vectors as measurement input
						print ("if==1")
						x_new,Px = kalman_update_predict(x_state, P, measurementX, R, motion, Q, F_small, H)
						y_new,Py = kalman_update_predict(x_state, P, measurementY, R, motion, Q, F_small, H)
						z_new,Pz = kalman_update_predict(x_state, P, measurementZ, R, motion, Q, F_small, H)
						#print (x_new)
						#print (y_new)
						#print (z_new)
								
					else:
						x_new,Px = kalman_update_predict(x_new, Px, measurementX, R, motion, Q, F_small, H)
						y_new,Py = kalman_update_predict(y_new, Py, measurementY, R, motion, Q, F_small, H)
						z_new,Pz = kalman_update_predict(z_new, Pz, measurementZ, R, motion, Q, F_small, H)
						print ("sjnksefgesuogwedwguy;odwiysad",x_new)
					avg_distance=0.
					distan=np.zeros((1))
					#print('..............................dasdada',(x_new,y_new,z_new))
					for rowsp0,rowsp1 in zip(p0_new,p1_new):

						distance=(rowsp0[0,0]-rowsp1[0,0])**2+(rowsp0[0,1]-rowsp1[0,1])**2
						distan=np.append(distan,distance)
						avg_distance=np.average(distan)
						#print (distance)
					#for rowsp0,rowsp1 in zip(p0_new,p1_new):
						#distance=(rowsp0[0,0]-rowsp1[0,0])**2+(rowsp0[0,1]-rowsp1[0,1])**2
					if(avg_distance>3000. and avg_distance<16000.):
						#print ("using 0.5")
						if self.frame_idx==1:
							print ("if==1")
							x_new,Px = kalman_update_predict(x_state, P, measurementX, R, motion, Q, F_HUGE, H)
							y_new,Py = kalman_update_predict(x_state, P, measurementY, R, motion, Q, F_HUGE, H)
							z_new,Pz = kalman_update_predict(x_state, P, measurementZ, R, motion, Q, F_HUGE, H)
							#print (x_new)
							#print (y_new)
							#print (z_new)
							
						else:
							x_new,Px = kalman_update_predict(x_new, Px, measurementX, R, motion, Q, F_HUGE, H)
							y_new,Py = kalman_update_predict(y_new, Py, measurementY, R, motion, Q, F_HUGE, H)
							z_new,Pz = kalman_update_predict(z_new, Pz, measurementZ, R, motion, Q, F_HUGE, H)
					if(avg_distance>1000. and avg_distance<3000.):
						#print ("using 0.125")
						if self.frame_idx==1:
							print ("if==1")
							x_new,Px = kalman_update_predict(x_state, P, measurementX, R, motion, Q, F_large, H)
							y_new,Py = kalman_update_predict(x_state, P, measurementY, R, motion, Q, F_large, H)
							z_new,Pz = kalman_update_predict(x_state, P, measurementZ, R, motion, Q, F_large, H)
							#print (x_new)
							#print (y_new)
							#print (z_new)
							
						else:
							x_new,Px = kalman_update_predict(x_new, Px, measurementX, R, motion, Q, F_large, H)
							y_new,Py = kalman_update_predict(y_new, Py, measurementY, R, motion, Q, F_large, H)
							z_new,Pz = kalman_update_predict(z_new, Pz, measurementZ, R, motion, Q, F_large, H)
					if(avg_distance<1000.):
						#print ("using 0.05")
						if self.frame_idx==1:
							print ("if==1")
							x_new,Px = kalman_update_predict(x_state, P, measurementX, R, motion, Q, F_small, H)
							y_new,Py = kalman_update_predict(x_state, P, measurementY, R, motion, Q, F_small, H)
							z_new,Pz = kalman_update_predict(x_state, P, measurementZ, R, motion, Q, F_small, H)
							
							#print (y_new)
							#print (z_new)
							
						else:
							x_new,Px = kalman_update_predict(x_new, Px, measurementX, R, motion, Q, F_small, H)
							y_new,Py = kalman_update_predict(y_new, Py, measurementY, R, motion, Q, F_small, H)
							z_new,Pz = kalman_update_predict(z_new, Pz, measurementZ, R, motion, Q, F_small, H)
							print (x_new)
					if(avg_distance>16000.):
						#print ("using 1")
						if self.frame_idx==1:
							print ("if==1")
							x_new,Px = kalman_update_predict(x_state, P, measurementX, R, motion, Q, F, H)
							y_new,Py = kalman_update_predict(x_state, P, measurementY, R, motion, Q, F, H)
							z_new,Pz = kalman_update_predict(x_state, P, measurementZ, R, motion, Q, F, H)
							#print (x_new)
							#print (y_new)
							#print (z_new)
							
						else:
							x_new,Px = kalman_update_predict(x_new, Px, measurementX, R, motion, Q, F, H)
							y_new,Py = kalman_update_predict(y_new, Py, measurementY, R, motion, Q, F, H)
							z_new,Pz = kalman_update_predict(z_new, Pz, measurementZ, R, motion, Q, F, H)
							
					#print('VELOCITIES ... X,Y,Z')
					#print('alallalalalalalalalalaldasdada',(x_new,y_new,z_new))
					#comparemtx=np.eye(3, dtype=float)
					x_new_pos=x_new/FPS
					y_new_pos=y_new/FPS
					#print()
					z_new_pos=z_new/FPS
					rotvector=np.zeros((3,1))
					rotvectorprev=rotvector
					rotvector[0]=x_new_pos
					#rotvector[0]=x_new[0,0]
					#rotvector[1]=x_new[1,1]
					#rotvector[2]=x_new[2,2]

					rotvector[1]=y_new_pos
					rotvector[2]=z_new_pos

					
					ROTMTX, jacobian= cv.Rodrigues(rotvector) #new rotation matrix from predicte direction vectors

					ROTMTXeye=np.eye(3)
					
					temp_nwp1s = np.zeros((p0_new.shape[0],1,3))
					#temp[:,:,:-1] = p1_new
					#temp[:,:,2]=focal_length  #????????
					#p1_new=temp
					#print("firstpoint")
					#print(p0_new[0,:,:])
					i=0
					#print("i")
					#print(i)
					#print("j")
					
					j=p0_new.shape[0]
					#print(j)
					#print("lastpoint")
					#print(p0_new[j-1,:,:])
					
					#print(p0_new[j,:,:])
					#for tr,(x, y), good_flag, in zip(self.tracks,p1_klt_after_pred.reshape(-1, 2), good)
					
					for rowsp0,rowsp1 in zip(p0_new,p1_new): #getting new points with predicted rotation matrx
						if (i<j):
							#print("i")
							#print(i)
							distance=(rowsp0[0,0]-rowsp1[0,0])**2+(rowsp0[0,1]-rowsp1[0,1])**2

							if(distance>4.):
								#print(rows)
								
								#print(rows2)
								abc=np.dot(ROTMTX,rowsp0.T)
								abc=abc.T
								#if np.allclose(ROTMTX, comparemtx)==True:
								#print("abc_true")
								#print(abc)
								#else:
								#	print("abc_false")
								#	print(abc[0])
								f_corrected=focal_length/abc[0,2]
								#if np.allclose(ROTMTX, comparemtx)==True:
									#print("abc")
									#print(abc)
									
								#else:
								#	print("f_corrected_false")
								#	print(f_corrected)
								abc_new=f_corrected*abc
							else:
								abc=np.dot(ROTMTXeye,rowsp0.T)
								abc=abc.T
								#if np.allclose(ROTMTX, comparemtx)==True:
								#print("abc_true")
								#print(abc)
								#else:
								#	print("abc_false")
								#	print(abc[0])
								f_corrected=focal_length/abc[0,2]
								#if np.allclose(ROTMTX, comparemtx)==True:
									#print("abc")
									#print(abc)
									
								#else:
								#	print("f_corrected_false")
								#	print(f_corrected)
								abc_new=f_corrected*abc

							#print("abc_true")
							#print(abc)
							#if np.allclose(ROTMTX, comparemtx)==True:
							#	print("abc_new")
							#	print(abc_new)
							#abc_fdel=np.delete(abc_new, matrix[:,:,2])
							#print("abc_fdel")
							#print(abc_fdel)
							#abc_new=rows
							temp_nwp1s[i,:,:]=abc_new

							i=i+1
						else:
							break
						
					temp_nwp1s_Fdel=temp_nwp1s[:, :, :2]
					#print("abc_fdel")
					#print(abc_fdel[0,:,:])	p0r, _st, _err = cv.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
					#print ("temp_nwp1s")
					#print (temp_nwp1s[0,:,:])
					p0_klt_after_pred=temp_nwp1s_Fdel
					p0_klt_after_pred=p0_klt_after_pred+principle_point
					p0_klt_after_pred=np.float32(p0_klt_after_pred)
					#print(p0_klt_after_pred)
					
					#print (np.allclose(ROTMTX, comparemtx))
					#if np.allclose(ROTMTX, comparemtx)==True:
					#print (np.allclose(ROTMTX, comparemtx))
					#p1_klt_after_pred=p1
					#print("ROTMTX")
					#print(ROTMTX)
					#print("rotvector")
					#print(rotvector)
				#else:
					p1_klt_after_pred, _st, _err = cv.calcOpticalFlowPyrLK(img0, img1, p0_klt_after_pred, None, **lk_params)
					p0r, _st, _err = cv.calcOpticalFlowPyrLK(img1, img0, p1_klt_after_pred, None, **lk_params) #recalculate optical flow
					kes= p1_klt_after_pred-p0_klt_after_pred #for normal 

					temp_nwk = np.zeros((kes.shape[0],1,2))
					#temp[:,:,:-1] = p1_new
					#temp[:,:,2]=focal_length  #????????
					#p1_new=temp
					#print("firstpoint")
					#print(p0_new[0,:,:])
					inc=0
					#print("i")
					#print(i)
					#print("j")
					
					jec=p0_new.shape[0]
					#print(j)
					#print("lastpoint")
					#print(p0_new[j-1,:,:])
					
					#print(p0_new[j,:,:])
					#for tr,(x, y), good_flag, in zip(self.tracks,p1_klt_after_pred.reshape(-1, 2), good)
					
					for rowsk in kes: #calculate normal
						if (inc<jec):
							
							kx=rowsk[0,0]
							#print(rowsk[0,1])
							ky=rowsk[0,1]*-1
							
							rowsk[0,0]=ky
							rowsk[0,1]=kx
							#print(rowsk[0,0])

							
							abc_new=rowsk
						
							temp_nwk[inc,:,:]=abc_new

							inc=inc+1
						else:
							break
					
					normals=temp_nwk
					i_remove=0
					p1_klt_after_pred_epi=p1_klt_after_pred
					p0_klt_after_pred_epi=p0_klt_after_pred
					j_remove=p1_klt_after_pred_epi.shape[0]
					#print("normals_befre")
					#print(normals)
					#print("p1_klt_after_pred_epi before")
					#print(p1_klt_after_pred_epi)

					'''
					for rowsnormal, rowspoints,rowspoints_prev in zip(normals,p1_klt_after_pred_epi,p0_klt_after_pred_epi):
						if(i_remove<j_remove):
							if (rowsnormal[0,0]==0. and rowsnormal[0,1]==0.):
								normals = np.delete(normals, i_remove, axis=0)
								p1_klt_after_pred_epi = np.delete(p1_klt_after_pred_epi, i_remove, axis=0)
								p0_klt_after_pred_epi = np.delete(p0_klt_after_pred_epi, i_remove, axis=0)
								j_remove=p1_klt_after_pred_epi.shape[0]
								continue
							else:
								i_remove=i_remove+1
								continue
							#i_remove=i_remove+1

						else:
							break	
					'''
					epipoles=np.zeros((1,2))
					upperbound=p1_klt_after_pred_epi.shape[0]-1
					p1_klt_after_pred_epi=p1_klt_after_pred_epi-principle_point
					p0_klt_after_pred_epi=p0_klt_after_pred_epi-principle_point
					#features_selected =select_feature_points(upperbound,normals,p1_klt_after_pred_epi,20)
					#print("ransactime")
					subsetnumber,subset_clusters,subset_normals,subset_epipoles,subset_prevpoints=getsubsets(p1_klt_after_pred_epi,normals,epipoles,p0_klt_after_pred_epi) #get subset clusters. (can be used to take out repeated points and form clusters)
					#print("normals")
					#print(normals)
					#print("p1_klt_after_pred_epi")
					#print(p1_klt_after_pred_epi)
					#print ("subset_clusters")
					#print (subset_clusters)
					#print ("subset_normals")
					#print (subset_normals)
					#print("epipoles")
					#print(epipoles)
					#print("subset_epipoles")
					#print(subset_epipoles)


					#print(".................frame...........................................")
					#subsetnumber=0
					ransac1(normals,p1_klt_after_pred_epi,8,20,subsetnumber,subset_clusters,subset_normals,epipoles,subset_epipoles,False,p0_klt_after_pred_epi,subset_prevpoints)
					#print ("subset_clusters")
					#print('last key')
					#print (subsetnumber)
					#subset_clusters[1]="lalalalalaal"
					#for (k,v),(pk,pv) in zip(subset_clusters.items(), subset_prevpoints.items())

					#print (abc.shape)
					#print (subset_epipoles)
					#print (len(subset_clusters))
					#if(len(subset_clusters)==3):
					#lalala = subset_prevpoints[0]
					#print(len(subset_prevpoints))
					#for (k,v), (k2,v2) in zip(d.items(), d2.items()):
					#TTI_sets=np.zeros((p1_klt_after_pred.shape[0],1,1))
					TTI_set={0:123}
					miss_set={0:123}
					beta_set={0:123}
					phi_set={0:123}
					TTIvsPhi_set={0:123}
					ind=0
					#print(subset_clusters[0].shape)
					#print(subset_epipoles)
					#print(subset_prevpoints[0].shape)
					for p1,v_e,p0 in zip(subset_clusters.values(), subset_epipoles.values(), subset_prevpoints.values() ) :
						if(ind>len(subset_clusters)):
							break
						#print(p1)
						#if v_e[0,0]==0 and v_e[0,1]==0:
							#print(v_e.shape)
							#print(p1.shape)
							#print(p0.shape)
							#print("dssfs......................................................................")
						TTI,miss,beta,phi,TTIvsPhi = calculate_TTI_cvalue(p1,p0,v_e,focal_length)
						#print(miss.shape)
						#print("cluster")
						#print(p1.shape)
						TTIavg= np.average(TTI)
						TTImed=np.median(TTI)
						phiavg=np.average(phi)
						#TTIvsPhiavg=np.average(TTIvsPhi, axis=0)
						TTI_set[ind]=TTI
						miss_set[ind]=miss
						beta_set[ind]=beta
						phi_set[ind]=phiavg
						TTIvsPhi_set[ind]=TTIvsPhi
						ind+=1
						#if(ind>len(subset_clusters))

					#print(subset_clusters)
					
					exiit=0
					
					finalkey=0
					#print("subset_clusters before......................................................")
					#print(subset_clusters)
					'''
					subset_epipoles
					TTI_set
					miss_set

					for key,values in subset_clusters.items():
						#print("finalkey")
						
						finalkey=key
						#print(finalkey)
					#print("finallllessst")
					#print(finalkey)
					#print(subset_clusters[4])
					
					for (key,values),(keytti,valueTTI),(keymiss,valuemiss) in zip (subset_clusters.items(), TTI_set.items(), miss_set.items()):
						#print("key")
						#print(key)
						#exiit=key
						j_remove=values.shape[0]
						i_remove=0
						subberset=np.zeros((values.shape[0],1,2))
						ttisubberset=np.zeros((values.shape[0],1,1))
						misssubberset=np.zeros((values.shape[0],1,1))
						newavg=np.mean(values, axis=0)
						#print(newavg)
						i=0
						for row,rowTTI,rowmiss in zip(values,valueTTI,valuemiss):
							#print("abc")
							#print(key)

							if euclidean_distance(row,newavg) >= 400:
								#print("key")
								#print(key)
								subberset[i]=row
								ttisubberset[i]=rowTTI
								misssubberset[i]=rowmiss
								i+=1
							else:
								continue

						key_remove = 0
						shape_remove=subberset.shape[0]
						for row,rowTTI,rowmiss in zip(subberset,ttisubberset,misssubberset):
							#print(row)
							if(key_remove<shape_remove):
								if row[0,0]==0. and row[0,1]==0.:
									subberset=np.delete(subberset, key_remove, axis=0)
									ttisubberset=np.delete(ttisubberset, key_remove, axis=0)
									misssubberset=np.delete(misssubberset, key_remove, axis=0)
									shape_remove=subberset.shape[0]
									continue
								else:
									key_remove=key_remove+1
									continue
							else:
								break

						if subberset.size!=0:
							subset_clusters[key+finalkey+1]=subberset
							TTI_set[key+finalkey+1]=ttisubberset
							miss_set[key+finalkey+1]=misssubberset
						'''


						#print(TTI_set)
						#print("full")
						#print(subberset)

					#print("subset_clusters after......................................................")
					#print(TTI_set)

					#print(subset_clusters)
					ttivsmiss={}
					
					for (key,value),(key2,val2) in zip(TTI_set.items(),miss_set.items()):
						abc=[]
						for v,v2 in zip (value,val2):
							pqr=float(v[0,0]),float(v2[0,0])
							abc.append(pqr)
						ttivsmiss[key]=abc
					


					d = abs(p0_klt_after_pred-p0r).reshape(-1, 2).max(-1)
					good = d < 3
					#print("good")
					#print (d)
					#print("epipole_points")

					#print(epipole_points)
					new_tracks = []
					#self.tracks2=self.tracks
					#for tr,(x, y),(xp1, yp1), good_flag,r,(epix,epiy) in zip(self.tracks,p1_klt_after_pred.reshape(-1, 2),p1.reshape(-1, 2), good,lines1,epipoles.reshape(-1, 2)):
					for tr,(x, y),good_flag in zip(self.tracks,p1_klt_after_pred.reshape(-1, 2),good):
						if not good_flag:
							continue
						
						#print("tracks")
						#print (self.tracks)
						tr.append((x,y))
						#tr2.append((xp1, yp1))
						#epix=epi[0]
						#epiy=epi[1]
						#print(epix)
						#print(epiy)
						#epix=math.floor(epix)
						#epiy=math.floor(epiy)
						#epix=abs(epix)
						#epiy=abs(epiy)
						#epipoleX=int(epipole[0])
						#epipoleY=int(epipole[1])
						#epiy=int(epipoles[0,0])
						#epix=int(epipoles[0,1])
						#print ("epix")
						#print(epix)
						#print("tr")
						#print (tr)
						#print((x, y))
						if len(tr) > self.track_len:
							del tr[0]

						new_tracks.append(tr)
						#print("appended")
						#x0,y0 = map(int, [0, -r[2]/r[1] ])
						#x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
						
						
						cv.circle(vis, (x, y), 2, (0, 255, 0), -1)
						#print("feature points")
						
						#cv.circle(vis, (epix, epiy), 20, (255, 0, 0), -1)
						#cv.rectangle(vis, (A_RECTa,A_RECTb), (A_RECTc,A_RECTd), (255, 0, 0), 1)
						#x0,y0,x1,y1 = drawlines(img1,img0,lines1,p1,p0)
						#if self.frame_idx % self.detect_interval == 0:
						#cv.line(vis, (x0,y0), (x1,y1), (255, 0, 0),1)
					
					self.tracks = new_tracks
					#print (self.tracks)
					#print (self.tracks[0,0])
					#print (self.tracks[5])
					#print (self.tracks[7])
					#if np.allclose(ROTMTX, comparemtx)==True:
					#cv.polylines(vis, [np.int32(tr2) for tr2 in self.tracks2], False, (0, 0, 0))
					#else:
					if (len(subset_clusters)>1) and (subset_clusters[1] is not None) and (np.array_equal(subset_clusters[1],subset_clusters[0]) is not True):
						length=1
						#COLOR=random_color():
						#color()
						for (key,values),(keygrid,valuegrid) in zip(subset_clusters.items(),ttivsmiss.items()):
						#while (length<len(subset_clusters)):

							#subpoints=subset_clusters[key]
							subpoints=values
							#subepi=subset_epipoles[length]
							subpoints=subpoints+principle_point
							#COLOR=random_color()
							avg=np.mean(valuegrid, axis=0)
							

							#print ("aaaaaaveraaaaage",avg)
							for (x, y),(tti,miss) in zip(subpoints.reshape(-1, 2), valuegrid):
								cv.circle(vis, (int(x), int(y)), 5, (255,0,0), -1)
								
								if tti<0:
									continue
								else:
									if tti<50. and abs(miss)<0.09 :
									#if abs(avg[0])<20. and abs(avg[1])<.5 :
										cv.circle(vis, (int(x), int(y)), 7, (0,0,255), -1)
							#			pi=math.pi
							#			m=(miss*180)/pi
										#print (tti/FPS,m)

					#for key,value in objects.items():
					#	count=len(value)
					#	print(key,":",count)
					#	alphaaaaa=0.
					#	ttiiiiii=0.
					#	countinternal=0
					#	for row in value:
					#		cv.circle(vis, (int(row[0]), int(row[1])), 2, (255,0,0), -1)
					#		if abs(row[2])<100 and abs(row[3])>1.43 and abs(row[3])<1.70:
					#			countinternal+=1
					#			cv.circle(vis, (int(row[0]), int(row[1])), 2, (0,0,255), -1)
					#			ttiiiiii=row[2]
								#print(row)
					#			alphaaaaa=row[3]
					#		else:
					#			continue
								#countinternal+=1
					#	print(key,"redcount:",countinternal)
						#break
						#if countinternal>(count/2):
						#	print ("collision with:",key)
							#print ("alpha:",alphaaaaa,"TTI:",ttiiiiii)
							

								#cv.circle(vis, (int(subepi[0,0]), int(subepi[0,1])), 10, (0,0,255), -1)
								
							#length+=1
					'''
					for (key,value),(k,v) in zip(boundingboxesssss.items(),alpha.items()):
						#if key == 0 or key == 1 or key == 2 or key==3:
						#	continue
						pt1x=float(boundingboxesssss[key][0])
						pt1y=float(boundingboxesssss[key][1])
						pt2x=float(boundingboxesssss[key][2])
						pt2y=float(boundingboxesssss[key][3])
						finalv=float(v)-(3.1415/2)
						#avg=np.mean(value,axis=0)
						#82.5
						#97.5
						#print("boxkey",key,"boxsize",value,"key",keygrid,"value",valuegrid)
						#cv.circle(vis, (int(avg[0]), int(avg[1])), 7, colorcode[key], -1)
						cv.rectangle(vis, (int(pt1x),int(pt1y)), (int(pt2x),int(pt2y)), (255,255,255), 5)
						cv.putText(vis, key, (int(pt1x), int(pt1y)-10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
						#cv.putText(vis, str(finalv), (int(pt1x), int(pt1y)+20), cv.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
						#print (key,"alpha",v,"alphamiss",finalv)
					print("..................frame........................................................")
						#subepi=subset_epipolesp[length]
					'''
					
					'''	
					for (key,value),(keygrid,valuegrid) in zip (bounding_boxes.items(),ttivsmiss.items()):
						#if key == 0 or key == 1 or key == 2 or key==3:
						#	continue
						pt1=bounding_boxes[key][0]
						pt2=bounding_boxes[key][1]
						avg=np.mean(value,axis=0)
						#print("boxkey",key,"boxsize",value,"key",keygrid,"value",valuegrid)
						#cv.circle(vis, (int(avg[0]), int(avg[1])), 7, colorcode[key], -1)
						cv.rectangle(vis, pt1, pt2, colorcode[key], 5)
					print("..................frame........................................................")
						#subepi=subset_epipolesp[length]
					'''


					cv.polylines(vis, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0))
					#if (subset_p1.shape[0]!=0 or subset_p1.shape[0]!=p1_klt_after_pred.shape[0]):
					#if (subset_p1.shape[0]!=0):
					#		for (xp1, yp1) in subset_p1.reshape(-1, 2):
					#			cv.circle(vis, (xp1, yp1), 2, (0, 0, 255), -1)
					#cv.circle(vis, (epipoleX, epipoleY), 20, (255, 0, 0), -1)
					draw_str(vis, (20, 20), 'track count: %d' % len(self.tracks))
					#abc=kalman.correct(R)
					#print("correction")
					#print(K_NEW)
				if self.frame_idx % self.detect_interval == 0:
					mask = np.zeros_like(frame_gray)
					mask[:] = 255
					for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
						cv.circle(mask, (x, y), 5, 0, -1)
					p = cv.goodFeaturesToTrack(frame_gray, mask = mask, **feature_params)
					if p is not None:
						for x, y in np.float32(p).reshape(-1, 2):
							self.tracks.append([(x, y)])
							#if self.frame_idx ==0:
								#self.tracks_first.append([(x, y)])
							#print("tracksafter")
							#print (self.tracks)
				#plt.plot(range(-10,10))
				#plt.axvspan(3, 6, color='red', alpha=0.5)
				#plt.show()

				#print ("self.frame_idx")
				#print (self.frame_idx)
				self.frame_idx += 1
				
				self.prev_gray = frame_gray
				#if len(self.tracks) > 0:

				cv.imshow("plot",imgfig)
				cv.imshow('lk_track', vis)
				print ("self.frame_idx")
				print (self.frame_idx)

				#plt.imshow(vis)
				#kalman correct

				#plt.show()
				ch = cv.waitKey(900)
				if ch == 27:
					break

def main():
    import sys
    try:
        video_src = sys.argv[1]
    except:
        video_src = 0
	#x_state = np.matrix('0. 0. 0. 0. 0. 0.').T
    App(video_src).run()
    print('Done')


if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()
