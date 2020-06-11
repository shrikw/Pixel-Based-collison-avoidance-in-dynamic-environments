from math import sqrt
import math
import numpy as np
import random
PI=math.pi
def calculate_TTI_cvalue(p1,p0,Epipole,focal_length):

        temp = np.zeros( (p1.shape[0],1,2) )
        #temp[:,:,2]=focal_length

        temp[:,:,:] = p1
        #print (temp)
        p_1=temp

        temp = np.zeros((p0.shape[0],1,2))
        temp[:,:,:] = p0
        #temp[:,:,2]=focal_length
        p_0=temp

        magepi=np.linalg.norm(Epipole)
        #print(magepi)
        TTI=np.zeros((p_1.shape[0],1,1))
        inc=0
        jec=p_1.shape[0]
        miss=np.zeros((p_1.shape[0],1,1))
        beta=np.zeros((p_1.shape[0],1,1))
        phi=np.zeros((p_1.shape[0],1,1))
        TTIvsPhi=np.zeros((p_1.shape[0],1,2))
        TTIPhi=np.zeros((1,2))
        for rowsp1,rowsp0 in zip(p_1,p_0):

                if (inc<jec):
                        #print (rowsp0)

                        magp0=abs(np.linalg.norm(rowsp0))
                        #print (magp0)
                        #print("lalala")
                        magp1=abs(np.linalg.norm(rowsp1))
                        magci=magp0*magepi
                        #print(magci)
                        magc1_plus1=magp1*magepi
                        #print(rowsp0.shape)
                        #print(Epipole.shape)

                        c_i=np.dot(rowsp0,Epipole.T)/magci
                        #print("c_i")
                        #print(c_i)
                        c_i_plus1=np.dot(rowsp1,Epipole.T)/magc1_plus1
                        #print("c_i_plus1")

                        #print(c_i_plus1)
                        #ciroot=sqrt(1 - c_i_plus1**2)
                        c_i_plus1root=sqrt(1 - c_i_plus1**2)

                        #print(ciroot)
                        #a=(c_i*c_i_plus1root)
                        #b=c_i_plus1*sqrt(1 - c_i**2)
                        #c=b-a
                        #print(c)
                        #k=a/c
                        #print(k)
                        k=(c_i*c_i_plus1root)/( ( c_i*sqrt(1 - c_i_plus1**2) ) - ( c_i_plus1*sqrt(1 - c_i**2) ) )
                        #print (Epipole)
                        #print ("sacdsivbisfsofuswepdiadpaiehfaihfldsh")
                        if 8<abs(rowsp1[0][0]-rowsp0[0][0])<100 and 0<abs(rowsp1[0][1]-rowsp0[0][1])<2:
                                #print(rowsp1[0][0]-rowsp0[0][0])
                                TTI[inc,:,:]=random.uniform(2000000, 4000000)
                        else:
                        #print()
                                TTI[inc,:,:]=k
                        #print()
                        m= math.atan2(k * (c_i_plus1*sqrt(1 - c_i**2)),1)
                        #m=math.atan(k * (c_i_plus1*sqrt(1 - c_i**2)))
                        if Epipole[0,0]==0. and Epipole[0,0]==0.:
                                
                                k=0.2
                                m=0
                                #mtan=0
                        '''        
                        if m!=mtan:
                                print("m...")
                                print(m)
                                print("m.tan..")
                                print(mtan)
                                print((Epipole.T))
                        ''' 
                        bet=rowsp0-Epipole
                        b=math.atan2(bet[0][0], bet[0][1])
                        ph=math.degrees(math.atan2(rowsp1[0][0],focal_length))
                        TTIPhi[0][0]=k
                        TTIPhi[0][1]=ph
                        TTIvsPhi[inc,:,:]=TTIPhi

                        miss[inc,:,:]=m
                        beta[inc,:,:]=b
                        phi[inc,:,:]=ph

                        inc=inc+1
                else:
                        break
        #print (TTIvsPhi)
        return TTI,miss,beta,phi,TTIvsPhi



#defcalculate_miss()                                                                                                                                                                                                               
