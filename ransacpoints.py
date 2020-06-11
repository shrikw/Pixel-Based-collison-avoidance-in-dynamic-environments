from itertools import combinations
import numpy as np
import random
from math import sqrt
import sys

def getsubsets(Points,normals,epipoles,prevpoints):
        subset_clusters={0:Points}
        subset_prevpoints={0:prevpoints}
        subset_normals={0:normals}
        subset_epipoles={0:epipoles}
        subsetnumber=0
        return subsetnumber,subset_clusters,subset_normals,subset_epipoles,subset_prevpoints



def get_neighbors(train, test_row, num_neighbors):
    distances = list()
    #print (test_row)
    for train_row in train:
        #print(train_row)
        dist = euclidean_distance(test_row, train_row)
        print (dist)
        distances.append((train_row, dist))
    #print(distances)
    distances.sort(key=lambda tup: tup[1])
    #print(distances)
    neighbors = list()
    for i in range(num_neighbors):
        neighbors.append(distances[i][0])
    return neighbors
    
def predict_classification(train, test_row, num_neighbors):
    neighbors = get_neighbors(train, test_row, num_neighbors)
    #print (neighbors)
    output_values = [row[-1] for row in neighbors]
    #print(output_values)
    prediction = max(set(output_values), key=output_values.count)
    return prediction
 
# kNN Algorithm
def k_nearest_neighbors(train, test, num_neighbors):
    predictions = list()
    for row in test:
        output = predict_classification(train, row, num_neighbors)
        predictions.append(output)
    return(predictions)






def select_feature_points(upperbound,normals,Points,epsilon):
        if (upperbound<7):
                #print("inside if selected features")
                features_selected=np.zeros((3))
                getout=True
                return features_selected,getout
        else:
                #print ("upperbound")
                #print (upperbound)



                selected_features=random.sample(range(0, upperbound), 7)
                results = combinations(selected_features,3)
                results = np.array(list(results))

                total=0
                prevtotal=0
                features_selected=np.zeros((3))
                for rows in results:
                        epipoles,d_three= calcepipole(rows,normals,Points)
                        if (d_three<epsilon):
                                msks= validate_all_points(normals,Points,epipoles,epsilon)
                                total=np.sum(msks)
                                if total>prevtotal:
                                        prevtotal=total
                                        features_selected=rows
                                else:
                                        continue
                        else:
                                continue
                if (prevtotal==0):
                        features_selected=random.sample(range(0, upperbound), 3)

                getout=False
                return features_selected,getout





def ransac1(normals,Points,remaining,epsilon,subsetnumber,subset_clusters,subset_normals,epipoles,subset_epipoles,getout,prevpoints,subset_prevpoints):
        #print("remaining")
        #print(remaining)
        #print("subset_clusters")
        #print(subset_clusters)
        msks= validate_all_points(normals,Points,epipoles,epsilon)
        if (remaining == 1 or getout==True or np.sum(msks)<=0):
                if epipoles[0,0]==0. and epipoles[0,1]==0.:
                    #print(epipoles+1)
                    subset_epipoles[subsetnumber]=epipoles+1
                else:
                    subset_epipoles[subsetnumber]=epipoles
                #print("remaining == 1 or getout==True or np.sum(msks)<=0")
                #print("getout")
                #print (getout)
                #print("np.sum(msks)<=0")
                #print(np.sum(msks))
                #print(subsetnumber)

                return
        else:
                upperbound=Points.shape[0]-1

                selected_features,getout=select_feature_points(upperbound,normals,Points,5)
                epipoless,d_three= calcepipole(selected_features,normals,Points)
                subset_epipoles[subsetnumber]=epipoless
                #print subset_epipoles

                new_points,new_normals,new_prevpoints=get_outliers(msks,normals,Points,prevpoints)
                #subberset=np.zeros((new_points.sape[0],1,2))
                #newavg=np.mean(new_points, axis=0)
                #i=0
                '''
                for row in new_points:
                    if euclidean_distance(row,newavg) >= 10 :
                        subberset[i]=row
                        i+=1
                    else:
                        continue
                        
                '''




                if new_points.size == 0:
                        print ("new_points.size == 0")
                        
                        return subsetnumber

                

                subset_clusters[subsetnumber+1]=new_points
                subset_normals[subsetnumber+1]=new_normals
                subset_prevpoints[subsetnumber+1]=new_prevpoints
                #print(subsetnumber)
                subsetnumber+=1


                ransac1(new_normals,new_points,remaining-1,epsilon,subsetnumber,subset_clusters,subset_normals,epipoless,subset_epipoles,getout,new_prevpoints,subset_prevpoints)



def calcepipole(selected_features,normals,points_p1):

        first=int(selected_features[0])

        second=int(selected_features[1])
        third=int(selected_features[2])
        if normals.shape[0]==0:
                epipoles=np.zeros((1,2))
                d_three=0
                print("0 epipoles")
                return epipoles,d_three


        d_one=np.dot(normals[first],points_p1[first].T)
        d_one=abs(d_one[0,0])
        #d_one=sqrt(d_one)
        d_one=d_one

        d_two=np.dot(normals[second],points_p1[second].T)
        #print(d_two)
        d_two=abs(d_two[0,0])
        #d_two=sqrt(d_two)
        d_two=d_two

        a1=normals[first,0,0]
        a2=normals[first,0,1]
        a3=normals[second,0,0]
        a4=normals[second,0,1]

        A = np.array([[a1,a2],[a3,a4]])

        B = np.array([d_one,d_two])

        if ( np.linalg.det(A)==0 ):

                epipoles=np.zeros((1,2))
                d_three=0
                return epipoles,d_three
        if (A.any()==0 or B.any()==0):
                #print("A")
                #print(A)
                #print("B")
                #print(B)
                epipoles=np.zeros((1,2))
                d_three=0
                return epipoles,d_three

        epipole= np.linalg.solve(A,B)
        epipoles=np.zeros((1,2))
        epipoles[0,0]=epipole[0]
        epipoles[0,1]=epipole[1]
        p3_minus_E= points_p1[third]-epipoles
        d_three=np.dot(normals[third],p3_minus_E.T)

        d_three=abs(d_three[0,0])
        d_three=sqrt(d_three)
        #if(epipole[0]==0. and epipole[1]==0.):
        #print ("sdfbsdifysifsfhbgfodwudf")
        #print (epipoles)


        return epipoles,d_three





def validate_all_points(normals,Points,epipoles,epsilon_all):
        msks=np.zeros((Points.shape[0],1,1))
        i_msk=0
        j_msk=Points.shape[0]
        for rowsn,rowsp1s in zip(normals,Points):
                
                p_minus_E=rowsp1s-epipoles
                d_n=np.dot(rowsn,p_minus_E.T)
                #if epipoles[0,0]==0 and epipoles[0,1]==0:
                    #print (epipoles)
                d_n= abs(d_n[0,0])
                d_n=sqrt(d_n)
                if (i_msk<j_msk):
                        if(d_n<epsilon_all):
                                msks[i_msk,:,:]=True
                                i_msk+=1
                        else:
                                msks[i_msk,:,:]=False
                                i_msk+=1
                else:
                        break
        #if epipoles[0,0]==0 and epipoles[0,1]==0:
            #print (sum(msks))
        return msks

def get_outliers(masks,normals,Points,prevpoints):

    index=0
    endindex=Points.shape[0]
    new_subset=np.zeros((Points.shape[0],1,2))
    new_subset_prevpoints=np.zeros((prevpoints.shape[0],1,2))
    new_normals=np.zeros((Points.shape[0],1,2))
    for rowm,rown,rowp,rowprev in zip(masks,normals,Points,prevpoints):
            if (index<endindex) :
                    if rowm[0,0]==False :
                            new_subset[index,:,:]=rowp
                            new_normals[index,:,:]=rown
                            new_subset_prevpoints[index,:,:]=rowprev
                            index+=1
                            continue
                    else:
                            new_subset=np.delete(new_subset, index, axis=0)
                            new_subset_prevpoints=np.delete(new_subset_prevpoints, index, axis=0)
                            new_normals=np.delete(new_normals, index, axis=0)
                            endindex=new_subset.shape[0]
                            continue
            else:
                    break

    return new_subset,new_normals,new_subset_prevpoints


                                                                                                                                                                                   
                                                                                                             

