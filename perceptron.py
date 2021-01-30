import numpy as np 
import matplotlib.pyplot as plt
import math

def generate_data(Pnum,Ndim):
    data = np.random.normal(0, 1, size=(Ndim, Pnum))
    #print (data.shape)
    return data

def generate_labels(teacher,data):
    #print (teacher.shape, data.shape)
    label_unsigned = np.dot(teacher,data)
    #print(label_unsigned)
    labels = np.zeros(len(label_unsigned))
    for i in range(len(label_unsigned)):
        if label_unsigned[i] < 0:
            labels[i] = -1
        else:
            labels[i] = 1
    #print (labels)
    return labels

def minover_algorithm(data,labels,teacher,maxEpochs):
    n,p = data.shape
    #print(n,p)
    #print(labels.shape,data.shape,teacher.shape)
    student = np.zeros((n,1))
    generalization_error = 0
    for t in range(maxEpochs):
        #print (np.matmul(data, labels))
        stability = np.zeros(p)
        for i in range (p):
            stability[i] = np.dot(np.squeeze(student),data[:,i]*labels[i,:])
        #print(stability)
        index = np.argmin(stability)
        #print(index)
        x =(1.0/n)*data[:,index]*labels[index,:]
        x = np.reshape(x,(n,1))
        #print (x)
        #print (x.shape)
        #print (student)
        student = student + x
        #print (student)
        #exit()
        #print ("final student shape",student.shape)
        
    generalization_error = np.arccos(np.dot(np.squeeze(student),np.squeeze(teacher))/(np.linalg.norm(student)*np.linalg.norm(teacher)))/np.pi
    #print ("gen error",generalization_error)
    return generalization_error
    
            

def compareN():        
    succeeded = 0
    failed = 0 
    ratio = []
    #num_examples = [15,20,25,30,35,40,45,50,55,60]
    alpha = [0.75,1.0,1.25,1.5,1.75,2.0,2.25,2.5,2.75,3.0]
    alpha = [1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5]
    alpha = [0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0,2.25,2.5,2.75,3.0,3.25,3.5,3.75,4.0,4.25,4.5,4.75,5.0]
    n = [20] 
    xaxis  = alpha
    totalruns = 30
    maxEpochs = 5
    labels = ['N = 20', 'N = 60','N = 100'] 
    sum_generror = 0

    for N in n:
        #num_examples = N * alpha
        for ex in range(len(alpha)):     
            for idx in range(totalruns):
                #print(alpha[ex]*N)
                teacher  = np.random.rand(N)
                factor = np.linalg.norm(teacher)/math.sqrt(N)
                teacher = teacher / factor
                #print (np.linalg.norm(teacher),math.sqrt(N))
                data = generate_data(int(alpha[ex]*N),N)
                labels = generate_labels(teacher,data)
                length = len(labels)
                labels = np.reshape(labels, (length, 1))
                generalization_error = minover_algorithm(data,labels,teacher,1000)
                sum_generror += generalization_error 
                #exit()
            fraction = sum_generror/float(totalruns)
            ratio.append(fraction)
            sum_generror = 0
        plt.plot(alpha,ratio)
        ratio = []

    #plt.legend(['N = 20', 'N = 60','N = 100'] ,loc = "upper right")
    plt.title("Generalization Error")
    plt.xlabel('alpha')
    plt.ylabel('Generalization error')
    plt.show()




compareN()