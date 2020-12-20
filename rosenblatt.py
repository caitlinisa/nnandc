import numpy as np 
import matplotlib.pyplot as plt


def generate_data(Pnum,Ndim):
    data = np.random.normal(0, 1, size=(Ndim, Pnum))
    #print (data.shape)
    return data

def generate_labels(Pnum):
    labels = np.random.randint(2, size=Pnum)
    labels = (labels * 2)-1 
    labels = np.reshape(labels,(Pnum,1))
    #print (labels.shape)
    return labels

def rosenblatt_training(data,labels,maxEpochs,c):
    n,p = data.shape
    weights = np.zeros((n,1))
    succeed = 0
    for t in range(maxEpochs):
        for idx in range(p):
            data_ex = data[:,idx]
            labels_ex = labels[idx,:]
            data_ex = np.reshape(data_ex,(n,1))
            labels_ex = np.reshape(labels_ex,(1,1))
            weight_vector = np.squeeze(np.asarray(weights))
            data_vector = np.squeeze(np.asarray(data_ex))
            E = ((np.dot(weight_vector , data_ex) * labels_ex)[0])[0]
            if E > c:
                weights = weights
                succeed += 1
            else:
                weights = weights + ((1.0/n)*data_ex*labels_ex)
        #print (succeed)
        if succeed != p and t == maxEpochs-1:
            #print ("Failed")
            return False
        elif succeed == p:
            #print ("Succeeded")
            return True
        succeed = 0 
            

def compareN():        
    succeeded = 0
    failed = 0 
    ratio = []
    #num_examples = [15,20,25,30,35,40,45,50,55,60]
    alpha = [0.75,1.0,1.25,1.5,1.75,2.0,2.25,2.5,2.75,3.0]
    alpha = [1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5]
    n = [20,60,100] 
    xaxis  = [0.75,1.0,1.25,1.5,1.75,2.0,2.25,2.5,2.75,3.0]
    totalruns = 50
    maxEpochs = 100
    labels = ['N = 20', 'N = 60','N = 100'] 

    for N in n:
        #num_examples = N * alpha
        for ex in range(len(alpha)):     
            for idx in range(totalruns):
                #print(alpha[ex]*N)
                data = generate_data(int(alpha[ex]*N),N)
                labels = generate_labels(int(alpha[ex]*N))
                success = rosenblatt_training(data,labels,100)
                if success:
                    succeeded += 1
                else:
                    failed += 1
            fraction = succeeded/totalruns
            ratio.append(fraction)
            succeeded = 0
            failed = 0
        plt.plot(alpha,ratio)
        ratio = []

    plt.legend(['N = 20', 'N = 60','N = 100'] ,loc = "upper right")
    plt.title("Fraction of successful runs against the value of alpha")
    plt.xlabel('alpha')
    plt.ylabel('Fraction of successful runs')
    plt.show()

def compareC():
    succeeded = 0
    failed = 0 
    ratio = []
    N = 100
    #num_examples = [15,20,25,30,35,40,45,50,55,60]
    alpha = [0.75,1.0,1.25,1.5,1.75,2.0,2.25,2.5,2.75,3.0]
    alpha = [1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5]
    c = [0,0.1,0.2] 
    xaxis  = [0.75,1.0,1.25,1.5,1.75,2.0,2.25,2.5,2.75,3.0]
    totalruns = 50
    maxEpochs = 100
    labels = ['c = 0', 'c = 0.1','c = 0.2'] 

    for C in c:
        #num_examples = N * alpha
        for ex in range(len(alpha)):     
            for idx in range(totalruns):
                #print(alpha[ex]*N)
                data = generate_data(int(alpha[ex]*N),N)
                labels = generate_labels(int(alpha[ex]*N))
                success = rosenblatt_training(data,labels,100,C)
                if success:
                    succeeded += 1
                else:
                    failed += 1
            fraction = succeeded/totalruns
            ratio.append(fraction)
            succeeded = 0
            failed = 0
        plt.plot(alpha,ratio)
        ratio = []

    plt.legend(['c = 0', 'c = 0.1','c = 0.2']  ,loc = "upper right")
    plt.title("Fraction of successful runs against the value of alpha")
    plt.xlabel('alpha')
    plt.ylabel('Fraction of successful runs')
    plt.show()



compareC()