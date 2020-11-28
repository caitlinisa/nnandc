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

def rosenblatt_training(data,labels,maxEpochs):
    n,p = data.shape
    weights = np.zeros((n,1))
    succeed = 0
    for t in range(maxEpochs):
        for idx in range(p):
            data_ex = data[:,idx]
            #print (data.shape)
            labels_ex = labels[idx,:]
            data_ex = np.reshape(data_ex,(n,1))
            labels_ex = np.reshape(labels_ex,(1,1))
            #print(data_ex.shape)
            #print(labels_ex.shape)
            #print (labels.shape)
            weight_vector = np.squeeze(np.asarray(weights))
            data_vector = np.squeeze(np.asarray(data_ex))
            #print(weight_vector)
            #print (data_vector.shape)
            E = ((np.dot(weight_vector , data_ex) * labels_ex)[0])[0]
            #print(E)
            #exit(0)
            if E > 0:
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
            

            
succeeded = 0
failed = 0 
ratio = []
num_examples = [15,20,25,30,35,40,45,50,55,60]
n = 20 
totalruns = 50
maxEpochs = 100  
for ex in range(len(num_examples)):     
    for idx in range(totalruns):
        data = generate_data(num_examples[ex],n)
        labels = generate_labels(num_examples[ex])
        success = rosenblatt_training(data,labels,100)
        if success:
            succeeded += 1
        else:
            failed += 1
    fraction = succeeded/totalruns
    ratio.append(fraction)
    succeeded = 0
    failed = 0


xaxis  = [0.75,1.0,1.25,1.5,1.75,2.0,2.25,2.5,2.75,3.0]
plt.plot(xaxis,ratio)
plt.title("Fraction of successful runs against the value of alpha")
plt.xlabel('alpha')
plt.ylabel('Fraction of successful runs')
plt.show()


