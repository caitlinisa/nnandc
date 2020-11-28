import numpy as np 


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
        print (succeed)
        if succeed != p and t == maxEpochs-1:
            print ("Failed")
            return False
        elif succeed == p:
            print ("Succeeded")
            return True
        succeed = 0 
            

            
succeeded = 0
failed = 0          
for idx in range(50):
    data = generate_data(20,20)
    labels = generate_labels(60)
    success = rosenblatt_training(data,labels,20)
    if success:
        succeeded += 1
    else:
        failed += 1
print (succeeded)
print (failed)


