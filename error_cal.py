from numpy import loadtxt,vstack
from sklearn.metrics import accuracy_score,mean_squared_error

string = ['train', 'test']

for x in range(2):
    mag = loadtxt("output_Amplitude_"+str(string[x])+".out", usecols = [1,2], dtype = float)
    sign = loadtxt("output_sign_"+str(string[x])+".out", usecols = [1,2], dtype = float)

    actual = []
    prediction = []
    count = 0

    for i in xrange (len(mag)):
        if (sign[i][0] == 0.0 and sign[i][1] == 0.0):
            actual.append(-mag[i][0])
            prediction.append(-mag[i][1])
            count += 1
            continue

        elif ( sign[i][0] == 0.0 and sign[i][1] == 1.0) :
            actual.append(-mag[i][0])
            prediction.append(mag[i][1])
            count += 1
            continue

        elif (  sign[i][0] == 1.0 and sign[i][1] == 0.0):
            actual.append(mag[i][0])
            prediction.append(-mag[i][1])
            count += 1
            continue

        else :
            actual.append(mag[i][0])
            prediction.append(mag[i][1])
            count += 1

    prediction, actual = vstack(prediction), vstack(actual)
    acc = mean_squared_error(prediction, actual)
    print (str(string[x])+" error(MSE)   "+str(round(acc,8))+";  number of data  "+str(count))

