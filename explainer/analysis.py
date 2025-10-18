import numpy as np
from MTF_picture import final_MTF
import joblib

Variable_names = ['Variable 0', 'Variable 1', 'Variable', 'Variable 3', 'Variable 4', 'Variable 5',
                'Variable 6', 'Variable 7', 'Variable 8', 'Variable 9', 'Variable 10', 'Variable 11',
                'Variable 12', 'Variable 13', 'Variable 14', 'Variable 15', 'Variable 16', 'Variable 17',
                'Variable 18', 'Variable 19', 'Variable 20', 'Variable 21', 'Variable 22', 'Variable 23',
                'Variable 24', 'Variable 25', 'Variable 26', 'Variable 27', 'Variable 28', 'Variable 29',
                'Variable 30', 'Variable 31', 'Variable 32', 'Variable 33', 'Variable 34', 'Variable 35',
                'Variable 36', 'Variable 37']



def is_normal(model, X, order, mid, normal):
    for i in range(mid+1):
        aindex=order[i] 
        X[0,aindex]=normal[aindex]
    X1=X
    X2=np.array(normal).reshape(1, X1.shape[1])


    X1 = np.float32(X1)
    X2 = np.float32(X2)

    y_predict=model.predict(X1, X2)

    if y_predict[0]: #y_predict==1,abnomal
        return False
    else: 
        return True

def identifier(model, X, order, normal):
    left=0
    right=len(order)-1
    target=-1
    while left <= right:
        if left == right:
            target = left
            break
        else:
            mid = (right + left) // 2
            if is_normal(model,X,order,mid, normal): 
                right = mid
            elif not is_normal(model,X,order,mid, normal): 
                left = mid + 1

    Avariables=[]
    for i in range(target+1):
        Avariables.append(order[i]+1)

    return Avariables


def classifier(X,A_C,dataset):
    for i in range(len(A_C)):
        X[0,i]=X[0,i]*A_C[i]

    kmeans=joblib.load('./explainer/'+dataset+'_kmeans.pkl')
    label=kmeans.predict(X)

    return label


def get_interpretation(path):
    # abnomal_set
    fileHandler = open(path, "r")
    listOfLines = fileHandler.readlines()
    start = []
    end=[]
    variable_list=[]
    for line in listOfLines:
        line=line.split(':')
        abrange=line[0]
        abrange=abrange.split('-')
        start.append(int(abrange[0]))
        end.append(int(abrange[1]))

        abvariable=line[1].replace('\n', '')
        abvariable=abvariable.split(',')
        abvariable=[int(num) for num in abvariable]
        variable_list.append(abvariable)

    fileHandler.close()

    return start,end,variable_list

def ground_truth(start,end,variable_list,index):
    for i in range(len(start)):
        if start[i]<=index and end[i]>index:
            return variable_list[i]
        elif start[i]>index:
            return []
    return []

