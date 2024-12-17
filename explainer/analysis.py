import numpy as np
from ts_2_img import final_MTF
import joblib

Variable_names = ['Variable 0', 'Variable 1', 'Variable', 'Variable 3', 'Variable 4', 'Variable 5',
                 'Variable 6', 'Variable 7', 'Variable 8', 'Variable 9', 'Variable 10', 'Variable 11',
                 'Variable 12', 'Variable 13', 'Variable 14', 'Variable 15', 'Variable 16', 'Variable 17',
                 'Variable 18', 'Variable 19', 'Variable 20', 'Variable 21', 'Variable 22', 'Variable 23',
                 'Variable 24', 'Variable 25', 'Variable 26', 'Variable 27', 'Variable 28', 'Variable 29',
                 'Variable 30', 'Variable 31', 'Variable 32', 'Variable 33', 'Variable 34', 'Variable 35',
                 'Variable 36', 'Variable 37']

# normal=[1.61616e-01, 7.80800e-03, 8.70000e-03, 1.09310e-02, 9.64706e-01, 4.66733e-01,
#      3.31906e-01, 0.00000e+00, 1.75000e-04, 7.08600e-03, 9.14000e-04, 8.84620e-02,
#      0.00000e+00, 1.15140e-02, 3.37800e-03, 1.06300e-03, 0.00000e+00, 0.00000e+00,
#      4.22164e-01, 2.79863e-01, 4.43785e-01, 4.26023e-01, 4.58568e-01, 1.54618e-01,
#      4.16393e-01, 1.39315e-01, 0.00000e+00, 4.50059e-01, 0.00000e+00, 1.26437e-01,
#      4.67644e-01, 4.72222e-01, 1.66667e-01, 1.02740e-01, 4.92975e-01, 4.92605e-01,
#      0.00000e+00, 0.00000e+00]

def is_normal(model, X, order, mid, normal):
    #替换X前mid个异常变量(包括mid)为正常值，得到X1
    for i in range(mid+1):
        aindex=order[i] #要进行替换的异常变量的索引
        X[0,aindex]=normal[aindex]
    X1=X
    X2=np.array(normal).reshape(1, X1.shape[1])

    #得到MTF时序数据X2
    # n_bins = 8
    # p=[0.3,0.7]
    # X2 = final_MTF(X1, 1, n_bins, p)

    X1 = np.float32(X1)
    X2 = np.float32(X2)

    y_predict=model.predict(X1, X2)
    # print("y_predict:")
    # print(y_predict)
    if y_predict[0]: #y_predict==1,异常
        return False
    else: #y_predict==1,正常
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
            if is_normal(model,X,order,mid, normal): #替换后正常
                right = mid
            elif not is_normal(model,X,order,mid, normal): #替换后异常
                left = mid + 1

    # print(target)
    Avariables=[]
    for i in range(target+1):
        # Avariables.append(Variable_names[order[i]])
        Avariables.append(order[i]+1)

    return Avariables


def classifier(X,A_C,dataset):
    for i in range(len(A_C)):
        X[0,i]=X[0,i]*A_C[i]

    kmeans=joblib.load('./explainer/'+dataset+'_kmeans.pkl')
    label=kmeans.predict(X)

    return label


def get_interpretation():
    # abnomal_set

    fileHandler = open("./ServerMachineDataset/interpretation_label/machine-1-6.txt", "r")
    # fileHandler = open("./Data/SMDataset/test_abnomal_variable.txt", "r")
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

if __name__ == '__main__':
    start,end,variable_list=get_interpretation()
    # print(start)
    # print(end)
    # print(variable_list)

    hit_1=0
    hit_2=0
    hit_3=0

    for i in range(400,405):
        Gt=ground_truth(start,end,variable_list,i)
        #Gt=[12,15]

        As=[12,15,13,16,9]

        hit = [0, 0, 0]
        for j in range(3):
            if j==0 and As[j] in Gt:
                hit[0] = 1
            elif As[j] in Gt and hit[j-1]==1:
                hit[j] = 1

        print(hit)
        hit_1 += hit[0]
        hit_2 += hit[1]
        hit_3 += hit[2]

    print(hit_1,hit_2,hit_3)