import os

import torch

import torch.nn as nn
import torch.nn.functional as F
from LayerProperty import LayerProperty
class ACASX(nn.Module):
    def __init__(self):
        super(ACASX,self).__init__()
        self.fc0 =nn.Linear(5,50)
        self.fc1 =nn.Linear(50,50)
        self.fc2 =nn.Linear(50,50)
        self.fc3 =nn.Linear(50,50)
        self.fc4 =nn.Linear(50,50)
        self.fc5 =nn.Linear(50,50)
        self.output_layer = nn.Linear(50,5)
    def forward(self,input):
        x=F.relu(self.fc0(input))
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=F.relu(self.fc3(x))
        x=F.relu(self.fc4(x))
        x=F.relu(self.fc5(x))
        out =(self.output_layer(x))
        return out
    def set_weight(self,path):
        with open (path) as f:
            lines = f.readlines()
            self.parameters
            weight=[]
            for indx in range(0,len(lines)):
                vals = lines[indx].split(',')
                #print vals
                weight.append(float(vals[4]))
            weight_tensor=torch.tensor(weight)
            count = 0
            params = self.parameters()
            for param in params:
                size = param.numel()
                param.data.copy_(weight_tensor[count:count + size].view_as(param))
                count += size
    def read_data(self,path):
        num = 0
        with open(path) as f:
            lines = f.readlines()
            print(len(lines), "examples")
            acas_train =torch.empty([len(lines), 5], dtype=torch.double)
            acas_train_labels = torch.zeros(len(lines), dtype=torch.int)

            for l in range(len(lines)):
                k = [float(stringIn) for stringIn in lines[l].split(',')]  # This is to remove the useless 1 at the start of each string. Not sure why that's there.
                # acas_train[l+num] = np.zeros(5,dtype=float) #we're asuming that everything is 2D for now. The 1 is just to keep numpy happy.
                if len(k) > 5:
                    lab = int(k[5])
                    # if ((lab == 0) or (lab == 2)):
                    #  lab = 0
                    # else:
                    #  lab = 1
                    acas_train_labels[l + num] = lab

                count = 0
                for i in range(0, 5):
                    # print(count)
                    acas_train[l + num][i] = k[i]

                    # print(k[i])

        return acas_train, acas_train_labels
def main():
    #download input
    #os.system("wget https://raw.githubusercontent.com/safednn-nasa/prophecy_DNN/master/clusterinACAS_0_short.csv -O ./clusterinACAS_0_shrt.csv")
    #os.system("wget 'https://raw.githubusercontent.com/safednn-nasa/prophecy_DNN/master/ACASX_layer.txt' -O ./ACASX_layer.txt")
    test=ACASX()
    print(list(test.state_dict().keys()))
    test.set_weight("./ACASX_layer.txt")
    data=test.read_data("clusterinACAS_0_shrt.csv")
    ACASX_LP=LayerProperty(test,data)
    ACASX_LP.get_pattern()
    print(len(ACASX_LP.layer_patterns))

if __name__=='__main__':
    main()
