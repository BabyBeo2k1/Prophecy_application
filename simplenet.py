import torch
import torch.nn as nn
from NetVerifier import  NetVerifier
from PropertyExtractor import netProperty
class sample(nn.Module):
    def __init__(self):
        super(sample,self).__init__()
        self.fc1=nn.Linear(2,2)
        self.rl1=nn.ReLU()
        self.fc2=nn.Linear(2,2)
        self.rl2=nn.ReLU()
        self.fc3=nn.Linear(2,2)
    def forward(self,input):
        out=self.fc1(input)
        out=self.rl1(out)
        out=self.fc2(out)
        out=self.rl2(out)
        out=self.fc3(out)
        return out
test=sample()
params=test.parameters()
sd=test.state_dict()
keys=list(sd.keys())
w=[[[1.0,-1.0],[1.0,1.0]],[0.0,0.0],[[0.5,-0.2],[-0.5,0.1]],[0.0,0.0],[[1.0,-1.0],[-1.0,1.0]],[0.0,0.0]]

for i,param in enumerate(params):
    param.data.copy_(torch.tensor(w[i]))

w1=torch.tensor([[1.0,1.0],[-1.0,1.0]])
w2=torch.tensor([[0.5,-0.5],[-0.2,0.1]])
w3=torch.tensor([[1.0,-1.0],[-1.0,1.0]])
b=torch.zeros(2)
test.state_dict()[keys[0]]=w1
test.state_dict()[keys[2]]=w1
test.state_dict()[keys[4]]=w1
test.state_dict()[keys[1]]=b
test.state_dict()[keys[3]]=b
test.state_dict()[keys[5]]=b
# pattern=[['on','off'],['skip','skip']]
# A=[[-1.0,1.0,0.0]]
#
# dt=[([1.0,2.0],[1,0]),([0.5,-12.0],[1,0]),([10.0,10.0],[1,0])]
# patterns,outputs=netProperty.get_input_property(test,dt)
A=[[-1.0,1.0,0.0]]
# res=[]
# test.eval()
# print(test.state_dict())
# for pattern in patterns:
#
#     res.append(NetVerifier.DP(pattern,A,test))
# for i in range(len(patterns)):
#     print(patterns[i])
#     print(outputs[i])
#     print(res[i])
ip=torch.rand((1,2))
pattern0=[['on','off'],['skip','skip']]
verifier=NetVerifier()
print(verifier.composeDP(test,A,pattern0,ip))
print()