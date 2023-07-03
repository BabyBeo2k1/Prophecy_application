import torch
import torch.nn as nn
from NetVerifier import  NetVerifier
from PropertyExtractor import netProperty
from iterative_relaxation import Algorithm1
class sample(nn.Module):
    def __init__(self):
        super(sample,self).__init__()
        self.fc1=nn.Linear(2,2)
        self.rl1=nn.ReLU()
        self.fc2=nn.Linear(2,2)
        self.rl2=nn.ReLU()
        self.fc3=nn.Linear(2,2)
        self.rl3=nn.ReLU()
        self.fc4=nn.Linear(2,2)
    def forward(self,input):
        out=self.fc1(input)
        out=self.rl1(out)
        out=self.fc2(out)
        out=self.rl2(out)
        out=self.fc3(out)
        out=self.rl3(out)
        out=self.fc4(out)
        return out
test=sample()
params=test.parameters()
sd=test.state_dict()
keys=list(sd.keys())
w=[[[1.0,-1.0],[1.0,1.0]],[0.0,0.0],[[0.5,-0.2],[-0.5,0.1]],[0.0,0.0],[[1.0,-1.0],[-1.0,1.0]],[0.0,0.0],[[1.0,-1.0],[-1.0,1.0]],[0.0,0.0]]

for i,param in enumerate(params):
    param.data.copy_(torch.tensor(w[i]))

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
ip=torch.tensor([[10.0,-10.0]])
ip=(torch.rand((10,2))-0.5)*10
B=[[[-100.0,100.0],[-100.0,100.0]]]
pattern0=[['on','off'],['on','off']]
print(ip,test(ip))
# verifier=NetVerifier()
# print(verifier.composeDP(test,A,pattern0,ip))

print(Algorithm1.solve(test,ip,B,A))