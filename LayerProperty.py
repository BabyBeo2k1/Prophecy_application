import torch
import torch.nn.functional as F
class LayerProperty():
    def __init__(self,net,data):
        self.net=net
        self.data=data
    def get_pattern(self):
        input, output = self.data
        state_dict=self.net.state_dict()
        keys=list(state_dict.keys())
        input=torch.tensor(input,dtype=torch.float32)
        out=input
        
        hidden= {}
        for i in range(int(len(keys)/2)):
            state_dict[keys[2 * i + 1]]=torch.reshape(state_dict[keys[2*i+1]],(1,state_dict[keys[2*i+1]].shape[0]))
            out=torch.mm(out,torch.transpose(state_dict[keys[2*i]],0,1))+state_dict[keys[2*i+1]]
            hidden[f"{i}"]=out>0
            out=F.relu(out)
        status=torch.zeros((1,len(input)))
        hidden["verified"]=status
        self.layer_patterns=hidden
    def writeProperty(self,layer,output):
        input,output=self.data


