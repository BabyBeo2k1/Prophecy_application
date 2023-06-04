import torch
import torch.nn.functional as F
import numpy as np
from NetVerifier import NetVerifier
class netProperty():
    encoder_status={
        True:'on',
        False:'off',
    }
    @classmethod
    def get_input_property(cls,net,data):
        patterns=[]
        outputs=[]

        for idx,datapoint in enumerate(data):
            pattern=[]
            ip,op=datapoint
            layer_outputs={}
            ip=torch.tensor(ip,dtype=torch.float)
            def forward_hook(module, input, output):
                layer_outputs[module] = output

            for name, module in net.named_modules():
                if type(module)==torch.nn.modules.Linear:

                    module.register_forward_hook(forward_hook)
            net.eval()
            out=net(ip)

            for module, output in layer_outputs.items():
                pattern.append(list((output>0).detach().numpy()))
            trash=pattern.pop()

            for i in range(len(pattern)):
                for j in range(len(pattern[i])):
                    pattern[i][j]=cls.encoder_status[pattern[i][j]]
            patterns.append(pattern)
            outputs.append(out)
        return patterns,outputs







