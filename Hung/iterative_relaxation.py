import torch
from NetVerifier import NetVerifier
from PropertyExtractor import netProperty
class Algorithm1:
    @classmethod
    def solve(cls,net:torch.nn.Module,inp:torch.Tensor,preconditions:list[list[list[float]]] , postcondition:list[list[float]]):
        prove_patterns=[]
        support_patterns=[]
        input_properties=[]
        patterns_packs=netProperty.get_decision_pattern(net,inp)

        for packs,precondition in zip(patterns_packs,preconditions):
            pattern,output,inp=packs
            if not cls.check_postcondition(inp,net,postcondition):
                continue
            check=False
            for idx,prove in enumerate(prove_patterns):
                if cls.check_pattern(cls,prove,pattern):
                    check=True
                    support_patterns[idx]=inp
                    break
            if check:
                continue

            verifier=NetVerifier()
            DP=verifier.composeDP(net,precondition,postcondition,pattern,inp)
            if DP:
                l=len(pattern)-1
                while l>=0:
                    pattern,layer_pattern=cls.remove_pattern(cls,pattern,l)
                    if verifier.composeDP(net,precondition,postcondition,pattern,inp):
                        l-=1
                    else:
                        cl=l
                        pattern=cls.revoke_pattern(cls,pattern,layer_pattern,l)
                        for idx,node in enumerate(pattern[cl]):
                            pattern[cl][idx]='skip'
                            if not verifier.composeDP(net,precondition,postcondition,pattern,inp):
                                pattern[cl][idx]=node
                        prove_patterns.append(pattern)
                        ipp = []
                        ipp.append(inp)
                        support_patterns.append(inp)

                        break
                # prove_patterns.append(pattern)
                # ipp = []
                # ipp.append(inp)
                # support_patterns.append(inp)
            else:
                input_properties.append(inp)
        return prove_patterns,support_patterns,input_properties

    @classmethod
    def check_postcondition(cls,inp,net,postcondition):
        output=net(inp).detach().numpy()
        for condition in postcondition:
            res=0
            for i in range(len(condition)-1):
                res+=output[i]*condition[i]
            res+=condition[-1]
            if res>0:
                return False
        return True


    def check_pattern(self,pattern1:list[list[str]],pattern2:list[list[str]]):
        for i in range(len(pattern1)):
            for j in range(len(pattern2)):
                if pattern1[i][j]!= 'skip' and pattern2[i][j] != "skip" and pattern2[i][j] != pattern1[i][j]:
                    return False
        return True
    def remove_pattern(self,pattern,layer):
        layer_pattern=[]
        for i in range(len(pattern[layer])):
            layer_pattern.append(pattern[layer][i])
            pattern[layer][i]='skip'
        return pattern,layer_pattern
    def revoke_pattern(self,pattern,layer_pattern,layer):
        for i in range(len(pattern[layer])):
            pattern[layer][i]=layer_pattern[i]
        return pattern
