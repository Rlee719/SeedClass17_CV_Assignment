from nn import Module

class xavier(Module):
    def __init__(self, module):
        for param in module.raise_params():
            if param["name"] == "weight" or "bias": 
                array_shape = param["value"].shape()
                param["value"] = np.random.rand(*array_shape) #change init method
