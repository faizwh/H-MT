

def model_conversion(model):
    model.T=args.time
    model.merge = MergeTemporalDim(0)
    model.expand = ExpandTemporalDim(model.T)
    model.init_forward = model.forward
    model.forward = types.MethodType(myforward, model)
    replace_relu_by_neuron(model, model.T)
    return model

def myforward(self, x):
    if self.T > 0:
        x = add_dimention(x, self.T)
        x = self.merge(x)
    out = self.init_forward(x)
    if self.T > 0:
        out = self.expand(out)
    return out