import torch

def approx_int(dx, x0, Ts=1):
        return x0 + Ts*dx

def approx_diff(x, xold, Ts=1):
    return (x - xold)/Ts

class ResGen2(torch.nn.Module):
    def __init__(self):
        super(ResGen2, self).__init__()
        self.linear1 = torch.nn.Linear(2, 1)
        self.linear1.bias.data.fill_(0)
        self.linear1.weight.data.uniform_(-1, 1)
        self.linear2 = torch.nn.Linear(3, 1)
        self.linear2.bias.data.fill_(0)
        self.linear2.weight.data.uniform_(-1, 1)

    def forward(self, y, x):
        # u, y1, y3, x1, x2, x3
        u, y1, y3 = y[0], y[1], y[2]
        x1, x2, x3 = x[0], x[1], x[2]

        dx3 = approx_diff(y3, x3)
        x3 = y3
        x2 = self.linear1(torch.tensor([[x3, dx3]]))
        dx1 = self.linear2(torch.tensor([[u, x1, x2]]))
  
        x1 = approx_int(dx1, x1)
        r = x1 - y1
        x = torch.tensor([x1[0][0], x2[0][0], x3])
        return x, r

class ResGen3(torch.nn.Module):
    def __init__(self):
        super(ResGen3, self).__init__()
        self.linear1 = torch.nn.Linear(4, 1)
        self.linear1.bias.data.fill_(0)
        self.linear1.weight.data.uniform_(-1, 1)
        self.linear2 = torch.nn.Linear(4, 1)
        self.linear2.bias.data.fill_(0)
        self.linear2.weight.data.uniform_(-1, 1)

    def forward(self, y, x):
        # u, y1, y3, x1, x2, x3
        u, y1, y3 = y[0], y[1], y[2]
        x1, x2, x3 = x[0], x[1], x[2]
        x3 = y3
        
        dx1 = self.linear1(torch.tensor([[u, x1, x2, x3]]))
        dx2 = self.linear2(torch.tensor([[u, x1, x2, x3]]))
        x1 = approx_int(dx1, x1)
        x2 = approx_int(dx2, x2)
        r = x1 - y1
        x = torch.tensor([x1[0][0], x2[0][0], x3])
        return x, r
    
