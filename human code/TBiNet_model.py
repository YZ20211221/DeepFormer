import torch.nn as nn
import torch
import torch.nn.functional as F

class Lambda(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.mean(x, dim=2)

class attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.Linear = nn.Linear(320, 1)
        self.lamb = Lambda()

    def forward(self, x):
        source = x
        x = x.permute(0, 2, 1)  
        x = self.Linear(x) 
        x = x.permute(0, 2, 1) 
        x = F.softmax(x, dim=2) 
        x = x.permute(0, 2, 1)
        x = self.lamb(x)
        
        x = x.unsqueeze(dim=1)
       
        x = x.repeat(1, 320, 1)
        return source * x



class model(nn.Module):
    def __init__(self, *, sequence_length, n_targets):
        super().__init__()

      
        self.conv_net = nn.Sequential(
            nn.Conv1d(4, 320, kernel_size=26),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=13, stride=13),
            nn.Dropout(p=0.2))
      
        self.attn = attention()

        self.bidirectional_rnn = nn.LSTM(input_size=320, hidden_size=320, num_layers=1, batch_first=True, bidirectional=False)
        self.dropout = nn.Dropout(p=0.5)
      
        self.Linear1 = nn.Linear(320*75, 695)
        self.act1 = nn.ReLU(inplace=True)
    
        self.Linear2 = nn.Linear(695, 919)
        self.act2 = nn.Sigmoid()

    def forward(self, x):
        x = self.conv_net(x)
     
        x = self.attn(x)
      
        x = x.permute(0, 2, 1)
        x,_ = self.bidirectional_rnn(x)
        x = self.dropout(x)
      
        x = x.reshape(x.size(0), 320*75)
        x = self.act1(self.Linear1(x))
        output = self.act2(self.Linear2(x))
        return output


        
def criterion():
    return nn.BCELoss()

def get_optimizer(lr):
    return (torch.optim.Adam, {"lr": lr, "weight_decay": 1e-6})

