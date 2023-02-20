import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from data_handler.dataset import Net

N = 10
K = 10
M = 20
net_path = "data/net.dat"
lr = 0.0001
sample_times = 1000
neg_sample = 10

# Device configuration
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

## MF model
class MF(nn.Module):
    def __init__(self):
        super(MF, self).__init__()
        # 讀資料取得 N, K, M
        # 定義 N*K 的矩陣
        # 定義 M*K 的矩陣
        self.net = Net(net_path)
        N, M = self.net.get_usr_itm_num()
        self.U = torch.nn.Parameter(torch.randn(N, K))
        self.I = torch.nn.Parameter(torch.randn(M, K))
        self.rating_matrix = self.net.get_rating_matrix()

    def bpr_update(self):
        usr, itm = self.net.sample_pos()
        negs = self.net.sample_neg(usr, neg_sample)

        u = self.net.__item__(usr)
        ui = self.net.__item__(itm)

        for neg in negs:
            # update U, I
            uj = self.net.__item__(neg)
            neg_uij = torch.dot(self.U[u], self.I[ui]) - torch.dot(self.U[u], self.I[uj]) * -1
            ft = torch.sigmoid(neg_uij)

            self.U[u] += lr * (1 - ft) * (self.I[ui] - self.I[uj])
            self.I[ui] += lr * (1 - ft) * self.U[u]
            self.I[uj] += lr * (1 - ft) * -self.U[u]

    def forward(self):
        return torch.mm(self.U, self.I.t())

    def get_rating_matrix(self):
        return self.rating_matrix

model = MF().to(device)
optimizer = optim.SGD(model.parameters(), lr = lr)

epochs = tqdm(range(sample_times))
for epoch in epochs:
    model.bpr_update()
    output = model()

    loss_func = nn.MSELoss()
    loss = loss_func(output, model.get_rating_matrix())
    epochs.set_description("training loss: {Loss:.8f}".format(Loss=loss.item()))
    
