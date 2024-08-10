"""
This is the file conataining the model and the training procedure for the 
Explainable Graph AI Model.

When run directly, the script will TRAIN the model using the dataset present at
\scheduler\BaGTI\datasets\energy_latency_50_scheduling.csv

When called from XGAI.py, the empty model can be imported, which will then be populated using
the latest checkpoint saved in the training process.

-------------

Ideally, to fully integrate this into COSCO, it would be better to move this whole model to /scheduler/BaGTI/src/models.py
and use /scheduler/BaGTI/train.py to train it, but this way I have better control over the code and can remove
most of the redundant parts (that I did not like) from COSCO.

"""

import time
from torch import nn
import random
import torch
import numpy as np

def load_energy_latencyGNN_data(HOSTS):
	dataset_path = 'datasets/energy_latency_'+str(HOSTS)+'_scheduling.csv'
	data = pd.read_csv(dataset_path) if os.path.exists(dataset_path) else pd.read_csv('scheduler/BaGTI/'+dataset_path)
	data = data.values.astype(float)
	max_ips_container = max(data.max(0)[HOSTS:2*HOSTS])
	dataset = []
	print("Dataset size", data.shape[0])
	for i in range(data.shape[0]):
		cpuH, cpuC, alloc = [], [], []
		u, v = [], []
		for j in range(HOSTS):
			cpuH.append(data[i][j]/100)
			cpuC.append(data[i][j+HOSTS]/max_ips_container)
			oneHot = [0] * HOSTS
			if int(data[i][j+(2*HOSTS)]) >= 0: 
				u.append(j); v.append(int(data[i][j+(2*HOSTS)]) + HOSTS)
				oneHot[int(data[i][j+(2*HOSTS)])] = 1
			alloc.append(oneHot)
		cpu = np.array(cpuC + cpuH).reshape(-1, 1)
		cpuH = np.array([cpuH]).transpose()
		cpuC = np.array([cpuC]).transpose()
		alloc = np.array(alloc)
		g = dgl.graph((u, v), num_nodes=2*HOSTS)
		g = dgl.add_self_loop(g)
		dataset.append((g, torch.Tensor(cpu), np.concatenate((cpuH, cpuC, alloc), axis=1), torch.Tensor([(data[i][-2]- data.min(0)[-2])/(data.max(0)[-2] - data.min(0)[-2]), max(0, data[i][-1])/data.max(0)[-1]])))
	return dataset, len(dataset), max_ips_container

class LayerNormGRUCell(nn.RNNCellBase):

    def __init__(self, input_size, hidden_size, bias=True):
        super(LayerNormGRUCell, self).__init__(input_size, hidden_size, bias, num_chunks=3)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.weight_ih = nn.Parameter(torch.Tensor(3 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.Tensor(3 * hidden_size, hidden_size))
        if bias:
            self.bias_ih = nn.Parameter(torch.Tensor(3 * hidden_size))
            self.bias_hh = nn.Parameter(torch.Tensor(3 * hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
        self.reset_parameters()

        self.ln_resetgate = nn.LayerNorm(hidden_size)
        self.ln_inputgate = nn.LayerNorm(hidden_size)
        self.ln_newgate = nn.LayerNorm(hidden_size)
        self.ln = {
            'resetgate': self.ln_resetgate,
            'inputgate': self.ln_inputgate,
            'newgate': self.ln_newgate,
        }

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input, hx):
        #self.check_forward_input(input)
        #self.check_forward_args(input, hx)
        return self._LayerNormGRUCell(
            input, hx,
            self.weight_ih, self.weight_hh, self.ln,
            self.bias_ih, self.bias_hh,
        )
class GatedRGCNLayer(nn.Module):
    # The Gated GCN layer which has been borrowed from DGL. Contains a link to the source module.
    # Source = https://docs.dgl.ai/_modules/dgl/nn/pytorch/conv/gatedgraphconv.html#GatedGraphConv
    def __init__(self, in_size, out_size, activation):
        super(GatedRGCNLayer, self).__init__()
        self.weight = nn.Linear(in_size, out_size)
        self.reduce = nn.Linear(in_size, out_size)
        self.activation = activation
        self.gru = LayerNormGRUCell(out_size, out_size, bias=True)

    def forward(self, G, features):
        funcs = {}; feat = self.activation(self.reduce(features))
        for _ in range(N_TIMESEPS):
            Wh = self.weight(features)
            G.ndata['Wh'] = Wh
            G.update_all(fn.copy_u('Wh', 'm'), fn.mean('m', 'h'))
            feat = self.gru(G.ndata['h'], feat)
        return self.activation(feat)

class energy_latencyGNN_50(nn.Module):
    def __init__(self):
        super(energy_latencyGNN_50, self).__init__()
        self.name = "energy_latencyGNN_50"
        self.emb = 5
        self.find = nn.Sequential(
            nn.Linear(50 * 2 * self.emb + 50 * 52, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64), 
            nn.LeakyReLU(),
            nn.Linear(64, 2),
            nn.Sigmoid())
        self.grapher = nn.ModuleList()
        self.grapher.append(GatedRGCNLayer(1, self.emb, activation=nn.LeakyReLU()))
        for i in range(2):
            self.grapher.append(GatedRGCNLayer(self.emb, self.emb, activation=nn.LeakyReLU()))

    def forward(self, graph, data, d):
        x = data; graph.ndata['h'] = data
        for layer in self.grapher:
            x = layer(graph, x)
        x = x.view(-1)
        x = torch.cat((x, d.view(-1)))
        x = self.find(x)
        #if not('train' in argv[0] and 'train' in argv[2]):
        #   x = Coeff_Energy*x[0] + Coeff_Latency*x[1]
        return x
    
def load_model(filename, model, data_type):
    MODEL_SAVE_PATH = ""
    optimizer = torch.optim.Adam(model.parameters() , lr=0.001, weight_decay=1e-5) if 'stochastic' not in data_type else torch.optim.AdamW(model.parameters() , lr=0.0001)
    file_path1 = MODEL_SAVE_PATH + "/" + filename + "_Trained.ckpt"
    file_path2 = 'scheduler/BaGTI/' + file_path1
    file_path = file_path1 #if os.path.exists(file_path1) else file_path2

    checkpoint = torch.load(file_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    accuracy_list = checkpoint['accuracy_list']

    return model, optimizer, epoch, accuracy_list

def backprop(dataset, model, optimizer):
    total = 0
    for feat in dataset:
        graph, data, d = feat[0], feat[1], torch.tensor(feat[2],dtype=torch.float)
        y_pred = model(graph, data, d)
        y_true = feat[-1]
        optimizer.zero_grad()
        loss = torch.sum((y_pred - y_true) ** 2)
        optimizer.step()
        total += loss
    return total/len(dataset)

def accuracy(dataset, model):
	total = 0
	for feat in dataset:
		feature = feat[0]
		if not 'GNN' in model.name:
			feature = torch.tensor(feature,dtype=torch.float)
			y_pred = model(feature)
		else:
			graph, data, d = feat[0], feat[1], torch.tensor(feat[2],dtype=torch.float)
			y_pred = model(graph, data, d)
		y_true = feat[-1]
		loss = torch.sum((y_pred - y_true) ** 2)
		total += loss
	return total/len(dataset)

if __name__ == "__main__":
    # When running directly ...
    # Using GPU
    device = torch.device("cuda")
    """
    data_type = argv[1] # can be 'energy', 'energy_latency', 'energy_latency2', 'energy_latencyGNN'
	# 'stochastic_energy_latency', 'stochastic_energy_latency2' + '_' + str(HOSTS)
	exec_type = argv[2] # can be 'train', ga', 'opt'"""

    model = energy_latencyGNN_50()
    optimizer = torch.optim.Adam(model.parameters() , lr=0.001, weight_decay=1e-5)

    dataset, dataset_size, _ = load_energy_latencyGNN_data(50)

    split = int(0.8 * dataset_size)

    EPOCHS = 50
    for epoch in range(EPOCHS):
        print('EPOCH', epoch)
        random.shuffle(dataset)
        trainset = dataset[:split]
        validation = dataset[split:]
        loss = backprop(trainset, model, optimizer)
        trainAcc, testAcc = float(loss.data), float(accuracy(validation, model).data)
        accuracy_list.append((testAcc, trainAcc))
        print("Loss on train, test =", trainAcc, testAcc)
        if epoch % 10 == 0:
            save_model(model, optimizer, epoch, accuracy_list)
    print ("The minimum loss on test set is ", str(min(accuracy_list)), " at epoch ", accuracy_list.index(min(accuracy_list)))

		#plot_accuracies(accuracy_list, data_type)

