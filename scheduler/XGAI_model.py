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


from torch import nn

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

    def _LayerNormGRUCell(self, input, hidden, w_ih, w_hh, ln, b_ih=None, b_hh=None):
    	
	    gi = F.linear(input, w_ih, b_ih)
	    gh = F.linear(hidden, w_hh, b_hh)
	    i_r, i_i, i_n = gi.chunk(3, 1)
	    h_r, h_i, h_n = gh.chunk(3, 1)

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
        if not('train' in argv[0] and 'train' in argv[2]):
            x = Coeff_Energy*x[0] + Coeff_Latency*x[1]
        return x

if __name__ == "__main__":
    # When running directly ...
    1
    
