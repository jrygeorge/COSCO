# graph modifyier
# run the whole thing
# selector
# re-optimizer
"""
This is the file containing the eXplainable Graph NN Scheduler.




"""

import sys
#sys.path.append('scheduler/BaGTI/')

from Scheduler import Scheduler
from .BaGTI.train import *
from .BaGTI.src.utils import *
from .BaGTI.src.opt import *
import dgl

def convertToOneHot(dat, cpu_old, HOSTS):
    alloc = []
    for i in dat:
        oneHot = [0] * HOSTS; alist = i.tolist()[-HOSTS:]
        oneHot[alist.index(max(alist))] = 1; alloc.append(oneHot)
    new_dat_oneHot = torch.cat((cpu_old, torch.FloatTensor(alloc)), dim=1)
    return new_dat_oneHot

def reoptimize(init, graph, data, model, bounds, data_type):
    HOSTS = int(data_type.split('_')[-1])
    optimizer = torch.optim.AdamW([init] , lr=0.8)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    iteration = 0; equal = 0; z_old = 100; zs = []
    while iteration < 200:
        cpu_old = deepcopy(init.data[:,0:-HOSTS]); alloc_old = deepcopy(init.data[:,-HOSTS:])
        z = model(graph, data, init)
        optimizer.zero_grad(); z.backward(); optimizer.step(); scheduler.step()
        init.data = convertToOneHot(init.data, cpu_old, HOSTS)
        equal = equal + 1 if torch.all(alloc_old.eq(init.data[:,-HOSTS:])) else 0
        if equal > 30: break
        iteration += 1; z_old = z.item()
    init.requires_grad = False 
    return init.data, iteration, model(graph, data, init)

def load_model(filename, model, data_type):
	optimizer = torch.optim.Adam(model.parameters() , lr=0.001, weight_decay=1e-5) if 'stochastic' not in data_type else torch.optim.AdamW(model.parameters() , lr=0.0001)
	file_path1 = MODEL_SAVE_PATH + "/" + filename + "_Trained.ckpt"
	file_path2 = 'scheduler/BaGTI/' + file_path1
	file_path = file_path1 if os.path.exists(file_path1) else file_path2
	if os.path.exists(file_path):
		print(color.GREEN+"Loading pre-trained model: "+filename+color.ENDC)
		checkpoint = torch.load(file_path)
		model.load_state_dict(checkpoint['model_state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		epoch = checkpoint['epoch']
		accuracy_list = checkpoint['accuracy_list']
	else:
		epoch = -1; accuracy_list = []
		print(color.GREEN+"Creating new model: "+model.name+color.ENDC)
	return model, optimizer, epoch, accuracy_list


class XGAIScheduler(Scheduler):
	def __init__(self, data_type):
		super().__init__()
		dtl = data_type.split('_')
		data_type = '_'.join(dtl[:-1])+'GNN_'+dtl[-1]
		self.model = eval(data_type+"()") # energy_latency_GNN50
		self.model, _, _, _ = load_model(data_type, self.model, data_type)
		self.data_type = data_type
		self.hosts = int(data_type.split('_')[-1])
		dtl = data_type.split('_')
		_, _, self.max_container_ips = eval("load_"+'_'.join(dtl[:-1])+"_data("+dtl[-1]+")")

	def generate_decision(self):
		# Get CPU metric
		cpu = [host.getCPU()/100 for host in self.env.hostlist]
		cpu = np.array([cpu]).T
		
		# Get "normalised" CPU metric of container
		cpuC = [(c.getApparentIPS()/self.max_container_ips if c else 0) for c in self.env.containerlist]
		cpuC = np.array([cpuC]).T
		
		cpu = np.concatenate((cpu, cpuC), axis=1)
		
		current_allocation = []
		previous_allocation = {}
		
		u, v = [], [] # edge goes from u[i] to v[i]
		for i,container in enumerate(self.env.containerlist):
			if container: # if container exists
				previous_allocation[container.id] = container.getHostID() # then save its current host
			if container and (container.getHostID() != -1): 
				current_allocation[i] = container.getHostID()
				# Creating Nodes
				u.append(container.id) # Container
				v.append(container.getHostID() + self.hosts) # Hosts
			else: 
				current_allocation[i] = np.random.randint(0,len(self.env.hostlist)) 
			
		# alloc is the one hotted
		empty = np.zeros((len(self.env.containerlist), len(self.env.hostlist)))
		empty[np.arange(current_allocation.size), current_allocation] = 1
		
		alloc = empty
		
		g = dgl.graph((u, v), num_nodes=2*self.hosts)
		g = dgl.add_self_loop(g)
		data = torch.Tensor(cpu.reshape(-1, 1))
		init = np.concatenate((cpu, alloc), axis=1)
		init = torch.tensor(init, dtype=torch.float, requires_grad=True)
		result , *_ = reoptimize(init, g, data, self.model, [], self.data_type)
		
		decision = []
		for cid in prev_alloc:
			one_hot = result[cid, -self.hosts:].tolist()
			new_host = one_hot.index(max(one_hot))
			if prev_alloc[cid] != new_host: 
				decision.append((cid, new_host))
		return decision

	def selection(self):
		return []

	def placement(self, containerIDs):
		first_alloc = np.all([not (c and c.getHostID() != -1) for c in self.env.containerlist])
		decision = self.generate_decision()
		return decision
