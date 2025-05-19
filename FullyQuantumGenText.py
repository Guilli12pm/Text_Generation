
import torch
import tqdm
import math
import torch.utils.data
import torchquantum as tq


window_size = 32
batch_size = 128

NUM_LAYER = 3
NUM_QUBITS = 6
EPOCHS = 5

with open("input.txt", "r") as f:
    chars = list(f.read())
    lookup = { c: i for i, c in enumerate(set(chars)) }
    lookup_back = { i: c for c, i in lookup.items() }
    vocab_size = len(lookup)
    chars = torch.tensor([lookup[c] for c in chars])

windows = torch.stack([
    chars[i:-(window_size-i)] for i in range(window_size)
], dim=1)
targets = chars[window_size:]

split_idx = -1024
train_dataset = torch.utils.data.TensorDataset(windows[:split_idx], targets[:split_idx])
test_dataset = torch.utils.data.TensorDataset(windows[split_idx:], targets[split_idx:])
train = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True, drop_last=True)
test = torch.utils.data.DataLoader(test_dataset, batch_size, drop_last=True)

epochs = EPOCHS

num_layer = NUM_LAYER
num_qubits = NUM_QUBITS
num_angles = num_layer * (4 * num_qubits - 1)
print("Layers :",num_layer)
print("Qubits :",num_qubits)
print("num_angles :",num_angles)


def add_matchgate(qdev:tq.QuantumDevice, angles):
    #print(angles.shape)
    #print(angles.shape)
    ang = 0
    #print(angles[:, ang].shape)
    for i in range(num_qubits-1):
        qdev.rxx(params=angles[:, ang], wires=[i, i+1])
        ang += 1
        qdev.u3(params=angles[:, ang:ang+3], wires=i)
        ang += 3
    qdev.u3(params=angles[:, ang:ang+3], wires=num_qubits-1)

class QRNN(torch.nn.Module):
    def __init__(self, input_size, num_angles, num_qubits):
        super(QRNN, self).__init__()
        self.input_size = input_size
        self.num_angles = num_angles
        self.num_qubits = num_qubits
        self.L = torch.nn.Linear(self.input_size, self.num_angles)

    def forward(self, input, qdev:tq.QuantumDevice):
        angles = self.L(input)
        for lay in range(num_layer):
            add_matchgate(qdev,angles[:,lay * (4 * num_qubits - 1) : (lay+1) * (4 * num_qubits - 1)])
        return qdev

class Classifier(torch.nn.Module):
    def __init__(self, num_class, num_qubits):
        super().__init__()
        self.num_class = num_class
        self.num_qubits = num_qubits
        self.Lclass = torch.nn.Linear(self.num_qubits, self.num_class)

    def forward(self, qdev:tq.QuantumDevice):
        class_list = []
        for i in range(self.num_qubits):
            meas = "I"*i + "Z" + "I"*(self.num_qubits - i - 1)
            class_list.append(tq.measurement.expval_joint_analytical(qdev, meas))
        stack = torch.stack(class_list, dim=-1)
        char_pred = self.Lclass(stack)
        return char_pred

qrnn = QRNN(vocab_size, num_angles, num_qubits)
classifier = Classifier(vocab_size,num_qubits)

def eval_model(window):
    hidden = tq.QuantumDevice(num_qubits, bsz=batch_size)
    for i in range(window.shape[1]):
        value = torch.nn.functional.one_hot(window[:, i], vocab_size).to(torch.float)
        hidden = qrnn.forward(value,hidden)
    return classifier.forward(hidden)

def validate():
    total_loss = 0.0
    for window, target in test:
        with torch.no_grad():
            pred = eval_model(window)
            loss = torch.nn.functional.cross_entropy(pred, target)
            total_loss += loss.item()
    total_loss /= len(test)
    return total_loss

def sample():
    window, _ = test_dataset[0]
    chars = [lookup_back[window[i].item()] for i in range(window_size)]
    for _ in range(200):
        pred = eval_model(window[None, :])[0, :]
        index = torch.distributions.Categorical(logits=pred).sample().item()
        chars.append(lookup_back[index])
        window = torch.cat((window[1:], torch.tensor([index])))
    print("Sample:")
    print(''.join(chars))
    print()

sample()

optim = torch.optim.Adam(list(qrnn.parameters()) + list(classifier.parameters()), 0.001)
for epoch in range(epochs):
    print(f"epoch {epoch}:")
    pbar = tqdm.tqdm(train, dynamic_ncols=True, position=0)
    for i, (window, target) in enumerate(pbar):
        if i % 1000 == 0:
            test_ppl = math.exp(validate())
            pbar.set_description(f"test PPL = {test_ppl:g}")
        
        optim.zero_grad()
        pred = eval_model(window)
        loss = torch.nn.functional.cross_entropy(pred, target)
        loss.backward()
        optim.step()
    sample()