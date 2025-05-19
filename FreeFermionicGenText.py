
import torch
import tqdm
import math
import torch.utils.data

from sim import FFQuantumDevice

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
num_angles = num_layer * (2 * num_qubits - 1)
print("Layers :",num_layer)
print("Qubits :",num_qubits)
print("num_angles :",num_angles)

class QRNN(torch.nn.Module):
    def __init__(self, input_size, num_angles, num_qubits):
        super(QRNN, self).__init__()
        self.input_size = input_size
        self.num_angles = num_angles
        self.num_qubits = num_qubits
        self.L = torch.nn.Linear(self.input_size, self.num_angles)

    def forward(self, input, circuit:FFQuantumDevice):
        angles = self.L(input)
        ang = 0
        for _ in range(num_layer):
            circuit.rxx_layer(angles[:,ang:ang+self.num_qubits -1])
            ang += self.num_qubits  - 1
            circuit.rz_layer(angles[:,ang:ang+self.num_qubits ])
            ang += self.num_qubits 

        return circuit

class Classifier(torch.nn.Module):
    def __init__(self, num_class, num_qubits):
        super().__init__()
        self.num_class = num_class
        self.num_qubits = num_qubits
        self.Lclass = torch.nn.Linear(self.num_qubits, self.num_class)

    def forward(self, circuit:FFQuantumDevice):
        meas = circuit.z_exp_all()
        char_pred = self.Lclass(meas)
        return char_pred

qrnn = QRNN(vocab_size, num_angles, num_qubits)
classifier = Classifier(vocab_size,num_qubits)

def eval_model(window):
    hidden = FFQuantumDevice(num_qubits, batch_size)
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