import torch
import tqdm
import math
import torch.utils.data


window_size = 32
batch_size = 128

HIDDEN_SIZE = 64

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

epochs = 7

hidden_size = HIDDEN_SIZE
print("Hidden layer size = ",hidden_size)

step = torch.nn.Sequential(
    torch.nn.Linear(hidden_size + vocab_size, hidden_size),
    torch.nn.ReLU(),
    torch.nn.Linear(hidden_size, hidden_size)
)
output = torch.nn.Linear(hidden_size, vocab_size)

def eval_model(window):
    hidden = torch.zeros((window.shape[0], hidden_size))
    for i in range(window.shape[1]):
        value = torch.nn.functional.one_hot(window[:, i], vocab_size)
        hidden = step.forward(torch.cat((hidden, value), dim=1))
    return output.forward(hidden)

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
    final = ''.join(chars)
    print(final)
    
sample()

optim = torch.optim.Adam(list(step.parameters()) + list(output.parameters()), 0.001)
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