import math
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from dataset import DataHandler
import matplotlib.pyplot as plt

class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))




class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    """

    def __init__(self, n_head, n_embd, block_size):
        super().__init__()
        self.n_head = n_head
        self.n_embd = n_embd
        self.block_size = block_size
        # TIP: 
        # It is common practive to initialze a single Linear layer to map each token to its query, key, and value, i.e. nn.Linear(self.n_embd, 3 * self.n_embd)
        self.Linear = nn.Linear(self.n_embd, 3 * self.n_embd)

        self.fc = nn.Linear(self.n_embd, self.n_embd)
        # After applying the linear layer on a token embedding you can split the layer's output to key, query, and value
        # The output key/query/value is of dimension n_embd, in practice this includes the embeddings for all heads,
        # therefore, embedding = [embd_1, embd_2, .. embd_nheads]. You can rearange as you please in the forward pass.

    def forward(self, x):
        B, t = x.size(0), x.size(1)
        q, k, v = self.Linear(x).chunk(3, dim=-1)

        d_h = self.n_embd // self.n_head
        q = q.view(B, t, self.n_head, d_h).transpose(1, 2)
        k = k.view(B, t, self.n_head, d_h).transpose(1, 2)
        v = v.view(B, t, self.n_head, d_h).transpose(1, 2)

        score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_h)

        mask = torch.tril(torch.ones(t, t)).view(1, 1, t, t).to(x.device)
        score = score.masked_fill(mask == 0, float('-inf'))

        score = torch.nn.functional.softmax(score, dim=-1)
        score = torch.matmul(score, v)

        score = score.transpose(1, 2).contiguous().view(B, t, self.n_embd)
        return self.fc(score)
        

class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, n_head, n_embd, block_size):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_head, n_embd, block_size)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(n_embd, 4 * n_embd),
            c_proj  = nn.Linear(4 * n_embd, n_embd),
            act     = NewGELU(),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.c_proj(m.act(m.c_fc(x))) # MLP forward

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))
        return x

class GPT(nn.Module):
    """ GPT Language Model """


    def __init__(self, n_layer, n_head, n_embd, vocab_size, block_size):
        super().__init__()

        self.block_size = block_size
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(vocab_size, self.n_embd),
            wpe = nn.Embedding(block_size, self.n_embd),            
            h = nn.ModuleList([Block(n_head, n_embd, block_size) for _ in range(self.n_layer)]),
            ln_f = nn.LayerNorm(self.n_embd),
        ))
        self.lm_head = nn.Linear(self.n_embd, self.vocab_size, bias=False)



    def forward(self, idx):
        device = idx.device
        b, t = idx.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
        
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        return logits





def train_model(
        train_path,
        test_path=None,
        model=None,                        
        block_size=10,
        n_layer=3,
        n_head=3,
        n_embd=48,
        learning_rate=3e-4,
        batch_size=64,
        epochs=10
):            
                    
    
    data_handler = DataHandler(train_path, test_path, block_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vocab_size = data_handler.get_vocab_size()
    if model is None:
        model = GPT(n_layer, n_head, n_embd, vocab_size, block_size)

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    criterion = nn.CrossEntropyLoss()
    print('Using device:', device)


    trainset = data_handler.get_dataset('train')
    testset = data_handler.get_dataset('test')
    
    # setup the dataloader
    train_loader = DataLoader(
        trainset,
        sampler=torch.utils.data.RandomSampler(trainset, replacement=True, num_samples=int(1e5)),
        shuffle=False,
        pin_memory=True,
        batch_size=batch_size,        
    )     
    if testset:       
        test_loader = DataLoader(
            testset,
            sampler=torch.utils.data.RandomSampler(testset, replacement=False, num_samples=int(1e4)),
            shuffle=False,
            pin_memory=True,
            batch_size=batch_size,            
        )
    train_losses = []
    test_losses = []
    train_accuracy = []
    test_accuracy = []
    santences = []
    k_sentences = []
    for ep in range(epochs):
        train_loss = 0
        test_loss = 0
        train_acc=0
        test_acc=0
        model.train()
        for i, batch in enumerate(tqdm(train_loader)):            
            optimizer.zero_grad()
            x, y = batch
            x = x.to(device)
            y = y.to(device).long()
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            train_acc += (logits.argmax(-1) == y).sum().item()/y.numel()

        with torch.no_grad():
            model.eval()
            for i, batch in enumerate(tqdm(test_loader)):
                x, y = batch
                x = x.to(device)
                y = y.to(device).long()
                logits = model(x)
                loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                test_loss+=loss.item()
                test_acc+=(logits.argmax(-1) == y).sum().item()/y.numel()
        print(f'Test loss: {test_loss/len(test_loader)}, Train loss: {train_loss/len(train_loader)}')
        print(f'Test accuracy: {test_acc/len(test_loader)}, Train accuracy: {train_acc/len(train_loader)}')
        train_losses.append(train_loss/len(train_loader))
        test_losses.append(test_loss/len(test_loader))
        train_accuracy.append(train_acc/len(train_loader))
        test_accuracy.append(test_acc/len(test_loader))



                

        # Complete the sentence:
        model.eval()
        sentence="the "
        for i in range(3):
            new_sentence = sentence
            for j in range(30):
                    tokens = torch.tensor(data_handler.encoder(new_sentence[-block_size:]))[None]
                    logits = model(tokens)
                    logits = logits[0, -1, :]
                    next_token = torch.multinomial(logits.softmax(dim=-1), num_samples=1)
                    new_sentence += data_handler.decoder([next_token.item()])
            santences.append(new_sentence)


        # Comple the sentence only considering the top k characters when sampling:
        for i in range(3):
            new_sentence = sentence
            for j in range(30):
                tokens = torch.tensor(data_handler.encoder(new_sentence[-block_size:]))[None]
                logits = model(tokens)
                logits = logits[0, -1, :]
                probs = torch.softmax(logits, dim=-1)
                _, topk = torch.topk(probs, 5)
                next_token = topk[torch.multinomial(probs[topk], num_samples=1)]
                new_sentence += data_handler.decoder([next_token.item()])
            k_sentences.append(new_sentence)
    print('Sentences:', santences)
    print('K sentences:', k_sentences)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'test_losses': test_losses,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'epoch': epochs
    }, 'transformer_checkpoint.pth')
    plt.plot(train_losses, label='train loss', color='blue')
    plt.plot(test_losses, label='test loss', color='red')
    plt.title('Loss')
    plt.legend()
    plt.show()
    plt.plot(train_accuracy, label='train accuracy', color='blue')
    plt.plot(test_accuracy, label='test accuracy', color='red')
    plt.title('Accuracy')
    plt.legend()
    plt.show()





if __name__=="__main__":
    torch.manual_seed(42)
    train_model('train_shakespeare.txt', 'test_shakespeare.txt')
    

