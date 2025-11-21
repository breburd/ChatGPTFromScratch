import torch
import numpy as np
from gpt import GPTModel
import matplotlib.pyplot as plt
import time


# since we didn't really cover how to do this in lecture-
# this creates a learning rate schedule for you. Refer to the 
# pytorch docs for more info on using a scheduler.

# This one is designed for you to call scheduler.step() on every
# model update step. 
def cosine_with_warmup_lr_scheduler(opt, total_steps, warmup_steps):
    def thunk(stepnum):
        if stepnum <= warmup_steps:
            # go from ~0 to 1.0
            prog = float(stepnum)/float(warmup_steps)
            lrmult = 0.00001 + prog
        else:
            # go from 1.0 to ~0
            steps_after_peak = stepnum-warmup_steps
            tail_steps = total_steps-warmup_steps
            prog = float(steps_after_peak) / float(tail_steps)
            lrmult = ((np.cos(3.141592*prog)+1.0)*0.5)*0.9 + 0.1
        return max(lrmult, 0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=thunk)
    return scheduler

# ===========================================================================


"""
Complete the following method which trains a GPT model and saves a loss curve.

To reiterate: you don't need to worry about weight decay, weight initialization, grad accumulation, or weight tying.
Use whatever batch size you are able, even something like 2 or 4 is fine.
Use a few hundred warmup steps and a peak learning rate that is (something x 10-4).
"""


def train(num_layers=4):

    device = torch.device("cpu") # use "cpu" if not gpu available

    # adjust as needed
    model = GPTModel(d_model=512, n_heads=8, layers=num_layers, vocab_size=10000, max_seq_len=256)
    param_count = sum(p.numel() for p in model.parameters())
    print("Model has", param_count, "parameters.")

    model = model.to(device)

    dataset = np.load('shuffled_tokens_old.npy', allow_pickle=True)

    opt = torch.optim.AdamW(model.parameters(), lr=2.0e-4)  # use learning rate from GPT-3 Small from slide 38
    scheduler = cosine_with_warmup_lr_scheduler(opt, len(dataset), 600)
    cross_entropy = torch.nn.CrossEntropyLoss()

    BATCH_SIZE = 4
    losses = []
    total_tokens = 0
    token_counts = []

    start = time.perf_counter()
    for i in range(0, len(dataset), BATCH_SIZE):
        batch = dataset[i:i + BATCH_SIZE]
        # using code from slide 62
        inputs = batch[:, 0:-1]  # first S tokens, size (B, S)
        target = batch[:, 1:]  # last S tokens, size (B, S)
        # convert to torch Tensors
        inputs = torch.tensor(inputs, dtype=torch.long, device=device)
        target = torch.tensor(target, dtype=torch.long, device=device)

        opt.zero_grad()

        # <run batch through model and compute loss>
        pred = model.forward(inputs)
        # VERY IMPORTANT ---
        # torch.nn.CrossEntropyLoss expects classes in the 2nd dimension.
        # You may need to transpose dimensions around to make this work.
        # It also expects unnormalized logits (no softmax).
        # ---
        pred = pred.transpose(1, 2)  # switch vocab_size and S dimensions
        loss = cross_entropy(pred, target)

        loss.backward()

        # clip the gradient
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # step the optimizer and scheduler
        opt.step()
        scheduler.step() 

        # log total tokens and loss
        total_tokens += inputs.numel()
        # periodically save a plot of loss vs tokens
        if i % 1000 == 0:
            end = time.perf_counter()

            print(f"{end - start:.4f} seconds")
            token_counts.append(total_tokens)
            losses.append(loss.item())
            print(f"Step {i}: Tokens={token_counts[-1]}, Loss={losses[-1]:.4f}")
            print(f"\t{end - start:.4f} seconds")
            save_loss_curve(token_counts, losses, f'{len(model.layers)}Layers', f'step_{i}')
            torch.save(model.state_dict(), f"./{len(model.layers)}Layers/model_weights_step_{i}.pt")

    # save model weights if you want
    save_loss_curve(token_counts, losses, f"{len(model.layers)}Layers")
    torch.save(model.state_dict(), f"./{len(model.layers)}Layers/model_weights_Overall.pt")


def save_loss_curve(token_counts, losses, prefix, suffix='Overall'):
    # Save loss curve
    plt.figure(figsize=(8, 5))
    plt.plot(token_counts, losses)
    plt.xlabel("Tokens Seen")
    plt.ylabel("Loss")
    plt.title("Training Loss vs Tokens")
    plt.grid(True)
    plt.savefig(f"./{prefix}/loss_curve_{suffix}.png")
    plt.close()
    np.save(f'./{prefix}/losses_{suffix}.npy', losses)
    np.save(f'./{prefix}/token_counts_{suffix}.npy', token_counts)

if __name__ == "__main__":
    train()

"""
References
    Lecture notes for Module 6
    https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html -> documentaiton for AdamW
    https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html -> documentation for CrossEntropyLoss
    
Model Parameters
    GPT Model
        d_model = 512 -> This was unchanged from the provided code
        n_heads = 8 -> This was decreased from the provided 16
        layers = 4 -> This was decreased to make the model smaller and have less Transformer layers
        vocab_size = 10000 -> Unchanged from the provided code
        max_seq_len = 256 -> Unchanged from the provided code
    Optimizer
        Learning Rate = 2.0e-4 -> I tested with several values within slide 38, but found this to be the best 
        out of the options I tried
    Scheduler
        Total Steps = len(dataset) -> The length of the dataset constructed in construct_dataset.py
        Warmup Steps = 600 -> This was arbitrary and I tested with several other values ranging from 300-700
    BATCH_SIZE = 4
"""