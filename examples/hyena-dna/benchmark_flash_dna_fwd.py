import torch
from benchmark import benchmark_forward, pytorch_profiler

from huggingface import load_model
import sys

'''
model options:
    'hyenadna-tiny-1k-seqlen'   # fine-tune on colab ok
    'hyenadna-tiny-1k-seqlen-d256'
    'hyenadna-tiny-16k-seqlen-d128'
    'hyenadna-small-32k-seqlen'
    'hyenadna-medium-160k-seqlen'  # inference only on colab
    'hyenadna-medium-450k-seqlen'  # inference only on colab
    'hyenadna-large-1m-seqlen'  # inference only on colab
'''

model_name = 'hyenadna-large-1m-seqlen'
B = 4
repeats = 10
use_flash = True

model, tokenizer, max_length = load_model(model_name, use_flash=use_flash)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#### Single embedding example ####

# create a sample 450k long, prepare
sequence = 'ACTG' * int(max_length/4)
tok_seq = tokenizer(sequence)
tok_seq = tok_seq["input_ids"]  # grab ids

# place on device, convert to tensor
tok_seq = torch.LongTensor(tok_seq).repeat(B, 1)  # unsqueeze for batch dim
tok_seq = tok_seq.to(device)

# prep model and forward
model.to(device)
model = model.half()
model.eval()

def run_model(model, tok_seq):
    return model(tok_seq)

with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
    with torch.no_grad():
        run_model(model, tok_seq)

torch.cuda.empty_cache()

with torch.no_grad():
    _, ret = benchmark_forward(run_model, model, tok_seq, repeats=repeats, verbose=True, amp_dtype=torch.float16, amp=True)

time = ret._mean
print('Time: ', time)
print('Tokens/ms: ', (tok_seq.shape[0] * tok_seq.shape[1])/time/1000)
print('Seqs/s: ', B/time)

# pytorch_profiler(run_model, model, tok_seq, backward=False, cpu=True, trace_filename=f'dna_fwd_{model_name}_flash_{use_flash}.json')