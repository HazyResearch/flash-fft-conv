import torch
from tqdm.auto import tqdm
from tabulate import tabulate
import csv

from benchmark import benchmark_forward, benchmark_backward, benchmark_memory

from flashfftconv import FlashFFTConv

def ref_fftconv(u, k, N):
    L = u.shape[-1]
    u_f = torch.fft.fft(u.float(), n = N)
    k_f = torch.fft.fft(k.float(), n = N)
    y_f = u_f * k_f
    y = torch.fft.ifft(y_f, n = N).real[..., :L].to(u.dtype).contiguous()
    return y

def ref_fftconv_gated(u, k, N, pregate, postgate):
    L = u.shape[-1]
    u_gate = pregate.float() * u.float()
    u_f = torch.fft.fft(u_gate, n = N)
    k_f = torch.fft.fft(k.float(), n = N)
    y_f = u_f * k_f
    y = torch.fft.ifft(y_f, n = N).real[..., :L] * postgate.float()
    y = y.to(u.dtype).contiguous()
    return y

def set_B_H(B, H, seqlen):
    if seqlen == 16384 and B > 32:
        B = 32
    if seqlen == 32768 and B > 16:
        B = 16
    if seqlen == 65536 and B > 8:
        B = 8
    if seqlen == 131072 and B > 8:
        B = 8
    if seqlen == 131072 and H > 384:
        H = 384
    if seqlen == 262144 and B > 8:
        B = 8
    if seqlen == 262144 and H > 192:
        H = 192
    if seqlen == 524288 and B > 8:
        B = 8
    if seqlen == 524288 and H > 96:
        H = 96
    if seqlen == 1048576 and B > 8:
        B = 8
    if seqlen == 1048576 and H > 48:
        H = 48
    if seqlen == 2097152 and B > 8:
        B = 8
    if seqlen == 2097152 and H > 32:
        H = 32
    if seqlen == 4194304 and B > 8:
        B = 8
    if seqlen == 4194304 and H > 16:
        H = 16
    return B, H

def set_repeats(seqlen):
    if seqlen <= 4096:
        return 20
    elif seqlen <= 32 * 32768:
        return 10
    else:
        return 5

save_filename = 'benchmark_results.csv'

B = 64
H = 768
total_seqs = B * H
dtype = torch.float16
device = 'cuda'
seqlens = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 32 * 32768, 64 * 32768, 128 * 32768]
benchmark_fns = ['forward', 'backward', 'memory']
benchmark_fn_mapping = {
    'forward': benchmark_forward,
    'backward': benchmark_backward,
    'memory': benchmark_memory,
}
funcs = ['conv', 'gated conv', 'padded conv', 'gated padded conv']
write_tex = True

flash_vals = {}
ref_vals = {}
savings_vals = {}
keys = []

if 128 * 32768 in seqlens:
    print('Warning: 4M sequence length only supported for 80GB memory, manually reduce the batch and head size by changing the set_B_H function for smaller GPUs!')

for func in funcs:
    for benchmark_fn_name in benchmark_fns:
        torch.cuda.empty_cache()
        print(f'Benchmarking {benchmark_fn_name} for {func} with seqlens {seqlens}')

        flash = []
        ref = []
        savings = []
        padded = 'padded' in func
        gated = 'gated' in func
        print(padded, gated)
        for seqlen in tqdm(seqlens):
            N = seqlen if not padded else seqlen // 2

            local_B, local_H = set_B_H(B, H, seqlen)
            repeats = set_repeats(seqlen)
            
            adjustment = total_seqs / (local_B * local_H)

            u = torch.randn(local_B, local_H, N, dtype=dtype).to(device)
            k = torch.randn(local_H, N, dtype=torch.float32).to(device)
            if gated:
                pregate = torch.randn(local_B, local_H, N, dtype=dtype).to(device)
                postgate = torch.randn(local_B, local_H, N, dtype=dtype).to(device)
            
            fftconv = FlashFFTConv(seqlen, dtype=dtype).to(device)
            benchmark_fn = benchmark_fn_mapping[benchmark_fn_name]

            u.requires_grad = True
            k.requires_grad = True
            if gated:
                pregate.requires_grad = True
                postgate.requires_grad = True

            if benchmark_fn_name in ['forward', 'backward']:
                if gated:
                    t, m = benchmark_fn(fftconv, u, k, pregate, postgate, repeats=repeats, desc=f'Flash FFT Conv, {seqlen}', verbose=False)
                else:
                    t, m = benchmark_fn(fftconv, u, k, repeats=repeats, desc=f'Flash FFT Conv, {seqlen}', verbose=False)
                flash.append(m.mean * 1000 * adjustment)
            else:
                if gated:
                    m = benchmark_fn(fftconv, u, k, pregate, postgate, desc=f'Flash FFT Conv, {seqlen}', verbose=False)
                else:
                    m = benchmark_fn(fftconv, u, k, desc=f'Flash FFT Conv, {seqlen}', verbose=False)
                flash.append(m * adjustment)
        
        for seqlen in tqdm(seqlens):
            N = seqlen if not padded else seqlen // 2

            local_B, local_H = set_B_H(B, H, seqlen)
            repeats = set_repeats(seqlen)
            
            adjustment = total_seqs / (local_B * local_H)

            u = torch.randn(local_B, local_H, N, dtype=dtype).to(device)
            k = torch.randn(local_H, N, dtype=torch.float32).to(device)
            if gated:
                pregate = torch.randn(local_B, local_H, N, dtype=dtype).to(device)
                postgate = torch.randn(local_B, local_H, N, dtype=dtype).to(device)

            u.requires_grad = True
            k.requires_grad = True
            if gated:
                pregate.requires_grad = True
                postgate.requires_grad = True

            if benchmark_fn_name in ['forward', 'backward']:
                if gated:
                    t, m = benchmark_fn(ref_fftconv_gated, u, k, seqlen, pregate, postgate, repeats=repeats, desc=f'Ref FFT Conv, {seqlen}', verbose=False)
                else:
                    t, m = benchmark_fn(ref_fftconv, u, k, seqlen, repeats=repeats, desc=f'Ref FFT Conv, {seqlen}', verbose=False)
                ref.append(m.mean * 1000 * adjustment)
            else:
                if gated:
                    m = benchmark_fn(ref_fftconv_gated, u, k, seqlen, pregate, postgate, desc=f'Ref FFT Conv, {seqlen}', verbose=False)
                else:
                    m = benchmark_fn(ref_fftconv, u, k, seqlen, desc=f'Ref FFT Conv, {seqlen}', verbose=False)
                ref.append(m * adjustment)

            savings = [ r / f for r, f in zip(ref, flash)]
        print('Flash', flash)
        print('Ref', ref)
        print('Savings', savings)
        flash_vals[(func, benchmark_fn_name)] = flash
        ref_vals[(func, benchmark_fn_name)] = ref
        savings_vals[(func, benchmark_fn_name)] = savings
        keys.append((func, benchmark_fn_name))

print('Seqlens:', seqlens)
print('Flash:', flash_vals)
print('Ref:', ref_vals)
print('Savings:' , savings_vals)

table = [
    ['Method'] + seqlens
]
for k in keys:
    table.append([f'{k[0]}, {k[1]}, FlashFFTConv'] + flash_vals[k])
    table.append([f'{k[0]}, {k[1]}, PyTorch'] + ref_vals[k])
    table.append([f'{k[0]}, {k[1]}, Savings'] + savings_vals[k])

print(f'Saving results as {save_filename}')
with open(save_filename, 'w') as f:
    writer = csv.writer(f)
    for row in table:
        writer.writerows(table)

print(tabulate(table))

if write_tex:
    for k in keys:
        header = [['\\textbf{Seq Len}', '\\textbf{PyTorch}', '\\textbf{\\sysname}', '\\textbf{Memory Reduction}' if 'memory' in k[1] else '\\textbf{Speedup}']]
        table_data = header + [
            ['\\textbf{' + str(seqlen) + '}', '%1.2f' % ref_vals[k][i], '%1.2f' % flash_vals[k][i], '%1.2f$\\times$' % savings_vals[k][i]]
            for i, seqlen in enumerate(seqlens)
        ]
        latex_table = tabulate(table_data, tablefmt='latex_raw')
        print(f'{k[0]}, {k[1]}, LaTex')
        print(latex_table) 