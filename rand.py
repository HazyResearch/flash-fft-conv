from flashfftconv import FlashDepthWiseConv1d
import torch.nn as nn
import torch

conv1d_torch = nn.Conv1d(
    in_channels=512*3,
    out_channels=512*3,
    kernel_size=3,
    groups=512*3,
    padding=2,
    dtype=torch.float32
).cuda()

flash_conv1d = FlashDepthWiseConv1d(
    channels=512*3,
    kernel_size=3,
    padding=1,
    weights=conv1d_torch.weight,
    bias=conv1d_torch.bias,
    dtype=torch.float32
).cuda()

x = torch.rand(1, 1536, 2048, requires_grad=True).cuda()
y = torch.rand(1, 1536, 2048, requires_grad=True).cuda()

out_torch = conv1d_torch(x) 
out_flash = flash_conv1d(x)

criterion = nn.MSELoss().cuda()
optimizer = torch.optim.AdamW(flash_conv1d.parameters())
scaler = torch.cuda.amp.GradScaler()

with torch.autocast(device_type='cuda', dtype=torch.float16):
    optimizer.zero_grad()
    logits = flash_conv1d(x)
    loss = criterion(logits, y)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()