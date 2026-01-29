import torch
from models.scorebased_models.sde_diffusion import SDEDiffusion
print("Testing SDE Diffusion Model...")
device = "cpu"
print("\n1. Testing VP-SDE model instantiation...")
model_vp = SDEDiffusion(sde_type="vpsde", image_size=32, channels=64, ch_mult=(1, 2), num_res_blocks=1, N=100, device=device)
print(f"   VP-SDE model created with {sum(p.numel() for p in model_vp.parameters())} parameters")
print("\n2. Testing forward pass...")
x = torch.randn(2, 3, 32, 32).to(device)
t = torch.rand(2).to(device)
with torch.no_grad():
    output = model_vp(x, t)
print(f"   Input shape: {x.shape}")
print(f"   Output shape: {output.shape}")
print("\n3. Testing loss computation...")
loss, t_sampled = model_vp.compute_loss(x)
print(f"   Loss: {loss.item():.4f}")
print("\n4. Testing train step...")
losses = model_vp.train_step(x, epoch=0)
print(f"   Train step losses: {losses}")
print("\nAll tests passed!")
