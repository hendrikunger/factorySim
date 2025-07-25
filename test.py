import torch

for i in range(torch.cuda.device_count()):
    print(i)
    for j in range(torch.cuda.device_count()):
        print(i,j)
        if i != j:
            print(f"P2P access from GPU {i} to {j}: {torch.cuda.can_device_access_peer(i, j)}")
