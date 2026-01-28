import torch

print("CUDA available:", torch.cuda.is_available())

try:
    x = torch.randn(1).cuda()
    print("Successfully allocated tensor on CUDA:", x)
except Exception as e:
    print("CUDA error:", e)

# from torch.utils.tensorboard import SummaryWriter
# import gc, weakref

# for obj in gc.get_objects():
#     if isinstance(obj, SummaryWriter):
#         obj.close()
