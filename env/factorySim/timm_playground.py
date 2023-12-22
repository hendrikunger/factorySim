#%%
import timm
import torch
# %%
timm.list_models()
# %%
model = timm.create_model('xception41', num_classes=10, in_chans=2, img_size=256)
#dir(model)
print(model.num_classes)
print(model.num_features)
# %%
x = torch.randn(4, 2, 224, 224)
model(x).shape

# %%
