# standard_model.pt, augmented_model.pt, downsized_model.pt
import cv2 as cv
import numpy as np
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import tv_tensors
from torchvision.transforms import v2
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import PIL
import random
import time
import math
from models import Our_Model
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt

# Seed random number generator
random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # Set device
#path = kagglehub.dataset_download("orvile/bone-fracture-dataset")


path = "./data"
normal_dataset = []
fracture_dataset = []
# !!! SET THIS EQUAL TO 1 IN ORDER TO TRAIN ON THE FULL DATASET!!!
divisor = 1
training_split = 0.7
validation_split = 0.15
test_split = 0.15

transform_base = v2.Compose([
    v2.Resize((512,512)),
    v2.Grayscale(1),
    v2.ConvertImageDtype(torch.float32)
])

for i in range(384 // divisor):
    try:
        normal_dataset.append([tv_tensors.Image(PIL.Image.open("./data/normal/" + str(i) + ".png")), 0])
    except:
        continue
for i in range(1999 // divisor):
    try:
        fracture_dataset.append([tv_tensors.Image(PIL.Image.open("./data/fracture/" + str(i) + ".png")), 1])
    except:
        continue

# SET MEAN AND STD
mean, std = 0.37908458, 0.30377366

# LOAD MODELS
k_size = 3
k_stride = 1
k_pad = 1
p_size = 2
p_stride = 2
p_pad = 0
i_size = 512
model_base = Our_Model(k_size=k_size, k_stride=k_stride, k_pad=k_pad,
                  p_size=p_size, p_stride=p_stride, p_pad=p_pad, i_size=i_size).to(device)
model_augmented = Our_Model(k_size=k_size, k_stride=k_stride, k_pad=k_pad,
                  p_size=p_size, p_stride=p_stride, p_pad=p_pad, i_size=i_size).to(device)
i_size = 224
model_downsized = Our_Model(k_size=k_size, k_stride=k_stride, k_pad=k_pad,
                  p_size=p_size, p_stride=p_stride, p_pad=p_pad, i_size=i_size).to(device)
model_base.load_state_dict(torch.load('./standard_model.pt', map_location=torch.device('cpu'), weights_only=True))
print('STANDARD MODEL LOADED')
model_augmented.load_state_dict(torch.load('./augmented_model.pt', map_location=torch.device('cpu'), weights_only=True))
print('AUGMENTED MODEL LOADED')
model_downsized.load_state_dict(torch.load('./downsized_model.pt', map_location=torch.device('cpu'), weights_only=True))
print('DOWNSIZED MODEL LOADED')

# ---------------------------------------------------------------------------------------------------------- #
# --------------------------------------- Deep Dream - GD on Image------------------------------------------ #

# classed_as_fracture = 0
# for i in range(100):
#     image = torch.randn(1, 1, 224, 224).to(device)
#     image = (image * 8 + 128) / 255  # background color = 128,128,128
#     print(image.shape)
#     output = model_downsized(image)
#     pred = (output>0.5).float()
#     if pred:
#         classed_as_fracture += 1
# print(classed_as_fracture)
# exit()
    
# # PERFORM GD ON THE IMAGE!!!
# def save_img(image, path):
#     # Push to CPU, convert from (1, 3, H, W) into (H, W, 3)
#     image = image[0].permute(1, 2, 0)
#     image = image.clamp(min=0, max=1)
#     image = (image * 255).cpu().detach().numpy().astype(np.uint8)
#     # opencv expects BGR (and not RGB) format
#     cv.imwrite(path, image[:, :, ::-1])

# def main(model, input_size):
#     print(model)
#     for label in [0, 1]:
#         image = torch.randn(1, input_size, input_size, 1).to(device)
#         image = (image * 8 + 128) / 255  # background color = 128,128,128
#         image = image.permute(0, 3, 1, 2)
#         save_img(image, f"./random_{label}.jpg")
#         image.requires_grad_()
#         image = gradient_descent(image, model, lambda tensor: tensor[0].mean(), input_size=input_size, label=label)
#         save_img(image, f"./img_{label}.jpg")
#         print(image)
#         out = model(image)
#         print(f"ANSWER_FOR_LABEL_{label}: {out}")
#         # print(f"ANSWER_FOR_LABEL_{label}: {out.softmax(1)[0, label].item()}")

# def normalize_and_jitter(img, step=32):
#     # You should use this as data augmentation and normalization,
#     # convnets expect values to be mean 0 and std 1
#     dx, dy = np.random.randint(-step, step - 1, 2)
#     return transforms.Normalize(mean, std)(
#         img.roll(dx, -1).roll(dy, -2)
#     )

# def gradient_descent(input, model, loss, input_size=224, iterations=200, label=0):
#     model.eval()
#     alpha = 0.5
#     lmda = 0.0001
#     input.requires_grad_()
#     gaussian_blur = transforms.GaussianBlur(kernel_size=(3,3), sigma=(0.2, 1.5))
#     for iter in range(iterations):
#         jittered = torch.clamp(normalize_and_jitter(input), min=0.0, max=1.0)
#         blurred = gaussian_blur(jittered)
#         preds = model(blurred)  # The output of the model is the probability that the image is 1
#         if label:
#             logits = torch.log(preds + 0.000000001) - torch.log(1-preds + 0.000000001)  # Convert prob 1 (fracture) to logits
#         else:
#             logits = torch.log(1-preds + 0.000000001) - torch.log(preds + 0.000000001)  # Convert prob 0 (normal) to logits
#         F = loss(logits)
#         print('ITER:', iter, 'label:', label, 'preds:', (torch.abs(1-preds + label)), "logits:", logits[0])
#         # time.sleep(0.2)
#         F.backward()
#         print(torch.max(input.grad))
#         # print(input.grad)
#         with torch.no_grad():
#             input += alpha * gaussian_blur(input.grad) + lmda * input   # Second part is regularization
#         input.grad.zero_()
#         # input.data = torch.clamp(input, min=0.0, max=1.0)
#     image = torch.clamp(input, min=0.0, max=1.0)
#     return image

# # main(model_downsized, input_size=224)   # Not enough pixels for meaningful image, looks at very abstract things
# main(model_base, input_size=512)

# ---------------------------------------------------------------------------------------------------------- #
# ----------------------------------------- SHOW IMAGES AND LAYERS ----------------------------------------- #

def show_image(img, title=None):
    plt.imshow(img.detach().cpu().permute(1, 2, 0))
    plt.title(title)
    plt.axis('off')
    plt.show()
    


image = transform_base(fracture_dataset[0][0]).to(torch.float32).to(device)
# image = (image * 8 + 128) / 255
show_image(image)

def plot_intermediate_output(result, title):
    n_filters = result.shape[1]
    N = int(math.sqrt(n_filters))
    M = (n_filters + N - 1) // N
    assert N * M >= n_filters

    fig, axs = plt.subplots(N, M)
    fig.suptitle(title)

    for i in range(N):
        for j in range(M):
            if i*N + j < n_filters:
                axs[i][j].imshow(result[0, i*N + j].cpu().detach())
                axs[i][j].axis('off')
    plt.show()

# pick a few intermediate representations from your network and plot them using
# the provided function.
print(image)
shape = image.size()
print(shape)
clipped = nn.Sequential(*list(model_base.children())[:1])
intermediate_output = clipped(image[0].expand(1, shape[0], shape[1], shape[2]).to(device))
plot_intermediate_output(intermediate_output, 'After Layer 1')