from visualization.plotting import plt

import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from functools import partial

from visualization.warhol_colormap import warhol, get_warhol_colormap, plot_colormap

transform = transforms.Compose([
    transforms.Resize((1008, 1344)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])  #

device = 'cuda'

image_path = "./data/images/chinese_garden.jpg"  # Replace with the actual path to your image
image = Image.open(image_path).convert('RGB')
input_tensor = transform(image).unsqueeze(0)  # Add batch dimension
input_tensor = input_tensor.to(device)


model = torchvision.models.vgg16(pretrained=True)
model.to(device)
model.eval()

outputs = []
names = []
def hook_fn(module, input, output, name: str = ''):  # noqa
    outputs.append(output)
    names.append(f"{str(module)} - {name}")  # 1.6.3


for name, module in model.named_modules():
    #  if isinstance(module, torch.nn.Conv2d):
    if isinstance(module, torch.nn.ReLU):
        hook = partial(hook_fn, name=name)
        module.register_forward_hook(hook)  # noqa

# 5. Perform a forward pass to trigger the hooks and collect feature maps
with torch.no_grad():
    _ = model(input_tensor)


#%%

warhol = get_warhol_colormap(n_colors=16)
plot_colormap(warhol, figure=len(outputs) + 2)

#%%


def process_image(img):
    img_max, img_min = np.percentile(img, (90, 0))
    # img_max, img_min = 1, 0 # np.percentile(img, (90, 10))
    if img_max == img_min:
        img_max = img_min + 1
    normalized_img = (img - img_min) / (img_max - img_min)
    normalized_img = np.clip(normalized_img, 0, 1)

    # rgb = np.transpose(normalized_feature_map[3 * j:3 * j + 3], (1, 2, 0))
    # for c in range(3):
    #    rgb[:, :, c] = rgb[:, :, c] / rgb[:, :, c].max()

    return normalized_img


for i, feature_map in enumerate(outputs):
    feature_map_np = feature_map.squeeze(0).cpu().numpy()  # Remove batch dimension, move to CPU, convert to NumPy

    max_n_plots = 8 * 8
    n_plots = min(len(feature_map_np), max_n_plots)

    n_rows = int(np.rint(np.sqrt(n_plots)))
    n_cols = n_plots // n_rows + (n_plots % n_rows > 0)

    plt.figure(i+1, figsize=(10, 10))
    plt.clf()

    for j in range(n_plots):
        plt.subplot(n_cols, n_rows, j + 1)  # Create a grid for plotting

        img = process_image(feature_map_np[j])

        plt.imshow(img, cmap=warhol)

        plt.axis('off')
        plt.title(f'Channel {j}')

    plt.suptitle(f'Feature Maps from Layer {i}: {names[i]} (total: {len(feature_map_np)})')
    plt.tight_layout()
    plt.show()


plt.figure(len(outputs) + 1)
plt.imshow(image)


#%% convert video

from visualization.plotting import plt

import numpy as np
import cv2
import torch
import torchvision
import torchvision.transforms as transforms
import functools as ft

from visualization.warhol_colormap import get_warhol_colormap, plot_colormap

from visualization.colors import (
    sort_colors, sort_colors_nn, step,
    luminosity,  hsv, hls,
    luminosity_r, hsv_r, hls_r
)

# video_file = './data/videos/shanghai.mp4'
video_file = './data/videos/zoomed.mp4'

locations = [(0, 8), (1, 9), (0, 47), (2, 41), (4, 19), (10, 17), (11, 25), (12, 2), (12, 0)]
# (7, 10),  (5, 9),  (8, 34), (2, 17),
n_features = 9
n_features_width = 3
n_features_height = 3
n_frames_per_chunk = 100
device = 'cuda'

video_file_ai = video_file[:-4] + '_ai' + video_file[-4:]

video = cv2.VideoCapture(video_file)
n_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
fps = int(video.get(cv2.CAP_PROP_FPS))
frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

frame_width = n_features_width * frame_width
frame_height = n_features_height * frame_height

feature_width = frame_width // n_features_width
feature_height = frame_height // n_features_height

model = torchvision.models.vgg16(pretrained=True)
model.to(device)
model.eval()

resize = transforms.Resize((feature_height, feature_width))

warhol = get_warhol_colormap(n_colors=32)

n_colors = [8, 9, 10,
            64, 16, 8,
            16, 5, 32]

color_sorts = [
    ft.partial(sort_colors, key=luminosity),
    ft.partial(sort_colors, key=luminosity_r),
    ft.partial(sort_colors, key=luminosity),
    ft.partial(sort_colors, key=hsv),
    ft.partial(sort_colors, key=hls),
    ft.partial(sort_colors, key=hls_r),
    sort_colors_nn,
    ft.partial(sort_colors, key=ft.partial(step, repetitions=5)),
    ft.partial(sort_colors, key=luminosity)
]

colormaps = {i: get_warhol_colormap(n_colors=n_colors[i], sort=color_sorts[i]) for i in range(len(n_colors))}


def process_image(img, i):
    img_max, img_min = np.percentile(img, (95, 15))
    # img_max, img_min = 1, 0 # np.percentile(img, (90, 10))
    if img_max == img_min:
        img_max = img_min + 1
    normalized_img = (img - img_min) / (img_max - img_min)
    normalized_img = np.clip(normalized_img, 0, 1)

    rgb = colormaps[i](normalized_img)[:, :, :3]

    return rgb


features = []
frames_out = []
def hook_fn(module, input, output, index: int = 0, last: bool = False, name: str = ''):  # noqa
    # print(f'{name}')
    features.append(resize(output[:, index]).cpu().numpy())
    # shape: B x H x W
    # len: F

    if last:
        B = len(features[0])

        for b in range(B):
            image_full = np.zeros((frame_height, frame_width, 3))
            image_features = [process_image(feature[b], i) for i, feature in enumerate(features)]
            k = 0
            for y in range(n_features_height):
                for x in range(n_features_width):
                    sx, sy = x * feature_width, y * feature_height
                    ex, ey = sx + feature_width, sy + feature_height
                    image_full[sy:ey, sx:ex, :] = image_features[k]
                    k += 1

            frames_out.append(image_full)


layer = 0
n_locations = 0
layers = [l[0] for l in locations]
indices = [l[1] for l in locations]

for name, module in model.named_modules():
    if isinstance(module, torch.nn.ReLU):
        for loc in locations:
            if layer == loc[0]:
                n_locations += 1
                index = loc[1]
                last = n_locations == n_features
                hook = ft.partial(hook_fn, index=index, last=last, name=name)
                module.register_forward_hook(hook)  # noqa
                print(f'registering: {name} {layer=} {loc=} {index=} {n_locations=} {last=}')
        layer += 1

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((frame_height, frame_width)),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])  #

#%%

video = cv2.VideoCapture(video_file)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_out = cv2.VideoWriter(video_file_ai, fourcc, fps, (frame_width, frame_height))

n_frames_per_chunk = 20
if n_frames_per_chunk is None:
    n_frames_per_chunk = n_frames
n_chunks = int(np.ceil(n_frames / n_frames_per_chunk))

import tqdm
for c in tqdm.tqdm(range(n_chunks)):
    chunk_size = n_frames_per_chunk if c < n_chunks - 1 else \
        n_frames - (n_frames // n_frames_per_chunk) * n_frames_per_chunk

    frames = torch.stack([transform(video.read()[1]) for _ in range(chunk_size)]).to(device=device)

    features = []
    frames_out = []

    with torch.no_grad():
        _ = model(frames)

    [video_out.write(cv2.cvtColor(np.asarray(frame * 255, dtype='uint8'), cv2.COLOR_RGB2BGR)) for frame in frames_out]

video_out.release()

#%% test

video_test = cv2.VideoCapture(video_file)

n_test = 15
frames = torch.stack([transform(video_test.read()[1]) for _ in range(n_test)]).to(device=device)

features = []
frames_out = []
with torch.no_grad():
    _ = model(frames)

plt.figure(20)
plt.clf()
plt.subplot(4, 4, 1)
plt.axis('off')
plt.imshow(frames[0].cpu().numpy().transpose((1, 2, 0)))
for k in range(15):
    plt.subplot(4, 4, 2 + k)
    plt.imshow(np.asarray(frames_out[k] * 255, dtype='uint8'))
    plt.axis('off')
    plt.tight_layout()
    plt.show()
