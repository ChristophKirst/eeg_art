#%% convert video
import os.path

from visualization.plotting import plt

import numpy as np
import cv2
import torch
import torchvision
import torchvision.transforms as transforms
import functools as ft

# from visualization.warhol_colormap import get_warhol_colormap, plot_colormap

from visualization.dominant_colormap import get_dominant_colormap, plot_colormap

from visualization.colors import (
    sort_colors, sort_colors_nn, step,
    luminosity,  hsv, hls,
    luminosity_r, hsv_r, hls_r
)

# video_file = './data/videos/shanghai.mp4'
# video_file = './data/videos/zoomed.mp4'
# video_file = '/run/media/ckirst/ERMu_24/ERMu2025/VIDEOART_EXPORTS/LOUIS.mov'
#
# image_file = ('./data/images/warhol.png', './data/images/warhol_2.png')
#
# video_file = '/home/ckirst/Downloads/MOSES.mov'
# image_files = ('./data/images/moses_1.jpg', './data/images/moses_2.jpg', './data/images/moses_3.jpg')

# video_file = '/home/ckirst/Downloads/NATHANIEL.mov'
# image_files = ('./data/images/nathaniel_1.jpg',)

# video_file = '/home/ckirst/Downloads/02_LOUIS2.mov'
# image_files = ('./data/images/earth_tones.webp',)
# image_files = ('./data/images/black-yellow-red-white-blue.png',)
#
# video_file = '/home/ckirst/Downloads/CICI.mov'
# image_files = ('./data/images/gray_2.webp',)

# video_file = '/home/ckirst/Downloads/ANGCAI.mov'
# image_files = ('./data/images/monet_3.jpeg',)
# image_files = ('./data/images/kusama.jpeg',)

video_file = '/home/ckirst/Downloads/ANGELA.mov'
image_files = ('./data/images/warhol.png', './data/images/warhol_2.png')


locations = [(0, 8), (1, 9), (0, 47), (2, 41), (4, 19), (10, 17), (11, 25), (12, 2), (12, 0)]
# (7, 10),  (5, 9),  (8, 34), (2, 17),
n_features = 9
n_features_width = 3
n_features_height = 3
n_frames_per_chunk = 3
device = 'cuda'

video_file_ai = video_file[:-4] + '_ai.mov'  # + video_file[-4:]

video = cv2.VideoCapture(video_file)
n_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
fps = int(video.get(cv2.CAP_PROP_FPS))
frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

# frame_width = n_features_width * frame_width
# frame_height = n_features_height * frame_height

feature_width = frame_width // n_features_width
feature_height = frame_height // n_features_height

model = torchvision.models.vgg16(pretrained=True)
model.to(device)
model.eval()

resize = transforms.Resize((feature_height, feature_width))

# warhol = get_warhol_colormap(n_colors=32)

n_colors = [8, 9, 10,
            64, 16, 8,
            16, 5, 32]

n_colors = np.array(n_colors) + 0

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

colormaps = {i: get_dominant_colormap(image_files, n_colors=n_colors[i], sort=color_sorts[i]) for i in range(len(n_colors))}

for i in range(5):
    plot_colormap(colormaps[i])

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
    # features.append(output[:, index].cpu().numpy())

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
    transforms.Resize((feature_height, feature_width)),
    # transforms.Resize((frame_height, frame_width)),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])  #

print(f"{frame_width=}, {frame_height=} {feature_width=} {feature_height=}")

#%%

video = cv2.VideoCapture(video_file)

# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fourcc = cv2.VideoWriter_fourcc(*'avc1')
# fourcc = cv2.VideoWriter_fourcc(*'x264')

import os
if os.path.exists(video_file_ai):
    raise FileExistsError

video_out = cv2.VideoWriter(video_file_ai, fourcc, fps, (frame_width, frame_height))

n_frames_per_chunk = 20
if n_frames_per_chunk is None:
    n_frames_per_chunk = n_frames
n_chunks = int(np.ceil(n_frames / n_frames_per_chunk))
# n_chunks = 2

import tqdm
for c in tqdm.tqdm(range(n_chunks)):
    chunk_size = n_frames_per_chunk if c < n_chunks - 1 else \
        n_frames - (n_frames // n_frames_per_chunk) * n_frames_per_chunk
    if chunk_size == 0:
        continue

    frames = []
    for f in range(chunk_size):
        success, frame = video.read()
        if not success:
            print('failed to read frame in chunk={c} f={s}')
        else:
            frames.append(transform(frame))
    if len(frames) == 0:
        continue

    frames = torch.stack(frames).to(device=device)

    features = []
    frames_out = []

    with torch.no_grad():
        _ = model(frames)

    frames = []

    [video_out.write(cv2.cvtColor(np.asarray(frame * 255, dtype='uint8'), cv2.COLOR_RGB2BGR)) for frame in frames_out]

video_out.release()

#%% test

video_test = cv2.VideoCapture(video_file)

n_test = 3
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
for k in range(n_test):
    plt.subplot(4, 4, 2 + k)
    plt.imshow(np.asarray(frames_out[k] * 255, dtype='uint8'))
    plt.axis('off')
    plt.tight_layout()
    plt.show()
