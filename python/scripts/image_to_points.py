from visualization.plotting import plt

import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from functools import partial

device = 'cpu'

transform = transforms.Compose([
    transforms.ToTensor(),
])

image_path = "./data/images/lightning.jpg"  # Replace with the actual path to your image
image = Image.open(image_path).convert('RGB')
input_tensor = transform(image).unsqueeze(0)  # Add batch dimension
input_tensor = input_tensor.to(device)

array = np.sum(input_tensor.cpu().numpy()[0], axis=0).T

x, y = 1080//2 + 50, 1920//2 - 100
w, h = 200, 300
cut = array[x-w:x+w, y-h:y+h]
pg.image(cut)

from visualization.plotting import pg


def create_points(array, n_points: int = 1000):
    xs, ys = np.where(array)
    n = len(xs)
    print(n)
    indices = np.random.randint(0, n, n_points)
    x = xs[indices] + np.random.rand(n_points)
    y = xs[indices] + np.random.rand(n_points)
    z = np.random.rand(n_points)

    points = np.array([x, y, z])

    return points


positions = create_points(cut, 1000)

from visualization.plotting import pv
pv.plot(positions.T)
