# -*- coding: utf-8 -*-
"""
@Author : Kwok
@Date   : 2023/3/22 10:14
"""
from PIL import Image
import numpy as np
import torch

import deep_danbooru_model

model = deep_danbooru_model.DeepDanbooruModel()
model.load_state_dict(torch.load("models/model-resnet_custom_v3.pt"))

model.eval()
model.half()
model.cuda()


def resize_image(im, width, height):
    ratio = width / height
    src_ratio = im.width / im.height

    src_w = width if ratio < src_ratio else im.width * height // im.height
    src_h = height if ratio >= src_ratio else im.height * width // im.width

    resized = im.resize((src_w, src_h), resample=Image.Resampling.LANCZOS)
    res = Image.new("RGB", (width, height))
    res.paste(resized, box=(width // 2 - src_w // 2, height // 2 - src_h // 2))

    if ratio < src_ratio:
        fill_height = height // 2 - src_h // 2
        res.paste(resized.resize((width, fill_height), box=(0, 0, width, 0)), box=(0, 0))
        res.paste(resized.resize((width, fill_height), box=(0, resized.height, width, resized.height)), box=(0, fill_height + src_h))
    elif ratio > src_ratio:
        fill_width = width // 2 - src_w // 2
        res.paste(resized.resize((fill_width, height), box=(0, 0, 0, height)), box=(0, 0))
        res.paste(resized.resize((fill_width, height), box=(resized.width, 0, resized.width, height)), box=(fill_width + src_w, 0))

    return res


pic = Image.open("test.jpeg").convert("RGB")
pic = resize_image(pic, 512, 512)

a = np.expand_dims(np.array(pic, dtype=np.float32), 0) / 255

with torch.no_grad(), torch.autocast("cuda"):
    x = torch.from_numpy(a).cuda()
    y = model(x)[0].detach().cpu().numpy()

probability_dict = {}

for tag, probability in zip(model.tags, y):
    if probability < 0.5:
        continue

    if tag.startswith("rating:"):
        continue

    probability_dict[tag] = probability

tags = sorted(probability_dict)

print(", ".join(tags))
