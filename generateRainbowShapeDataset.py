import itertools
import math
import os

import cairo
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm.autonotebook import tqdm

DATA_PATH = "./data"  # path to imageFolder
RAINBOW_DATA_PATH = "./data/rainbow"

rainbow_colors = ["red", "orange", "yellow", "green", "blue", "indigo", "violet"]
rainbow_rgb = [np.array(mcolors.to_rgb(name)) * 255 for name in rainbow_colors]


def make_pic(size, shape_maker, color_name, scale, fill, ditherer, rotation) :
	data = np.ones((size, size, 4), dtype=np.uint8)
	surface = cairo.ImageSurface.create_for_data(data, cairo.FORMAT_ARGB32, size, size)
	cr = cairo.Context(surface)
	cr.set_antialias(cairo.ANTIALIAS_NONE)
	cr.set_fill_rule(cairo.FILL_RULE_EVEN_ODD)

	cr.rectangle(0, 0, size, size)
	cr.set_source_rgb(1, 1, 1)
	cr.fill()
	cr.set_line_width(1 / (scale * size / 2))

	cr.translate(size // 2, size // 2)
	cr.scale(scale * size / 2, scale * size / 2)
	if color_name == "rainbow" :
		cr.set_source_rgb(0, 0, 0)
	else :
		cr.set_source_rgb(*reversed(mcolors.to_rgb(color_name)))

	shape_maker(cr)
	if fill :
		cr.stroke_preserve()
		cr.fill()
	else :
		cr.stroke()

	for _ in range(rotation) :
		data = np.rot90(data)

	ditherer(data)

	if color_name == "rainbow" :
		mask = data.sum(axis=2) != 255 * 4
		for r in range(size) :
			data[r, mask[r, :], :3] = rainbow_rgb[r % len(rainbow_colors)]

	return data


def circle_maker(cr) :
	cr.arc(0, 0, 1, 0, 2 * math.pi)


def triangle_maker(cr) :
	s3 = math.sqrt(3)
	cr.move_to(0, -1)
	cr.line_to(s3 / 2, 0.5)
	cr.line_to(-s3 / 2, 0.5)
	cr.line_to(0, -1)


def square_maker(cr) :
	cr.rectangle(-0.9, -0.9, 1.8, 1.8)


def rectangle_maker(cr) :
	cr.rectangle(-0.9, -0.5, 1.8, 1)


def rhombus_maker(cr) :
	cr.move_to(0, -1)
	cr.line_to(0.5, 0)
	cr.line_to(0, 1)
	cr.line_to(-0.5, 0)
	cr.line_to(0, -1)


def star_maker(cr) :
	s3 = math.sqrt(3)
	cr.move_to(0, -1)
	cr.line_to(s3 / 2, 0.5)
	cr.line_to(-s3 / 2, 0.5)
	cr.line_to(0, -1)

	cr.move_to(0, 1)
	cr.line_to(s3 / 2, -0.5)
	cr.line_to(-s3 / 2, -0.5)
	cr.line_to(0, 1)


def hexagon_maker(cr) :
	s3 = math.sqrt(3)
	cr.move_to(0, -1)
	cr.line_to(s3 / 2, -0.5)
	cr.line_to(s3 / 2, 0.5)
	cr.line_to(0, 1)
	cr.line_to(-s3 / 2, 0.5)
	cr.line_to(-s3 / 2, -0.5)
	cr.line_to(0, -1)


def crescent_maker(cr) :
	s3 = math.sqrt(3)
	cr.arc(0, 0, 1, -math.pi * 0.5, math.pi * 0.5)
	cr.move_to(0, -1)
	cr.arc(-s3, 0, 2, -math.pi / 6, math.pi / 6)


def dither_solid(data) :
	pass


def dither(img) :
	for y in range(0, img.shape[0] - 1) :
		for x in range(1, img.shape[1] - 1) :
			pix = img[y][x]
			newpix = round(pix)

			error = pix - newpix

			img[y][x] = newpix

			img[y, x + 1] += error * 7 / 16
			img[y + 1, x - 1] += error * 3 / 16
			img[y + 1, x] += error * 5 / 16
			img[y + 1, x + 1] += error * 1 / 16


def dither_halftone(data) :
	mask = (data.astype(int).sum(axis=2) != 255 * 3).astype(float)
	mask *= 0.5
	dither(mask)
	data[mask > 0.5, :] = 255


def dither_shaded(data) :
	mask = (data.astype(int).sum(axis=2) != 255 * 3).astype(float)
	mask *= 0.3
	dither(mask)
	data[mask > 0.5, :] = 255


variations = {
	"scale" : {"big" : 1, "bigger" : 0.8, "smaller" : 0.6, "small" : 0.4},
	"fill" : {"filled" : True, "" : False},
	"ditherer" : {"" : dither_solid, "shaded" : dither_shaded, "halftone" : dither_halftone},
	"color_name" : {c : c for c in rainbow_colors + ["cyan", "saddlebrown", "black", "gray", "rainbow"]},
	"shape_maker" : {"circle" : circle_maker, "triangle" : triangle_maker, "square" : square_maker,
					"rhombus" : rhombus_maker, "rectangle" : rectangle_maker, "star" : star_maker,
					"hexagon" : hexagon_maker,
					"crescent" : crescent_maker},
	"rotation" : {"" : 0, "rotated clockwise" : 1, "rotated twice" : 2, "rotated counterclockwise" : 3},
}

print(f"Total shapes: {np.prod([len(v) for v in variations.values()])}")

pics = []
for _ in range(20) :
	row = []
	for _ in range(40) :
		params = {name : np.random.choice(list(values.values())) for name, values in variations.items()}
		row.append(make_pic(32, **params)[:, :, :3])

	pics.append(np.concatenate(row, axis=1))

pics = np.concatenate(pics)
# write output
plt.figure(figsize=(20, 10))
plt.axis("off")
plt.imshow(pics)

all_variations = list(itertools.product(*[v.items() for v in variations.values()]))
if not os.path.exists(DATA_PATH) :
	os.mkdir(DATA_PATH)

if not os.path.exists(RAINBOW_DATA_PATH) :
	os.mkdir(RAINBOW_DATA_PATH)
	for vars in tqdm(all_variations) :
		name = " ".join(n for n, v in vars if n != "").replace(" ", "_")
		params = dict(zip(variations, [v for n, v in vars]))
		pic = make_pic(32, **params)[:, :, :3]
		im = Image.fromarray(pic)
		im.save(RAINBOW_DATA_PATH + '/' + name + ".png")
