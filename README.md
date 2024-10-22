# DALL-E Pytorch

Implementation / replication of <a href="https://openai.com/blog/dall-e/">DALL-E</a>, OpenAI's Text to Image Transformer, in Pytorch.
This project is based on <a href="https://github.com/htoyryla/DALLE-pytorch">Htoyryla's code</a>.
The dataset is taken from the original implemetation by <a href="https://github.com/lucidrains/dalle-pytorch">Lucidrains</a>.

About the classes in dalle_pytorch I've separated them into several files for clarity.
I've also changed some variable's names to allow me to better understand their use.

Regarding instead the files that do training of both DALLE and VAE 
I have modified the code in a more substantial way to make it work on my computer.

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/Diego430/DALLE_deep_learning.git
   ```
2. Install requirements
   ```sh
   pip install -r requirements.txt
   ```

Be careful that the version of pytorch has installed the cuda version
suitable for your system if you want to use the GPU for training and generation.

## Usage

###  Dataset generation

For the creation of the dataset run this command
```sh
   python generateRainbowShapeDataset.py
```

If necessary, modify these constants to change the dataset target folder

```python
DATA_PATH = "./data" 
RAINBOW_DATA_PATH = "./data/rainbow"
```

The possible variation that can be generated are these:
```python
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
```

### Training

Regarding the training of all the components both trainVAErainbow.py and trainDALLErainbow.py 
starts with the declaration of the constant that tune the execution of the training.
The constants are separated in "paragraph" by comments to highlight what they are useful for.
The same thing is done with the generateDALLEimage_rainbow.py.

It is important that VAE and DALLE parameters are consistent across all files.

After all the tuning to the constant you need to run in order
1. VAE training, recommended at least 200 epoch.
```sh
   python trainVAErainbow.py
```

2. DALLE training, recommended at least 30 epoch.
```sh
   python trainDALLErainbow.py
```

3. Finally test what is DALLE generating.
Modify CAPTIONS content to select what shape DALLE as to generate. 
```sh
   python generateRainbowShapeDataset.py
```

For both DALLE and VAE training is created a csv file where for each log_interval is written the loss regarding that particular iteration.
This allows to check the loss value during every iteration if it is done unsupervised and to better understand how fast the loss decreases. 

## Results

With this implementation I've managed to train a DALLE model that generated this images:

| big black circle  | filled green star | red hexagon rainbow | smaller yellow triangle rotated twice |
| ------------- | ------------- | ------------- | ------------- |
| <img src="./generated_images/big_black_circle_rainbow_v3dalle_.png" width=200 title="big black circle"></img>  | <img src="./generated_images/filled_green_star_rainbow_v3dalle_.png" width=200 title="filled green star"></img>  | <img src="./generated_images/red_hexagon_rainbow_v3dalle_.png" width=200 title="red hexagon rainbow"></img> | <img src="./generated_images/smaller_yellow_triangle_rotated_twice_rainbow_v3dalle_.png" width=200 title="smaller yellow triangle rotated twice"></img> |

