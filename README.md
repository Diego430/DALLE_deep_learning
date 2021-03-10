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

###Dataset generation

For the creation of the dataset run this command
```sh
   python generateRainbowShapeDataset.py
```

If necessary, modify these constants to change the dataset target folder

```python
DATA_PATH = "./data" 
RAINBOW_DATA_PATH = "./data/rainbow"
```

###Training

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

##Results

With this implementation I've managed to train a DALLE model that generated this images:

<img src="./generated_images/big_black_circle_rainbow_v3dalle_.png" width=200></img>