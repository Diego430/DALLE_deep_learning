import os

import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets

from dalle_pytorch.DiscreteVAE import DiscreteVAE
from dalle_pytorch.DALLE import DALLE
from dalle_pytorch.Vocabulary import Vocabulary

from torchvision.utils import save_image

# EXEC params
BATCH_SIZE = 256  # batch size for training
DATA_PATH = "./data"  # path to imageFolder
IMAGE_SIZE = 32  # image size for training
N_EPOCHS = 250  # number of epochs
LEARNING_RATE = 1e-4  # learning rate
TEMPERATURE = 0.9  # vae TEMPERATURE
MAX_DATASET_ELEMENTS = -1

# LOG params
LOG_INTERVAL = -1  # status print interval
VERBOSE = True
NUM_WORKERS = 0  # number of concurrent process that load the dataset

# VAE params
VAE_NUM_LAYERS = 3  # number of downsamples - ex. 256 / (2 ** 3) = (32 x 32 feature map)
VAE_CHANNELS = 3  # encoder channels
VAE_IMAGE_CODEBOOK_SIZE = 2048  # number of visual tokens. in the paper, they used 8192, but could be smaller for downsized projects
VAE_CODEBOOK_DIM = 256  # codebook dimension
VAE_HIDDEN_DIM = 128  # hidden dimension
VAE_NAME = "rainbow_v1vae256"
VAE_LOAD_EPOCH = 399

# DALLE params
DALLE_DIM = 256  # 512,
DALLE_TEXT_SEQ_LEN = 256  # text sequence length
DALLE_DEPTH = 12  # should be 64
DALLE_HEADS = 16  # attention heads
DALLE_DIM_HEAD = 64  # attention head dimension
DALLE_ATTN_DROPOUT = 0.1  # attention dropout
DALLE_FF_DROPOUT = 0.1  # feedforward dropout

# to continue training from a saved checkpoint, give checkpoint path as toLoadDALLE and START_EPOCH
START_EPOCH = 50
START_IMAGE = 0
name = "rainbow_v3dalle"  # experiment NAME
to_load_DALLE = "./models/" + name + "_dalle_" + str(START_EPOCH) + "_" + str(
	START_IMAGE) + ".pth" if START_EPOCH != 0 and START_IMAGE != 0 else ''

print("Start")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

image_transformer = transforms.Compose([
	transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
	transforms.ToTensor(),
	transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# load train set
train_set = datasets.ImageFolder(DATA_PATH, transform=image_transformer, target_transform=None)

if MAX_DATASET_ELEMENTS == -1 :
	MAX_DATASET_ELEMENTS = len(train_set)
if LOG_INTERVAL == -1 :
	LOG_INTERVAL = MAX_DATASET_ELEMENTS

# Build vocabulary
print("Build Vocabulary...")
vocab = Vocabulary("captions")
vocab_dataset_element = 0
filenames = os.listdir(DATA_PATH + "/rainbow")
captions = [filename.replace(".png", '').replace('_', ' ') for filename in filenames]
for caption in captions :
	vocab.add_sentence(caption)
	if vocab_dataset_element + 1 >= MAX_DATASET_ELEMENTS :
		break
	vocab_dataset_element += 1
print("Vocabulary size : {}".format(vocab.num_words))
# 11355 words with 10000 elements of the dataset

vae = DiscreteVAE(
	image_size=IMAGE_SIZE,
	num_layers=VAE_NUM_LAYERS,
	channels=VAE_CHANNELS,
	image_codebook_size=VAE_IMAGE_CODEBOOK_SIZE,
	codebook_dim=VAE_CODEBOOK_DIM,
	hidden_dim=VAE_HIDDEN_DIM,
	temperature=TEMPERATURE
)

# Load pretrained vae
vae_path = "./models/" + VAE_NAME + "-" + str(VAE_LOAD_EPOCH) + ".pth"
print("loading VAE from " + vae_path)
vae_dict = torch.load(vae_path)
vae.load_state_dict(vae_dict)
vae.to(DEVICE)

dalle = DALLE(
	dimension=DALLE_DIM,
	vae=vae,  # automatically infer (1) image sequence length and (2) number of image tokens
	vocabulary=vocab,  # the vocabulary used to train
	text_seq_len=DALLE_TEXT_SEQ_LEN,  # text sequence length
	depth=DALLE_DEPTH,  # should be 64
	heads=DALLE_HEADS,  # attention heads
	dim_head=DALLE_DIM_HEAD,  # attention head dimension
	attn_dropout=DALLE_ATTN_DROPOUT,  # attention dropout
	ff_dropout=DALLE_FF_DROPOUT  # feedforward dropout
)

# load pretrained dalle if continuing training
if to_load_DALLE != "" :
	dalle_dict = torch.load(to_load_DALLE)
	dalle.load_state_dict(dalle_dict)

dalle.to(DEVICE)

# datactr = 0

# instantiate optimizer
optimizer = optim.Adam(dalle.parameters(), lr=LEARNING_RATE)

# TRAINING
print("Training")
for epoch in range(START_EPOCH, N_EPOCHS) :
	batch_index = 0
	train_loss = 0
	for iimage, ccaption in zip(train_set, captions) :  # loop through dataset by minibatch
		if batch_index >= START_IMAGE :
			codes_captions = [0]
			codes_caption = dalle.vocabulary.sentence_to_index(ccaption)
			codes_caption = codes_caption + [0] * (DALLE_TEXT_SEQ_LEN - len(codes_caption))
			codes_captions[0] = codes_caption
			text = torch.LongTensor(codes_captions)  # a minibatch of text (numerical tokens)
			images = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE)  # placeholder for images
			images[0] = iimage[0]

			text = text.to(DEVICE)
			images = images.to(DEVICE)

			mask = torch.ones_like(text).bool().to(DEVICE)

			# train and optimize a single minibatch
			optimizer.zero_grad()
			loss = dalle.forward(text, images, mask=mask, return_loss=True)
			train_loss += loss.item()
			loss.backward()
			optimizer.step()

			if batch_index % LOG_INTERVAL == 0 and VERBOSE :
				loss_int = train_loss / (batch_index + 1)
				percentage = 100. * batch_index / MAX_DATASET_ELEMENTS
				print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
					epoch, batch_index, MAX_DATASET_ELEMENTS, percentage, train_loss / (batch_index + 1)))
				dalle_state_filename = "./models/" + name + "_dalle_" + str(epoch + 1) + "_" + str(batch_index) + ".pth"
				torch.save(dalle.state_dict(), dalle_state_filename)
				# log for the loss
				toWriteLoss = open("lossPerEpochDALLE.csv", "a")
				toWriteLoss.write("\n{}, {}, {}".format(epoch, batch_index, train_loss / (batch_index + 1)))
				toWriteLoss.close()
		if batch_index + 1 >= MAX_DATASET_ELEMENTS :
			break
		batch_index += 1

	print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / MAX_DATASET_ELEMENTS))
	toWriteLoss = open("lossPerEpochDALLE.csv", "a")
	toWriteLoss.write("\n{}, {}, {}".format(epoch, -1, train_loss / MAX_DATASET_ELEMENTS))
	toWriteLoss.close()

	torch.save(dalle.state_dict(), "./models/" + name + "_dalle_" + str(epoch + 1) + ".pth")

	# generate a test sample from the captions in the last minibatch
	text_string = []
	for word in text[0] :
		text_string.append(dalle.vocabulary.to_word(word.item()))
	print(text_string)
	image_generated = dalle.generate_images(text, mask=mask)
	save_image(image_generated, 'results/' + name + '_dalle_epoch_' + str(epoch) + '.png', normalize=True)
	START_IMAGE = 0
