import torch
import torch.nn.functional as F
from torch import optim
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torchvision.transforms as transforms

from dalle_pytorch.DiscreteVAE import DiscreteVAE


# FILE NAMES
RESULTS_DIR_PATH = "./results"
MODELS_DIR_PATH = "./models"
VAE_LOSS_LOGFILE = "lossPerEpochVAE.csv"

# EXEC params
BATCH_SIZE = 8  # batch size for training
DATA_PATH = "./data"  # path to imageFolder
IMAGE_SIZE = 32  # image size for training
N_EPOCHS = 300  # number of epochs
LEARNING_RATE = 1e-4  # learning rate
TEMPERATURE = 0.9  # vae TEMPERATURE
NAME = "rainbow_v1vae256"  # experiment NAME
MAX_DATASET_ELEMENTS = -1

# VAE params
NUM_LAYERS = 3  # number of downsamples - ex. 256 / (2 ** 3) = (32 x 32 feature map)
CHANNELS = 3  # encoder channels
IMAGE_CODEBOOK_SIZE = 2048  # number of visual tokens. in the paper, they used 8192, but could be smaller for downsized projects
CODEBOOK_DIM = 256  # codebook dimension
HIDDEN_DIM = 128  # hidden dimension

# LOG params
LOG_INTERVAL = 100  # status print interval
VERBOSE = True
NUM_WORKERS = 0  # number of concurrent process that load the dataset
RESULT_ROWS = 6  # = number of images % batch size

# Continuing training
# set toLoadDict: path to pretrained model
# START_EPOCH: start epoch numbering from this
START_EPOCH = 99  # start epoch numbering for continuing training (default: 0)')
TO_LOAD_DICT = MODELS_DIR_PATH + NAME + "-" + str(START_EPOCH) + ".pth" if START_EPOCH != 0 else ''

if VERBOSE :
	print("Start")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# class for unsupervised learning of class object in an image and the specific realization in pixels
vae = DiscreteVAE(
	image_size=IMAGE_SIZE,
	num_layers=NUM_LAYERS,
	channels=CHANNELS,
	image_codebook_size=IMAGE_CODEBOOK_SIZE,
	codebook_dim=CODEBOOK_DIM,
	hidden_dim=HIDDEN_DIM,
	temperature=TEMPERATURE
)

if TO_LOAD_DICT != "" :
	vae_dict = torch.load(TO_LOAD_DICT)
	vae.load_state_dict(vae_dict)

vae.to(DEVICE)

# Initialize image transformer to resize and convert to tensor images, normalizing each channel
image_transformer = transforms.Compose([
	transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
	transforms.ToTensor(),
	transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

if VERBOSE:
	print("Loading dataset...")

# load train set
train_set = datasets.ImageFolder(DATA_PATH, transform=image_transformer, target_transform=None)
train_loader = DataLoader(dataset=train_set, num_workers=NUM_WORKERS, batch_size=BATCH_SIZE, shuffle=True)

# instantiate optimizer
optimizer = optim.Adam(vae.parameters(), lr=LEARNING_RATE)

# TRAINING
if VERBOSE :
	print("Training")

if MAX_DATASET_ELEMENTS == -1:
	MAX_DATASET_ELEMENTS = len(train_set)

if __name__ == "__main__" :
	for epoch in range(START_EPOCH, START_EPOCH + N_EPOCHS) :
		train_loss = 0
		dataset_elements = 0
		for batch_index, (images, _) in enumerate(train_loader, 0) :

			images = images.to(DEVICE)
			recons = vae.forward(images)  # decoder(sampled) where sampled is a reordered part of softmax return

			# Smooth L1-loss can be interpreted as a combination of L1-loss and L2-loss.
			# and Mean square error is the average of the squared errors
			loss = F.smooth_l1_loss(images, recons) + F.mse_loss(images, recons)

			# OPTIMIZE
			optimizer.zero_grad()
			loss.backward()
			train_loss += loss.item()  # convert tensor number into python number
			optimizer.step()

			# print information
			if batch_index % LOG_INTERVAL == 0 and VERBOSE :
				loss_int = train_loss / (batch_index+1)
				percentage = 100. * batch_index * len(images) / MAX_DATASET_ELEMENTS
				print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.8f}'.format(
					epoch, batch_index * len(images), MAX_DATASET_ELEMENTS, percentage, loss_int))
			if dataset_elements+1 >= MAX_DATASET_ELEMENTS:
				break
			dataset_elements += 1

		with torch.no_grad() :
			# get the codebook indices of the images
			codes = vae.get_codebook_indices(images)
			# rebuild the image based on its codes from the codebook
			decodedImage = vae.decode(codes)  # decoder(self.codebook(codes))

		# create the grid of the image to be saved as results
		grid = torch.cat([images[:RESULT_ROWS], recons[:RESULT_ROWS], decodedImage[:RESULT_ROWS]])

		# for each epoch save results and state of the dict
		image_file_path = RESULTS_DIR_PATH + '/' + NAME + '_epoch_' + str(epoch) + '.png'
		save_image(grid, image_file_path, normalize=True, nrow=RESULT_ROWS)

		print('====> Epoch: {} Average loss: {:.8f}'.format(epoch, train_loss / MAX_DATASET_ELEMENTS))

		vae_filename = MODELS_DIR_PATH + '/' + NAME + "-" + str(epoch + 1) + ".pth"
		torch.save(vae.state_dict(), vae_filename)

		# log for the loss
		toWriteLoss = open(VAE_LOSS_LOGFILE, "a")
		toWriteLoss.write("\n{}, {}".format(epoch, train_loss / MAX_DATASET_ELEMENTS))
		toWriteLoss.close()
