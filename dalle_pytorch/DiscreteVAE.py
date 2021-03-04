from math import log2, sqrt

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn, einsum

from .ResBlock import ResBlock


class DiscreteVAE(nn.Module):
    """
    This class of models efficiently learns both the class of objects in an image,
    and their specific realization in pixels, from unsupervised data.
    init:
        IMAGE_SIZE=256,
        IMAGE_CODEBOOK_SIZE=512, # number of visual tokens. in the paper, they used 8192, but could be smaller for downsized projects
        CODEBOOK_DIM=512, # codebook dimension
        NUM_LAYERS=3,  # number of downsamples - ex. 256 / (2 ** 3) = (32 x 32 feature map)
        num_resnet_blocks=0, # number of resnet blocks
        HIDDEN_DIM=64,  # hidden dimension
        channels=3,
        TEMPERATURE=0.9,  # gumbel softmax TEMPERATURE, the lower this is, the harder the discretization

    """
    # costruttore
    def __init__(
            self,
            image_size=256,
            image_codebook_size=512,  # number of visual tokens. in the paper, they used 8192, but could be smaller for downsized projects
            codebook_dim=512,  # codebook dimension
            num_layers=3,  # number of downsamples - ex. 256 / (2 ** 3) = (32 x 32 feature map)
            num_resnet_blocks=0,  # number of resnet blocks
            hidden_dim=64,  # hidden dimension
            channels=3,  # encoder channels
            temperature=0.9,  # gumbel softmax TEMPERATURE, the lower this is, the harder the discretization
    ):
        super().__init__()
        assert log2(image_size).is_integer(), 'image size must be a power of 2'
        assert num_layers >= 1, 'number of layers must be greater than or equal to 1'
        has_resblocks = num_resnet_blocks > 0

        # passo i parametri alle variabili
        self.image_size = image_size
        self.num_tokens = image_codebook_size
        self.num_layers = num_layers
        self.temperature = temperature
        self.codebook = nn.Embedding(image_codebook_size, codebook_dim)

        hdim = hidden_dim

        # create encoder e decoder channels
        enc_channels = [hidden_dim] * num_layers
        dec_channels = list(reversed(enc_channels))

        enc_channels = [channels, *enc_channels]

        dec_init_channels = codebook_dim if not has_resblocks else dec_channels[0]
        dec_channels = [dec_init_channels, *dec_channels]

        enc_channels_io, dec_channels_io = map(lambda t: list(zip(t[:-1], t[1:])), (enc_channels, dec_channels))

        enc_layers = []
        dec_layers = []

        # create layers for encoder and decoder,
        # encoder layers:
        #       convolutional layer [enc_in, enc_out, 4] with stride 2 (output half the dimension)
        #       relu layer
        # decoder layers:
        #       convolutional transposed layer [dec_in, dec_out, 4] with stride 2 (output half the dimension)
        #       relu layer
        for (enc_in, enc_out), (dec_in, dec_out) in zip(enc_channels_io, dec_channels_io):
            enc_layers.append(nn.Sequential(nn.Conv2d(enc_in, enc_out, 4, stride=2, padding=1), nn.ReLU()))
            dec_layers.append(nn.Sequential(nn.ConvTranspose2d(dec_in, dec_out, 4, stride=2, padding=1), nn.ReLU()))

        # inser resnet block, for the decoder in the head, for the encoder in the tail
        for _ in range(num_resnet_blocks):
            dec_layers.insert(0, ResBlock(dec_channels[1]))
            enc_layers.append(ResBlock(enc_channels[-1]))

        if num_resnet_blocks > 0:
            dec_layers.insert(0, nn.Conv2d(codebook_dim, dec_channels[1], 1))

        enc_layers.append(nn.Conv2d(enc_channels[-1], image_codebook_size, 1))
        dec_layers.append(nn.Conv2d(dec_channels[-1], channels, 1))

        self.encoder = nn.Sequential(*enc_layers)
        self.decoder = nn.Sequential(*dec_layers)

    # cerca di adattare i logits trovati ai token nel codebook usando la best fit
    @torch.no_grad()
    def get_codebook_indices(self, images):
        logits = self.forward(images, return_logits=True)
        codebook_indices = logits.argmax(dim=1).flatten(1)
        return codebook_indices

    # decodifica dell'immagine passata come param
    def decode(
            self,
            img_seq
    ):
        image_embeds = self.codebook(img_seq)
        b, n, d = image_embeds.shape
        h = w = int(sqrt(n))

        image_embeds = rearrange(image_embeds, 'b (h w) d -> b d h w', h=h, w=w)
        images = self.decoder(image_embeds)
        return images

    # passo di forward su img
    def forward(
            self,
            img,
            return_recon_loss=False,
            return_logits=False
    ):
        logits = self.encoder(img)

        if return_logits:
            return logits  # return logits for getting hard image indices for DALL-E training

        soft_one_hot = F.gumbel_softmax(logits, tau=self.temperature, dim=1)
        sampled = einsum('b n h w, n d -> b d h w', soft_one_hot, self.codebook.weight)
        out = self.decoder(sampled)

        # se non viene richiesta la loss in output ritorno il risultato del decoder
        if not return_recon_loss:
            return out

        # calcolo mean squared error del img per ritornarla se richiesta
        loss = F.mse_loss(img, out)
        return loss
