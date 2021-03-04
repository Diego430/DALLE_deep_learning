import torch.nn.functional as F
from axial_positional_embedding import AxialPositionalEmbedding
from einops import rearrange
from torch import nn

from .DiscreteVAE import DiscreteVAE
from .Transformer import Transformer
from .helperFunctions import *


class DALLE(nn.Module):
    def __init__(
            self,
            *,
            dimension,  # dimension
            vae,  # vae object
            vocabulary,  # the vocabulary used to train
            text_seq_len=256,
            depth,  # should aim to 64
            heads=8,
            dim_head=64,  # attention heads
            attn_dropout=0.,  # attention droppout
            ff_dropout=0,  # feedforward dropout
            sparse_attn=False,
            noncausal_attn_len=0,
            ignore_index=-100
    ):
        super().__init__()
        assert isinstance(vae, DiscreteVAE), 'vae must be an instance of DiscreteVAE'

        image_size = vae.image_size
        image_codebook_size = vae.num_tokens
        image_seq_len = (vae.image_size // (2 ** vae.num_layers)) ** 2

        # initialize text and image codebook
        text_codebook_size = vocabulary.num_words
        self.text_codebook = nn.Embedding(text_codebook_size, dimension)
        self.image_codebook = nn.Embedding(image_codebook_size, dimension)

        self.text_pos_emb = nn.Embedding(text_seq_len + 1, dimension)  # +1 for <bos>
        self.image_pos_emb = AxialPositionalEmbedding(dimension, axial_shape=(image_size, image_size))

        self.num_text_tokens = text_codebook_size  # for offsetting logits index and calculating cross entropy loss
        self.num_image_tokens = image_codebook_size

        self.text_seq_len = text_seq_len
        self.image_seq_len = image_seq_len

        seq_len = text_seq_len + image_seq_len
        total_tokens = text_codebook_size + image_codebook_size
        self.total_tokens = total_tokens
        self.total_seq_len = seq_len

        self.noncausal_attn_len = noncausal_attn_len

        self.vae = vae
        self.vocabulary = vocabulary

        if exists(self.vae):
            self.vae = vae
            self.image_codebook = vae.codebook  # nn.Embedding

        # transformer per codifica del text
        self.transformer = Transformer(
            dim=dimension,
            causal=True,
            seq_len=seq_len,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            noncausal_attn_len=(noncausal_attn_len + 1),
            sparse_attn=sparse_attn,
            sparse_attn_global_indices=range(text_seq_len)
        )

        # logits = the vector of raw (non-normalized) predictions that a classification model generates
        self.to_logits = nn.Sequential(
            nn.LayerNorm(dimension),
            nn.Linear(dimension, self.total_tokens),
        )

        seq_range = torch.arange(seq_len)
        logits_range = torch.arange(total_tokens)

        seq_range = rearrange(seq_range, 'n -> () n ()')  # riordina array
        logits_range = rearrange(logits_range, 'd -> () () d')

        logits_mask = (
                ((seq_range >= text_seq_len) & (logits_range < text_codebook_size)) |
                ((seq_range < text_seq_len) & (logits_range >= text_codebook_size))
        )

        self.register_buffer('logits_mask', logits_mask)

        self.ignore_index = ignore_index

    def sentence2codes(self,
                       sentence,
                       device,
                       text_size=None):

        text_token = []
        for word in sentence.split(' ') :
            token = self.vocabulary.to_index(word)
            text_token.append(token)
        if text_size is not None:
            text_token = text_token + [0] * (text_size - len(text_token))
        text = torch.LongTensor([text_token])
        text = text.to(device)
        return text

    # generate image based on text codebook codes passed as text
    @torch.no_grad()
    @eval_decorator
    def generate_images(
            self,
            text,
            *,
            clip=None,
            mask=None,
            filter_thres=0.5,
            temperature=1.,
            verbose=False,
    ):
        # convert string text to vocab codebook codes
        total_len = self.text_seq_len + self.image_seq_len

        out = text
        for cur_len in range(text.shape[1], total_len):
            if verbose:
                print(str(cur_len) + " / " + str(total_len))
            is_image = cur_len >= self.text_seq_len

            text, image = out[:, :self.text_seq_len], out[:, self.text_seq_len:]

            logits = self.forward(text, image, mask=mask)[:, -1, :]

            filtered_logits = top_k(logits, thres=filter_thres)
            probs = F.softmax(filtered_logits / temperature, dim=-1)
            sample = torch.multinomial(probs, 1)

            # offset sampled token if it is an image token, since logit space is composed of text and then image tokens
            sample -= (self.num_text_tokens if is_image else 0)
            out = torch.cat((out, sample), dim=-1)

            if out.shape[1] <= self.text_seq_len:
                mask = F.pad(mask, (0, 1), value=True)

        text_seq = out[:, :self.text_seq_len]

        img_seq = out[:, -self.image_seq_len:]
        images = self.vae.decode(img_seq)

        if exists(clip):
            scores = clip(text_seq, images, return_loss=False)
            return images, scores

        return images

    def forward(
            self,
            text,
            image=None,
            mask=None,
            return_loss=False,
            verbose=False
    ):
        device = text.device

        text = F.pad(text, (1, 0), value=0)  # use padding as <bos>

        if exists(mask):
            mask = F.pad(mask, (1, 0), value=True)

        tokens = self.text_codebook(text)
        tokens += self.text_pos_emb(torch.arange(text.shape[1], device=device))

        seq_len = tokens.shape[1]

        if exists(image) and not is_empty(image):
            is_raw_image = len(image.shape) == 4
            if is_raw_image:
                image = self.vae.get_codebook_indices(image)

            image_len = image.shape[1]
            image_emb = self.image_codebook(image)
            image_emb += self.image_pos_emb(image_emb)

            tokens = torch.cat((tokens, image_emb), dim=1)

            seq_len += image_len
            if exists(mask):
                mask = F.pad(mask, (0, image_emb.shape[1]), value=True)

        # when training, if the length exceeds the total text + image length
        # remove the last token, since it needs not to be trained
        if tokens.shape[1] > self.total_seq_len:
            seq_len -= 1
            tokens = tokens[:, :-1]

            if exists(mask):
                mask = mask[:, :-1]

        if verbose:
            print("Transformer...")
        out = self.transformer(tokens, mask=mask)

        if verbose:
            print("To Logits...")
        logits = self.to_logits(out)

        if verbose:
            print("Mask logits...")
        # mask logits to make sure text predicts text (except last token), and image predicts image
        logits_mask = self.logits_mask[:, :seq_len]
        max_neg_value = -torch.finfo(logits.dtype).max
        logits.masked_fill_(logits_mask, max_neg_value)

        if not return_loss:
            return logits

        assert exists(image), 'when training, image must be supplied'
        noncausal_attn_len = self.noncausal_attn_len
        offsetted_image = image + self.num_text_tokens
        labels = torch.cat((text[:, 1:], offsetted_image), dim=1)

        if noncausal_attn_len > 0:
            seq_range = torch.arange(seq_len, device=device)
            noncausal_attn_mask = seq_range < noncausal_attn_len
            noncausal_attn_mask = rearrange(noncausal_attn_mask, 'n -> () n')
            labels.masked_fill_(noncausal_attn_mask, self.ignore_index)

        if verbose:
            print("Cross Entropy...")
        loss = F.cross_entropy(rearrange(logits, 'b n c -> b c n'), labels)
        return loss
