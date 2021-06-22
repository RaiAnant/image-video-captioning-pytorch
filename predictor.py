import torch

from transformers import BertTokenizer
from PIL import Image
import argparse

from transformer.models import caption
from transformer.datasets import coco, utils
from transformer.configuration import Config
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
from torchvision import transforms
from RNN.build_vocab import Vocabulary
from RNN.model import EncoderCNN, DecoderRNN
from PIL import Image


class Predictor:

    def __init__(self):
        self.config = Config()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.start_token = self.tokenizer.convert_tokens_to_ids(self.tokenizer._cls_token)
        self.end_token = self.tokenizer.convert_tokens_to_ids(self.tokenizer._sep_token)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])

        with open('RNN/data/vocab.pkl', 'rb') as f:
            self.vocab = pickle.load(f)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.models = {"trn": torch.hub.load('saahiluppal/catr', 'v3', pretrained=True),
                       "rnn": (EncoderCNN(256),
                               DecoderRNN(256, 512, len(self.vocab), 1))}

        self.models["trn"].eval()
        self.models["trn"].to(self.device)
        self.models["rnn"][0].eval()
        self.models["rnn"][0].to(self.device)
        self.models["rnn"][0].load_state_dict(torch.load("RNN/models/encoder-5-3000.pkl"))
        self.models["rnn"][1].eval()
        self.models["rnn"][1].to(self.device)
        self.models["rnn"][1].load_state_dict(torch.load("RNN/models/decoder-5-3000.pkl"))

    def create_caption_and_mask(self, start_token, max_length):
        self.caption = torch.zeros((1, max_length), dtype=torch.long)
        self.mask = torch.ones((1, max_length), dtype=torch.bool)

        self.caption[:, 0] = start_token
        self.mask[:, 0] = False

    def predict(self, arch="trn", pth='/home/anant/Pictures/test.png'):
        if arch == "trn":
            model = self.models[arch]
            image = Image.open(pth)
            image = coco.val_transform(image)
            image = image.unsqueeze(0)

            self.create_caption_and_mask(self.start_token, self.config.max_position_embeddings)

            for i in range(self.config.max_position_embeddings - 1):
                predictions = model(image, self.caption, self.mask)
                predictions = predictions[:, i, :]
                predicted_id = torch.argmax(predictions, axis=-1)

                if predicted_id[0] == 102:
                    break

                self.caption[:, i + 1] = predicted_id[0]
                self.mask[:, i + 1] = False

            return self.tokenizer.decode(self.caption[0].tolist(), skip_special_tokens=True).capitalize()
        else:
            image = Image.open(pth).convert('RGB')
            image = image.resize([224, 224], Image.LANCZOS)
            image = self.transform(image).unsqueeze(0)
            image_tensor = image.to(self.device)
            encoder, decoder = self.models[arch]

            feature = encoder(image_tensor)
            sampled_ids = decoder.sample(feature)
            sampled_ids = sampled_ids[0].cpu().numpy()  # (1, max_seq_length) -> (max_seq_length)

            # Convert word_ids to words
            sampled_caption = []
            for word_id in sampled_ids:
                word = self.vocab.idx2word[word_id]
                sampled_caption.append(word)
                if word == '<end>':
                    break
            sentence = ' '.join(sampled_caption)

            # Print out the image and the generated caption
            return sentence



print(Predictor().predict(arch="rnn"))

# Device configuration



# def load_image(image_path, transform=None):
#     image = Image.open(image_path).convert('RGB')
#     image = image.resize([224, 224], Image.LANCZOS)
#
#     if transform is not None:
#         image = transform(image).unsqueeze(0)
#
#     return image
#
#
# def main(args):
#     # Image preprocessing
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.485, 0.456, 0.406),
#                              (0.229, 0.224, 0.225))])
#
#     # Load vocabulary wrapper
#     with open(args.vocab_path, 'rb') as f:
#         vocab = pickle.load(f)
#
#     # Build models
#     encoder = EncoderCNN(args.embed_size).eval()  # eval mode (batchnorm uses moving mean/variance)
#     decoder = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), args.num_layers)
#     encoder = encoder.to(device)
#     decoder = decoder.to(device)
#
#     # Load the trained model parameters
#     encoder.load_state_dict(torch.load(args.encoder_path))
#     decoder.load_state_dict(torch.load(args.decoder_path))
#
#     # Prepare an image
#     image = load_image(args.image, transform)
#     image_tensor = image.to(device)
#
#     # Generate an caption from the image
#     feature = encoder(image_tensor)
#     sampled_ids = decoder.sample(feature)
#     sampled_ids = sampled_ids[0].cpu().numpy()  # (1, max_seq_length) -> (max_seq_length)
#
#     # Convert word_ids to words
#     sampled_caption = []
#     for word_id in sampled_ids:
#         word = vocab.idx2word[word_id]
#         sampled_caption.append(word)
#         if word == '<end>':
#             break
#     sentence = ' '.join(sampled_caption)
#
#     # Print out the image and the generated caption
#     print(sentence)
#     image = Image.open(args.image)
#     plt.imshow(np.asarray(image))
#
#
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--image', type=str, required=True, help='input image for generating caption')
#     parser.add_argument('--encoder_path', type=str, default='models/encoder-5-3000.pkl',
#                         help='path for trained encoder')
#     parser.add_argument('--decoder_path', type=str, default='models/decoder-5-3000.pkl',
#                         help='path for trained decoder')
#     parser.add_argument('--vocab_path', type=str, default='data/vocab.pkl', help='path for vocabulary wrapper')
#
#     # Model parameters (should be same as paramters in train.py)
#     parser.add_argument('--embed_size', type=int, default=256, help='dimension of word embedding vectors')
#     parser.add_argument('--hidden_size', type=int, default=512, help='dimension of lstm hidden states')
#     parser.add_argument('--num_layers', type=int, default=1, help='number of layers in lstm')
#     args = parser.parse_args()
#     main(args)
