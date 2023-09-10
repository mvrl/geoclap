from transformers import ClapProcessor
import torchaudio
import torch
import os
from transformers import RobertaTokenizer
import numpy as np
import warnings
warnings.filterwarnings("ignore")

tokenize = RobertaTokenizer.from_pretrained('roberta-base')
def tokenizer(text):
    result = tokenize(
        text,
        padding="max_length",
        truncation=True,
        max_length=77,
        return_tensors="pt",
    )
    return {k: v.squeeze(0) for k, v in result.items()}

def get_text_clap(text):#accepts list of text prompts
    text = tokenizer(text)
    return text


processor = ClapProcessor.from_pretrained("laion/clap-htsat-fused")
SAMPLE_RATE = 48000

def get_audio_clap(path_to_audio,sr=41000):
    track = torch.load(path_to_audio)  # Faster!!!
    track = track.mean(axis=0)
    track = torchaudio.functional.resample(track, orig_freq=sr, new_freq=SAMPLE_RATE)
    output = processor(audios=track, sampling_rate=SAMPLE_RATE, max_length_s=10, return_tensors="pt",padding=True)
    output['input_features'] = output['input_features'][0,:,:,:]
    return output


if __name__ == '__main__':
    path_to_audio = '/home/k.subash/storage/data/aporee/raw_audio_tensors/aporee_3893_5243.pt'
    sample =  get_audio_clap(path_to_audio)
    print(sample.keys())
    print(sample['input_features'].shape,sample['is_longer'].shape)

    print(get_text_clap(['dummy text'])['input_ids'].shape)
    print(get_text_clap(['dummy text'])['attention_mask'].shape)
    



