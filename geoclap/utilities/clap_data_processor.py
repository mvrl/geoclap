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

def get_audio_clap(path_to_audio,padding="repeatpad",truncation="fusion"):
    track, sr = torchaudio.load(path_to_audio,format="mp3")
    track = track.mean(axis=0)
    track = torchaudio.functional.resample(track, orig_freq=sr, new_freq=SAMPLE_RATE)
    output = processor(audios=track, sampling_rate=SAMPLE_RATE, max_length_s=10, return_tensors="pt",padding=padding,truncation=truncation)
    output['input_features'] = output['input_features'][0,:,:,:]
    return output


if __name__ == '__main__':
    path_to_audio = '/storage1/fs1/jacobsn/Active/user_k.subash/data_archive/aporee/raw_audio/aporee_7549_9294/BBosuilWesteinde1.mp3'
    sample =  get_audio_clap(path_to_audio)
    print(sample.keys())
    print(sample['input_features'].shape,sample['is_longer'].shape)

    print(get_text_clap(['dummy text'])['input_ids'].shape)
    print(get_text_clap(['dummy text'])['attention_mask'].shape)
    



