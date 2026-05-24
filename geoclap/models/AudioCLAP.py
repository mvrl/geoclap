#Hugging face way of loading AudioCLAP model

from transformers import ClapAudioModelWithProjection, ClapTextModelWithProjection
import pytorch_lightning as pl
import torch.nn as nn
import torch
import numpy as np

class AudioCLAP_audiomodel(pl.LightningModule):
    def __init__(self,freeze=True):
        super().__init__()
        if freeze:
            self.model = ClapAudioModelWithProjection.from_pretrained("laion/clap-htsat-fused").eval()
        else:
            self.model = ClapAudioModelWithProjection.from_pretrained("laion/clap-htsat-fused").train()
    def forward(self,audio):
        batch_embeddings_audio = self.model(**audio)['audio_embeds']
        return batch_embeddings_audio


class AudioCLAP_textmodel(pl.LightningModule):
    def __init__(self,freeze=True):
        super().__init__()
        if freeze:
            self.model = ClapTextModelWithProjection.from_pretrained("laion/clap-htsat-fused").eval()
        else:
            self.model = ClapTextModelWithProjection.from_pretrained("laion/clap-htsat-fused").train()

    def forward(self,text):
        batch_embeddings_text = self.model(**text)['text_embeds']
        return batch_embeddings_text
    


class temp_layer(pl.LightningModule):
    def __init__(self,hparams):
        super().__init__()
        #instantiate the learnable temperature parameter: Fixed to t = 2.6592; the clip default
        self.logit_scale_ia = nn.Parameter(torch.ones([]) * np.log(1 /hparams.temperature))
        if hparams.data_type == 'sat_audio_text':
            self.logit_scale_it = nn.Parameter(torch.ones([]) * np.log(1 /hparams.temperature))
            if not hparams.freeze_text_model:
                self.logit_scale_at = nn.Parameter(torch.ones([]) * np.log(1 /hparams.temperature))
