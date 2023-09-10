# This script just extracts and saves CLAP embeddings for audio and captions of SoundingEarth dataset.
# This will enable faster training of GeoCLAP with frozen CLAP.
import os
from ..config import cfg
from ..models import AudioCLAP
import torch
from tqdm import tqdm
from ..dataloader import Dataset_soundscape

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
audio_encoder = AudioCLAP.AudioCLAP_audiomodel(freeze=True).eval().to(device)
text_encoder = AudioCLAP.AudioCLAP_textmodel(freeze=True).eval().to(device)
for params in audio_encoder.parameters():
    params.requires_grad=False

for params in text_encoder.parameters():
    params.requires_grad=False

## Test the dataloaders
data_type = 'audio_text'                              # Options: sat_audio, sat_audio_text
sat_input_size = 224
sat_model = 'SatMAE'                                  # Options: SatMAE
audio_model = 'AudioCLAP'                             # Options: AudioCLAP
metadata_type = 'none'                                # Options: none, lat_long

data_csv = os.path.join(cfg.data_path,'final_metadata_with_captions.csv')
audio_embeds_path = os.path.join(cfg.embeddings_path,'AudioCLAP/audio/')
if not os.path.isdir(audio_embeds_path):
    os.makedirs(audio_embeds_path, exist_ok=True)

textwaddr_embeds_path = os.path.join(cfg.embeddings_path,'AudioCLAP/text/')
if not os.path.isdir(textwaddr_embeds_path):
    os.makedirs(textwaddr_embeds_path, exist_ok=True)

textwoaddr_embeds_path = os.path.join(cfg.embeddings_path,'AudioCLAP/text_without_address/')
if not os.path.isdir(textwoaddr_embeds_path):
    os.makedirs(textwoaddr_embeds_path, exist_ok=True)

textaddr_embeds_path = os.path.join(cfg.embeddings_path,'AudioCLAP/text_only_address/')
if not os.path.isdir(textaddr_embeds_path):
    os.makedirs(textaddr_embeds_path, exist_ok=True)

is_train = False
data_loader = torch.utils.data.DataLoader(Dataset_soundscape(data_file=data_csv,
                                                                    is_train = is_train,
                                                                    sat_input_size= 224,
                                                                    sat_model= 'SatMAE',
                                                                    audio_model= 'AudioCLAP',
                                                                    data_type = 'sat_audio_text',
                                                                    metadata_type= 'none',
                                                                    saved_audio_embeds= False,
                                                                    saved_text_embeds= False,
                                                                    sat_type = 'sentinel',
                                                                    text_type = 'with_address'),
                                        num_workers=16, batch_size=256, shuffle=False, drop_last=False,pin_memory=True)


for batch_idx, (batch) in tqdm(enumerate(data_loader)):
    long_ids = batch['long_id']
    batch_text = {}
    for key in batch['text'].keys():
        batch_text[key] = batch['text'][key].to(device)
    text_embeds = text_encoder(batch_text)
    audio_embeds = audio_encoder(audio=batch['audio'])
    batch_size = len(long_ids)
    for i in tqdm(range(batch_size)):
        long_id = long_ids[i]
        #save embeddings
        torch.save(audio_embeds[i,:].cpu(),os.path.join(cfg.embeddings_path,'AudioCLAP/audio/',long_id+'.pt'))
        torch.save(text_embeds[i,:].cpu(),os.path.join(cfg.embeddings_path,'AudioCLAP/text/',long_id+'.pt'))


#Now do the same for text without address
data_loader = torch.utils.data.DataLoader(Dataset_soundscape(data_file=data_csv,
                                                                    is_train = is_train,
                                                                    sat_input_size= 224,
                                                                    sat_model= 'SatMAE',
                                                                    audio_model= 'AudioCLAP',
                                                                    data_type = 'sat_audio_text',
                                                                    metadata_type= 'none',
                                                                    saved_audio_embeds= False,
                                                                    saved_text_embeds= False,
                                                                    sat_type = 'sentinel',
                                                                    text_type = 'without_address'),
                                        num_workers=16, batch_size=256, shuffle=False, drop_last=False,pin_memory=True)

for batch_idx, (batch) in tqdm(enumerate(data_loader)):
    long_ids = batch['long_id']
    batch_text = {}
    for key in batch['text'].keys():
        batch_text[key] = batch['text'][key].to(device)
    text_embeds = text_encoder(batch_text)
    batch_size = len(long_ids)
    for i in tqdm(range(batch_size)):
        long_id = long_ids[i]
        #save embeddings
        torch.save(text_embeds[i,:].cpu(),os.path.join(cfg.embeddings_path,'AudioCLAP/text_without_address/',long_id+'.pt'))



#Now do the same for text containing only address
data_loader = torch.utils.data.DataLoader(Dataset_soundscape(data_file=data_csv,
                                                                    is_train = is_train,
                                                                    sat_input_size= 224,
                                                                    sat_model= 'SatMAE',
                                                                    audio_model= 'AudioCLAP',
                                                                    data_type = 'sat_audio_text',
                                                                    metadata_type= 'none',
                                                                    saved_audio_embeds= False,
                                                                    saved_text_embeds= False,
                                                                    sat_type = 'sentinel',
                                                                    text_type = 'only_address'),
                                        num_workers=16, batch_size=256, shuffle=False, drop_last=False,pin_memory=True)

for batch_idx, (batch) in tqdm(enumerate(data_loader)):
    long_ids = batch['long_id']
    batch_text = {}
    for key in batch['text'].keys():
        batch_text[key] = batch['text'][key].to(device)
    text_embeds = text_encoder(batch_text)
    batch_size = len(long_ids)
    for i in tqdm(range(batch_size)):
        long_id = long_ids[i]
        #save embeddings
        torch.save(text_embeds[i,:].cpu(),os.path.join(cfg.embeddings_path,'AudioCLAP/text_only_address/',long_id+'.pt'))