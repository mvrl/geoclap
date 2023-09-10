import warnings
warnings.filterwarnings("ignore")
from torch.utils.data import Dataset
import torch
import numpy as np
import pandas as pd
import os
from torchvision.io import read_image
from .config import cfg
from .utilities import clap_data_processor
from .utilities.SATMAE_transform import build_transform as SATMAE_transform
from .models import AudioCLAP
import code

metadata_df = pd.read_csv(cfg.detailed_metadata_path)

def read_csv(data_file):
    
    split_file = pd.read_csv(os.path.join(data_file)) #train/val/test data split csv
    split_file['description'] = split_file['description'].fillna("This is a sound") #Fill missing description with a sample prompt
    short_ids = list(split_file['key'])
    long_ids = list(split_file['long_key'])

    return short_ids, long_ids, split_file

def get_processed_audio(audio_path,sr,model_type='AudioCLAP'):
    if model_type == 'AudioCLAP':
        out = dict(clap_data_processor.get_audio_clap(path_to_audio=audio_path,sr=sr))
    else:
        print(model_type)
        raise NotImplementedError("Only Audio-Text Model selected to be used was CLAP")
    
    return out

def get_processed_text(text, model_type='AudioCLAP'):
    if model_type == 'AudioCLAP':
        out = dict(clap_data_processor.get_text_clap([text]))
    else:
        raise NotImplementedError("Only Audio-Text Model selected to be used was CLAP")
    
    return out

def get_audio_embeddings(long_id,model_type='AudioCLAP'):  
    embed = torch.load(os.path.join(cfg.embeddings_path,model_type+'/audio/',long_id+'.pt'))
    return embed

def get_text_embeddings(long_id,model_type='AudioCLAP',text_type='with_address'): 
    if text_type == 'with_address': 
        embed = torch.load(os.path.join(cfg.embeddings_path,model_type+'/text/',long_id+'.pt'))
    elif text_type == 'without_address':
        embed = torch.load(os.path.join(cfg.embeddings_path,model_type+'/text_without_address/',long_id+'.pt'))
    elif text_type == 'only_address':
        embed = torch.load(os.path.join(cfg.embeddings_path,model_type+'/text_only_address/',long_id+'.pt'))
    else:
        raise NotImplementedError("Only implemented text types are: [with_address, without_address, only_address]")

    return embed

class Dataset_soundscape(Dataset):
    def __init__(self,
                 data_file,                                 # Provide path for metadata of train/validate/test
                 is_train = True,                           # Flag set True if it is train dataloader
                 sat_input_size = 224,                      # Input size of satellite image
                 data_type = 'sat_audio_text',              # data choices: [sat_audio, sat_audio_text]
                 sat_model = 'SatMAE',                      # Choice of satellite image model: [SatMAE]
                 audio_model = 'AudioCLAP',                 # Choice of text_audio model: [AudioCLAP]
                 metadata_type = 'lat_long',                # What extra metadata to pass, currently only supports: [lat_long, None]
                 saved_audio_embeds= False,
                 saved_text_embeds = False,
                 sat_type = 'SoundingEarth',                #what type of satellite image to use: [SoundingEarth, sentinel]
                 text_type = 'with_address'                 #What type of text prompt to use: [with_address, without_address, only_address]
                 ):    
        self.short_ids, self.long_ids, self.data_file   = read_csv(data_file)
        self.data_type = data_type
        self.sat_transform = SATMAE_transform(is_train=is_train, input_size=sat_input_size)
        self.audio_model = audio_model
        self.sat_model = sat_model
        self.metadata_type = metadata_type
        self.saved_audio_embeds = saved_audio_embeds
        self.saved_text_embeds = saved_text_embeds
        self.sat_type = sat_type
        self.text_type = text_type

    def __len__(self):
        return len(self.short_ids)
    def __getitem__(self,idx):
        sample = self.data_file.iloc[idx]
        mp3name = sample.mp3name
        if self.text_type == 'with_address':
            caption = sample.caption
        elif self.text_type == 'only_address':
            #address = ["The location of the sound is" + caption.split("location of the sound is")[1] for caption in captions]
            caption = sample.address 
        else:
            caption = sample.description
        short_id = sample.key
        long_id = sample.long_key
        samplerate = sample.mp3samplerate
        lon = np.radians(sample.longitude)
        lat = np.radians(sample.latitude)
        x = np.cos(lat) * np.cos(lon)
        y = np.cos(lat) * np.sin(lon)
        z = np.sin(lat)
        v = torch.from_numpy(np.stack([x, y, z])).float()

        if self.sat_type == 'SoundingEarth':
            sat_img = read_image(os.path.join(cfg.data_path,'images','googleEarth',str(short_id)+'.jpg'))
        elif self.sat_type == 'sentinel':
            sat_img = read_image(os.path.join(cfg.data_path,'images','sentinel_geoclap',str(short_id)+'.jpeg'))
        else:
            raise NotImplementedError("supported satellite image types are:[SoundingEarth, sentinel]")
        if self.sat_model == 'SatMAE':
            sat_img = np.array(torch.permute(sat_img,[1,2,0]))
            sat_img = self.sat_transform(sat_img)

        audio_path = os.path.join(cfg.sat_audio_tensors_path,long_id+'.pt')
        out_dict = {}
        out_dict['long_id'] = long_id
        if 'sat' in self.data_type:
            out_dict['sat']= sat_img
        if 'audio' in self.data_type:
            if self.saved_audio_embeds:
                out_dict['audio'] = get_audio_embeddings(long_id,model_type=self.audio_model)
            else:
                out_dict['audio'] = get_processed_audio(audio_path=audio_path,sr=samplerate,model_type=self.audio_model)
        if self.metadata_type == 'lat_long':
             out_dict['lat_long'] = v
        if 'text' in self.data_type:
            if self.saved_text_embeds:
                out_dict['text'] = get_text_embeddings(long_id,model_type=self.audio_model,text_type=self.text_type)
            else:
                out_dict['text']= get_processed_text(text=caption, model_type= self.audio_model)
            
        return out_dict


if __name__ == '__main__':
    ## Test the dataloaders
    data_type = 'sat_audio_text'                                # Options: sat_audio, sat_audio_text
    sat_input_size = 224
    sat_model = 'SatMAE'                                        # Options: SatMAE
    audio_model = 'AudioCLAP'                                   # Options: AudioCLAP
    saved_audio_embeds = False
    saved_text_embeds = False
    metadata_type = 'lat_long'                                  # Options: none, lat_long

    train_csv = os.path.join(cfg.data_path,'train_df.csv')
    validate_csv = os.path.join(cfg.data_path,'validate_df.csv')
    test_csv = os.path.join(cfg.data_path,'test_df.csv')

    is_train = True
    train_loader = torch.utils.data.DataLoader(Dataset_soundscape(data_file=train_csv,
                                                                    is_train = is_train,
                                                                    sat_input_size= sat_input_size,
                                                                    sat_model= sat_model,
                                                                    audio_model= audio_model,
                                                                    metadata_type= metadata_type,
                                                                    saved_audio_embeds= saved_audio_embeds,
                                                                    saved_text_embeds= saved_text_embeds,
                                                                    sat_type = 'sentinel'),
                                            num_workers=2, batch_size=2, shuffle=True, drop_last=False,pin_memory=True)
    is_train = False
    valid_loader = torch.utils.data.DataLoader(Dataset_soundscape(data_file=validate_csv,
                                                                    is_train = is_train,
                                                                    sat_input_size= sat_input_size,
                                                                    sat_model= sat_model,
                                                                    audio_model= audio_model,
                                                                    metadata_type= metadata_type,
                                                                    saved_audio_embeds= saved_audio_embeds,
                                                                    saved_text_embeds= saved_text_embeds,
                                                                    sat_type = 'sentinel'),
                                            num_workers=2, batch_size=2, shuffle=True, drop_last=False,pin_memory=True)

    test_loader = torch.utils.data.DataLoader(Dataset_soundscape(data_file=test_csv,
                                                                is_train = is_train,
                                                                sat_input_size= sat_input_size,
                                                                sat_model= sat_model,
                                                                audio_model= audio_model,
                                                                metadata_type= metadata_type,
                                                                saved_audio_embeds= saved_audio_embeds,
                                                                saved_text_embeds= saved_text_embeds,
                                                                sat_type = 'sentinel'),
                                            num_workers=2, batch_size=2, shuffle=True, drop_last=False,pin_memory=True)

    batch = next(iter(test_loader))
    print(type(batch['sat']),type(batch['audio']),(type(batch['text'])))
    # import code;code.interact(local=dict(globals(), **locals()));
    if not saved_audio_embeds :
        print(batch['audio'].keys())
        print(batch['audio']['input_features'].shape, batch['audio']['is_longer'].shape)
    else:
         print(batch['audio'].shape)
    
    if not saved_text_embeds :
        print(batch['text'].keys())
        print(batch['text']['input_ids'].shape, batch['text']['attention_mask'].shape)
    else:
         print(batch['text'].shape)

    print(batch['long_id'], batch['sat'].shape ,batch['lat_long'].shape)
    
    ## Assuming embeddings were not saved already
    # audio_encoder = AudioCLAP.AudioCLAP_audiomodel(freeze=True).eval()
    # text_encoder = AudioCLAP.AudioCLAP_textmodel(freeze=True).eval()

    # audio_embeds = audio_encoder(audio=batch['audio'])
    # text_embeds = text_encoder(text=batch['text'])
    # print(audio_embeds.shape,text_embeds.shape) #torch.Size([2, 512]) torch.Size([2, 512])

