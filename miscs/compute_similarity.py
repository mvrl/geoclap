#Note: This script was used to obtain results in the main paper.

# This script computes similarity between a set of (textual or audio) query embeddings with overhead imagery for all locations listed as latitude longitude in a .csv file for the region of interest.
# Once this similarity matrix is computed one can build heatmaps of these similarities and overlay on top of basemap of the region of interest.

#This code currently only supports:
#1. audio query randomly sample from following classes of ESC50 dataset: ["car_horn","chirping_birds","fireworks","sea_waves"]. These classes can be changed by changing cfg.heatmap_classes.
#2. textual query prompted from classes: ["car_horn","chirping_birds","fireworks","sea_waves"] or any free-form textual query

from argparse import ArgumentParser, RawTextHelpFormatter
from ..utilities.SATMAE_transform import *
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import torch.nn.functional as F
from ..train import GeoCLAP, l2normalize
from tqdm import tqdm
import pandas as pd
from torchvision.io import read_image
from ..config import cfg
import random
from ..utilities import clap_data_processor
from ..utilities.SATMAE_transform import build_transform as SATMAE_transform

N_audio_sample = 5
def stratified_sample_df(df, col, n_samples):
    n = min(n_samples, df[col].value_counts().min())
    df_ = df.groupby(col).apply(lambda x: x.sample(n))
    df_.index = df_.index.droplevel(0)
    return df_

def read_csv(csv_path,data_path):
    region_file = pd.read_csv(os.path.join(csv_path))
    ids = list(region_file['key'])
    paths = [os.path.join(data_path,str(id)+'.jpeg') for id in ids]
    lats = list(region_file['latitude'])
    longs = list(region_file['longitude'])
    return ids, paths, lats, longs

def get_processed_audio(audio_path,sr,model_type='AudioCLAP'):
    if model_type == 'AudioCLAP':
        out = dict(clap_data_processor.get_audio_clap(path_to_audio=audio_path,sr=sr))
    else:
        print(model_type)
        raise NotImplementedError("Only Audio-Text Model selected to be used was CLAP")
    
    return out

class Dataset_sentinel(Dataset):
    def __init__(self, saved_embeds_path, region_file,data_path):
        super().__init__()
        self.saved_embeds_path = saved_embeds_path
        self.keys, self.img_paths, self.lats, self.longs = read_csv(region_file,data_path)
        self.sat_transform = SATMAE_transform(is_train=False, input_size=224)
    def __len__(self):
        return len(self.img_paths)
    def __getitem__(self, idx):
        if self.saved_embeds_path == 'none':
            out = {'key':None,'sat':None, 'latitude':None,'longitude':None}
            sat_img = read_image(self.img_paths[idx])
            sat_img = np.array(torch.permute(sat_img,[1,2,0]))
            sat_img = self.sat_transform(sat_img)
            out['key'] = self.keys[idx]
            out['sat'] = sat_img
            out['latitude'] = self.lats[idx]
            out['longitude'] = self.longs[idx]
        else:
            out = {'key':None,'sat_embed':None, 'latitude':None,'longitude':None}
            key = self.keys[idx]
            sat_embed = torch.load(os.path.join(self.saved_embeds_path,str(key)+'.pt'))
            out['key'] = key
            out['sat_embed'] = sat_embed
            out['latitude'] = self.lats[idx]
            out['longitude'] = self.longs[idx]
        return out
    
class Dataset_ESC(Dataset):
    def __init__(self,
                 sub_meta):       
        self.metadata = sub_meta
        self.metadata = stratified_sample_df(df=self.metadata,col='category',n_samples=N_audio_sample)

    def __len__(self):
        return len(self.metadata)
    def __getitem__(self,idx):
        sample = self.metadata.iloc[idx]
        wavname = sample.filename.replace('.wav','.pt')
        wavpath = os.path.join(cfg.esc_audio_tensors_path,wavname)
        samplerate = sample.samplerate
        out_dict = {}
        out_dict['audio'] = get_processed_audio(audio_path=wavpath,sr=samplerate,model_type='AudioCLAP')
        out_dict['filename'] = sample.category+'_'+sample.filename
        return out_dict
    
def set_seed(seed: int = 56) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

class Similarity(object):
    def __init__(self,ckpt_path,saved_embeds_path,esc_audio_tensors_path,esc_data_path,sat_data_path,heatmap_classes,region_file_path,device,query_type,text_query=''):
        super().__init__()
        self.ckpt_path = ckpt_path
        self.saved_embeds_path = saved_embeds_path
        self.esc_audio_tensors_path = esc_audio_tensors_path
        self.esc_data_path = esc_data_path
        self.sat_data_path= sat_data_path
        self.region_file_path = region_file_path
        self.device = device
        self.text_query = text_query
        self.query_type = query_type
        self.audio_query_classes = heatmap_classes
        self.esc_metadata = pd.read_csv(os.path.join(self.esc_data_path,'detailed_esc_subset.csv'))
        self.model, self.hparams = self.get_geoclap()

    def get_geoclap(self):
        set_seed()
        #load geoclap model from checkpoint
        pretrained_ckpt = torch.load(self.ckpt_path)
        hparams = pretrained_ckpt['hyper_parameters']
        pretrained_weights = pretrained_ckpt['state_dict']
        model = GeoCLAP(hparams).to(self.device)
        model.load_state_dict(pretrained_weights)
        geoclap = model.eval()
        #set all requires grad to false
        for params in geoclap.parameters():
            params.requires_grad=False
        return geoclap, hparams 
       
    def get_audio_embeddings(self):
        filenames =[]
        audio_embeddings = []
        audio_dataloader = torch.utils.data.DataLoader(Dataset_ESC(self.esc_metadata),
                                    num_workers=1, batch_size=N_audio_sample*len(self.audio_query_classes), shuffle=False, drop_last=False,pin_memory=True)
        for _,batch in enumerate(audio_dataloader):
            batch_audio = {}
            for key in batch['audio'].keys():
                batch_audio[key] = batch['audio'][key].to(self.device)
            batch_embed = l2normalize(self.model.audio_encoder(batch_audio))
            audio_embeddings.append(batch_embed)
            filenames = filenames + batch['filename']
        audio_embeds = {}
        audio_embeds['audio_embeddings'] = torch.vstack(audio_embeddings).to(self.device)
        audio_embeds['filenames'] = filenames
        return audio_embeds
    
    def get_text_embeddings(self):
        text_embeds = {'text_embeddings':None,'classes':None}
        text_queries = self.text_query.split(';')
        text_queries = [q.strip() for q in text_queries]
        text_prompts = ['The is a sound of '+q for q in text_queries]
        processed_text = clap_data_processor.get_text_clap(text_prompts)
        text_input ={}
        for key in processed_text.keys():
            text_input[key] = processed_text[key].to(self.device)
        text_embeds['text_embeddings'] = l2normalize(self.model.text_encoder(text_input))
        text_embeds['classes'] = text_queries
        return text_embeds
    
    def get_sat_embeddings(self):
        predloader = DataLoader(Dataset_sentinel(saved_embeds_path=self.saved_embeds_path,region_file=self.region_file_path,data_path=self.sat_data_path),
                                            num_workers=8, batch_size=512, shuffle=True, drop_last=False, pin_memory=True)
        out = {'key':None,'sat_embeddings':None, 'latitude':None, 'longitude':None}
        sat_embeddings = []
        sat_keys = []
        sat_lats = []
        sat_longs = []
        for batch in tqdm(predloader):
            if self.saved_embeds_path == 'none':
                batch_embeds = l2normalize(self.model.sat_encoder(batch['sat'].to(self.device)))
            else:
                batch_embeds = batch['sat_embed']
            sat_embeddings.append(batch_embeds)
            sat_keys.extend(batch['key'])
            sat_lats.extend(batch['latitude'])
            sat_longs.extend(batch['longitude'])      
        out['sat_embeddings'] = torch.vstack(sat_embeddings).to(self.device)
        out['latitude'] = sat_lats
        out['longitude'] = sat_longs
        out['key'] = sat_keys
        return out
    
    def get_cosine_sim(self):
        sat_embeds = self.get_sat_embeddings()
        latitudes = sat_embeds['latitude']
        longitudes = sat_embeds['longitude']
        keys = sat_embeds['key']
        audio_filenames = []
        text_classes = []
        if 'audio' in self.query_type:
            audio_embeds = self.get_audio_embeddings()
            audio_filenames = audio_embeds['filenames']
            sim_audio_sat = torch.matmul(audio_embeds['audio_embeddings'],sat_embeds['sat_embeddings'].t()).detach().cpu().numpy()
            columns = ['keys','latitude','longitude'] + audio_filenames
            df_sim = pd.DataFrame(columns=columns)
            df_sim['keys'] = [k.item() for k in keys]
            df_sim['latitude'] = [l.item() for l in latitudes]
            df_sim['longitude'] = [l.item() for l in longitudes]
            df_sim[audio_filenames] = np.transpose(sim_audio_sat)
            df_sim.to_csv(self.region_file_path.replace('.csv','_sim_audio_sat.csv'))

        if 'text' in self.query_type:
            text_embeds = self.get_text_embeddings()
            text_classes = text_embeds['classes']
            sim_text_sat = torch.matmul(text_embeds['text_embeddings'],sat_embeds['sat_embeddings'].t()).detach().cpu().numpy()
            columns = ['keys','latitude','longitude'] + text_classes
            df_sim = pd.DataFrame(columns=columns)
            df_sim['keys'] = [k.item() for k in keys]
            df_sim['latitude'] = [l.item() for l in latitudes]
            df_sim['longitude'] = [l.item() for l in longitudes]
            df_sim[text_classes] = np.transpose(sim_text_sat)
            df_sim.to_csv(self.region_file_path.replace('.csv','_sim_text_sat.csv'))
        else:
            NotImplementedError("Only audio and text query supported")

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = ArgumentParser(description='', formatter_class=RawTextHelpFormatter)
    parser.add_argument('--ckpt_path', type=str)
    parser.add_argument('--saved_embeds_path', type=str)
    parser.add_argument('--region_file_path', type=str)
    parser.add_argument('--sat_data_path', type=str)
    parser.add_argument('--text_query', type=str, default="car horn;chirping birds;animal farm") #Provide text query(s) in the format same as the default option here
    parser.add_argument('--query_type', type=str, default='audio_text')                          #Options:[audio_text, text, audio]
    args = parser.parse_args()

    sim = Similarity(ckpt_path = args.ckpt_path,
                    saved_embeds_path = args.saved_embeds_path,
                    esc_audio_tensors_path = cfg.esc_audio_tensors_path,
                    esc_data_path = cfg.esc_data_path,
                    sat_data_path= args.sat_data_path,
                    region_file_path = args.region_file_path,
                    device = device,
                    heatmap_classes = cfg.heatmap_classes,
                    text_query = args.text_query,
                    query_type = args.query_type,
                    )
    sim.get_cosine_sim()




    
   