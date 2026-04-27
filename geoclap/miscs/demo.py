#Note: This script produced results in supplementary materials of the paper.

# This script is a simple demonstration of the utility of the GeoCLAP model.
# Given a region file (a .csv file with columns latitude, longitude for the image tiles in a region of interest), and sound categories given as a set of textual queries seperated by ';'
# This script returns an output .csv file containing: similarity of every image in the region with textual query of sound categories, as well as IDs of top 5 audio (and their corresponding text) retrevied from our SoundingEarth test set.

# For audio-retrevial part of the demo, this script assumes avaliablity of all the embeddings of test set precomputed and saved as a single tensor. (using the script geoclap_audio_embeddings.py)

from argparse import ArgumentParser, RawTextHelpFormatter
from ..utilities.SATMAE_transform import *
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from ..train import GeoCLAP, l2normalize
from tqdm import tqdm
import pandas as pd
from torchvision.io import read_image
from ..config import cfg
import random
from ..utilities import clap_data_processor
from ..utilities.SATMAE_transform import build_transform as SATMAE_transform
audio_gallery_embeds = cfg.GeoCLAP_gallery_audio_embeds #tensor of dimension: (10159, 1+512), where 1 extra channel dimension in the beginning stores key id of the audio in the dataset.
N_audio_samples = 5
meta_df = pd.read_csv(cfg.detailed_metadata_path)

def read_csv(csv_path,data_path):
    region_file = pd.read_csv(os.path.join(csv_path))
    ids = list(region_file['key'])
    paths = [os.path.join(data_path,str(id)+'.jpeg') for id in ids]
    lats = list(region_file['latitude'])
    longs = list(region_file['longitude'])
    return ids, paths, lats, longs

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

def format_topk_audios(audio_keys,sat_keys,sat_lats,sat_longs,topk_scores):
    #This function receives topk_scores obtained by the operation: torch.matmul(audio_gallery[:,1:],sat_embeds.t())
    #Finds the corresponding IDs of topk audios and returns: topk_audios, topk_sim, topk_texts
    sim_values = topk_scores.values         #eg. shape 5 x 100 ; for k=5 and number of sat images in the regions=100
    audio_key_indices = topk_scores.indices #eg. shape 5 x 100 ; for k=5 and number of sat images in the regions=100
    out_df = pd.DataFrame(columns=['sat_key','latitude','longitude','audio_keys','sat_audio_sim','audio_texts'])
    out_df['sat_key'] = sat_keys
    out_df['latitude'] = sat_lats
    out_df['longitude'] = sat_longs
    audio_long_keys = []
    sat_audio_sims = []
    audio_texts = []
    for im in tqdm(range(sim_values.shape[1])):
        sim_vals = list(sim_values[:,im])
        sim_vals = [str(round(v.item(),4)) for v in sim_vals]
        sim_vals = ';'.join(sim_vals)
        sat_audio_sims.append(sim_vals)

        audio_keys_index = [int(i) for i in list(audio_key_indices[:,im])]
        audio_keys_int =[audio_keys[ix] for ix in audio_keys_index]
        # import code;code.interact(local=dict(globals(), **locals()));
        audio_captions = [meta_df[meta_df['key']==audio_key_int].caption.item() for audio_key_int in audio_keys_int]
        audio_captions = ';'.join(audio_captions)
        audio_texts.append(audio_captions)

        audio_long_key = [meta_df[meta_df['key']==audio_key_int].long_key.item() for audio_key_int in audio_keys_int]
        audio_long_key = ';'.join(audio_long_key)
        audio_long_keys.append(audio_long_key)
 
    out_df['audio_keys'] = audio_long_keys
    out_df['sat_audio_sim'] = sat_audio_sims
    out_df['audio_texts'] = audio_texts

    return out_df


class GeoCLAP_Demo(object):
    def __init__(self,ckpt_path,saved_embeds_path,sat_data_path,region_file_path,output_filename,device,query_type,text_query):
        super().__init__()
        self.ckpt_path = ckpt_path
        self.saved_embeds_path = saved_embeds_path
        self.sat_data_path= sat_data_path
        self.region_file_path = region_file_path
        self.output_filename = output_filename
        self.device = device
        self.text_query = text_query
        self.query_type = query_type
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
        audio_embeds = torch.load(audio_gallery_embeds)
        keys = audio_embeds[:,0]
        embeds = audio_embeds[:,1:]
        keys = [int(k) for k in keys]
        return keys, embeds
    
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
                                            num_workers=8, batch_size=512, shuffle=False, drop_last=False, pin_memory=True)
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
        out['latitude'] = [l.item() for l in sat_lats]
        out['longitude'] = [l.item() for l in sat_longs]
        out['key'] = [k.item() for k in sat_keys]
        return out
    
    def get_cosine_sim(self):
        sat_embeds = self.get_sat_embeddings()
        latitudes = sat_embeds['latitude']
        longitudes = sat_embeds['longitude']
        sat_keys = sat_embeds['key']
        if 'audio' in self.query_type:
            print("Perfroming topk retrevial using embeddings of our audio gallery")
            audio_keys, audio_embeds = self.get_audio_embeddings()
            sim_audio_sat = torch.matmul(audio_embeds.to(device),sat_embeds['sat_embeddings'].t()).detach().cpu()
            topk_scores = torch.topk(sim_audio_sat,N_audio_samples,dim=0)
            out_df = format_topk_audios(audio_keys=audio_keys,sat_keys=sat_keys,sat_lats=latitudes,sat_longs=longitudes,topk_scores=topk_scores)
            out_df.to_csv(self.region_file_path.replace('.csv','_sim_audio_sat_detailed.csv'))

        if 'text' in self.query_type:
            text_embeds = self.get_text_embeddings()
            text_classes = text_embeds['classes']
            sim_text_sat = torch.matmul(text_embeds['text_embeddings'],sat_embeds['sat_embeddings'].t()).detach().cpu()
            columns = ['keys','latitude','longitude'] + text_classes
            df_sim = pd.DataFrame(columns=columns)
            df_sim['keys'] = sat_keys
            df_sim['latitude'] = latitudes
            df_sim['longitude'] = longitudes
            df_sim[text_classes] = np.transpose(sim_text_sat)

            if self.output_filename == 'none':
                df_sim.to_csv(self.region_file_path.replace('.csv','_sim_text_sat_detailed.csv'))
            else:
                df_sim.to_csv(self.region_file_path.replace('.csv',self.output_filename+'_sim_text_sat_detailed.csv'))
        else:
            NotImplementedError("Only audio and text query supported")

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = ArgumentParser(description='', formatter_class=RawTextHelpFormatter)
    parser.add_argument('--ckpt_path', type=str)
    parser.add_argument('--saved_embeds_path', type=str)
    parser.add_argument('--region_file_path', type=str)
    parser.add_argument('--output_filename', type=str, default='demofile')
    parser.add_argument('--sat_data_path', type=str)
    parser.add_argument('--text_query', type=str, default="car horn;chirping birds;animal farm") #Provide text query(s) in the format same as the default option here
    parser.add_argument('--query_type', type=str, default='audio_text')                          #Options:[audio_text, text, audio]
    args = parser.parse_args()


    sim = GeoCLAP_Demo(ckpt_path = args.ckpt_path,
                    saved_embeds_path = args.saved_embeds_path,
                    sat_data_path= args.sat_data_path,
                    region_file_path = args.region_file_path,
                    output_filename = args.output_filename,
                    device = device,
                    text_query = args.text_query,
                    query_type = args.query_type,
                    )
    sim.get_cosine_sim()