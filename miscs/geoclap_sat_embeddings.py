# Given the checkpoint of a pretrained GeoCLAP model, this script computes and saves overhead-imagery embeddings for all the images indexed in a region file csv.
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
from ..utilities.SATMAE_transform import build_transform as SATMAE_transform


def read_csv(csv_path,data_path):
    region_file = pd.read_csv(os.path.join(csv_path))
    ids = list(region_file['key'])
    paths = [os.path.join(data_path,str(id)+'.jpeg') for id in ids]
    lats = list(region_file['latitude'])
    longs = list(region_file['longitude'])
    return ids, paths, lats, longs

class Dataset_sentinel(Dataset):
    def __init__(self, region_file,data_path):
        super().__init__()
        self.keys, self.img_paths, self.lats, self.longs = read_csv(region_file,data_path)
        self.sat_transform = SATMAE_transform(is_train=False, input_size=224)
    def __len__(self):
        return len(self.img_paths)
    def __getitem__(self, idx):
        out = {'key':None,'sat':None, 'latitude':None,'longitude':None}
        sat_img = read_image(self.img_paths[idx])
        sat_img = np.array(torch.permute(sat_img,[1,2,0]))
        sat_img = self.sat_transform(sat_img)
        out['key'] = self.keys[idx]
        out['sat'] = sat_img
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

class GeoCLAP_embeds(object):
    def __init__(self,ckpt_path,sat_data_path,region_file_path,save_embeds_path,device):
        super().__init__()
        self.ckpt_path = ckpt_path
        self.sat_data_path= sat_data_path
        self.region_file_path = region_file_path
        self.save_embeds_path = save_embeds_path
        self.device = device
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

    
    def get_sat_embeddings(self):
        predloader = DataLoader(Dataset_sentinel(region_file=self.region_file_path,data_path=self.sat_data_path),
                                            num_workers=8, batch_size=5, shuffle=False, drop_last=False, pin_memory=True)
        for batch in tqdm(predloader):
            long_ids = list(batch['key'].cpu())
            batch_embeds = l2normalize(self.model.sat_encoder(batch['sat'].to(self.device))).cpu()
            bs = batch_embeds.shape[0]
            for b in range(bs):
                long_id = int(long_ids[b])
                torch.save(batch_embeds[b,:],os.path.join(self.save_embeds_path,str(long_id)+'.pt'))
            

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = ArgumentParser(description='', formatter_class=RawTextHelpFormatter)
    parser.add_argument('--ckpt_path', type=str,help='path to the GeoCLAP ckpt')
    parser.add_argument('--region_file_path', type=str,help='path to the region file over which GeoCLAP satellite embeddings are to be computed')
    parser.add_argument('--sat_data_path', type=str,help='path of directory where the satellite images for the region of interest were downloaded')
    parser.add_argument('--save_embeds_path', type=str,help='path of directory where satellite image embeddings are to be saved')

    args = parser.parse_args()
    embed_class = GeoCLAP_embeds(ckpt_path=args.ckpt_path,
                                 sat_data_path=args.sat_data_path,
                                 region_file_path=args.region_file_path,
                                 save_embeds_path=args.save_embeds_path,
                                 device=device)
    
    embed_class.get_sat_embeddings()
    



    
   