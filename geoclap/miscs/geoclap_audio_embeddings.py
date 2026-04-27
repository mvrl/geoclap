#This script computes audio embeddings for the test split of our dataset and saves that as single tensor of dimension: (10159, 1+512), 
# where 1 extra channel dimension stores key-id of the audio in the dataset.

##local imports
from ..train import GeoCLAP, l2normalize
import torch
import numpy as np
import random
import os
from tqdm import tqdm
from ..dataloader import Dataset_soundscape
from argparse import Namespace, ArgumentParser, RawTextHelpFormatter
from ..config import cfg

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


class Audio_Embeds(object):
    def __init__(self, validation, ckpt_path,device):
        super().__init__()
        self.validation = validation
        self.ckpt_path = ckpt_path
        self.device = device
       
    def get_embeddings(self,batch,model,hparams):
        hparams = Namespace(**hparams)
        self.hparams = hparams

        model.sat_encoder.eval()
        for params in model.sat_encoder.parameters():
            params.requires_grad=False

        embeds = {'audio_embeddings':None,'long_id':None}
        batch_audio = {}
        for key in batch['audio'].keys():
            batch_audio[key] = batch['audio'][key].to(self.device) 
        embeds['audio_embeddings'] = l2normalize(model.audio_encoder(batch_audio)).detach().cpu()
        embeds['long_id'] = batch['long_id']
  
        return embeds

    def get_dataloader(self,hparams):
        hparams = Namespace(**hparams)
        if self.validation == 'validation':
            validate_csv = cfg.validate_csv
        else:
            validate_csv = cfg.test_csv
        try: #Initial code/geoclap model did not have text_type argument. Therefore, just a hack to make the code compatible for both scenarios:
            testloader = torch.utils.data.DataLoader(Dataset_soundscape(
                                                                        data_file=validate_csv,
                                                                        is_train = False,
                                                                        sat_input_size= hparams.sat_input_size,
                                                                        sat_model= hparams.sat_encoder,
                                                                        audio_model= hparams.audio_encoder,
                                                                        data_type = hparams.data_type,
                                                                        metadata_type= hparams.metadata_type,
                                                                        saved_audio_embeds= hparams.saved_audio_embeds,
                                                                        saved_text_embeds= hparams.saved_text_embeds,
                                                                        sat_type = hparams.sat_type,
                                                                        text_type = hparams.text_type),
                                            num_workers=16, batch_size=256, shuffle=False, drop_last=False,pin_memory=True)
        except:
            testloader = torch.utils.data.DataLoader(Dataset_soundscape(
                                                                        data_file=validate_csv,
                                                                        is_train = False,
                                                                        sat_input_size= hparams.sat_input_size,
                                                                        sat_model= hparams.sat_encoder,
                                                                        audio_model= hparams.audio_encoder,
                                                                        data_type = hparams.data_type,
                                                                        metadata_type= hparams.metadata_type,
                                                                        saved_audio_embeds= hparams.saved_audio_embeds,
                                                                        saved_text_embeds= hparams.saved_text_embeds,
                                                                        sat_type = hparams.sat_type
                                                                        ),
                                            num_workers=16, batch_size=256, shuffle=False, drop_last=False,pin_memory=True)
        return testloader
    
    def get_geoclap(self):
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
    
    @torch.no_grad()
    def save_audio_embeds(self):
        set_seed(56)
        geoclap, hparams  = self.get_geoclap()
        print(hparams)
        test_dataloader = self.get_dataloader(hparams)
        audio_embeddings = []
        long_ids = []
        for i,batch in tqdm(enumerate(test_dataloader)):
            print("batch no:",str(i))
            embeds = self.get_embeddings(batch=batch,model=geoclap,hparams=hparams)
            audio_embeddings.append(embeds['audio_embeddings'])
            long_ids = long_ids + embeds['long_id']
        short_ids = torch.tensor([int(id.strip().split('_')[-1]) for id in long_ids]).reshape(len(long_ids),1)
        audio_embeddings = torch.cat(audio_embeddings,axis=0)
        final_tensor = torch.cat([short_ids,audio_embeddings],axis=1)
        print(final_tensor.shape)
        torch.save(final_tensor.cpu(),os.path.join(cfg.embeddings_path,'GeoCLAP_gallery_audio_embeds.pt'))


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = ArgumentParser(description='', formatter_class=RawTextHelpFormatter)
    parser.add_argument('--ckpt_path', type=str)
    args = parser.parse_args()
    #params
    set_seed(56)
    #configure evaluation
    gallery_audio_embeds = Audio_Embeds(validation='test', ckpt_path=args.ckpt_path,device=device)
    print(gallery_audio_embeds.save_audio_embeds())