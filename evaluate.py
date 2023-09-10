##local imports
from .metrics import get_retrevial_metrics
from .train import GeoCLAP, l2normalize
import torch
import numpy as np
import random
import os
import sys
from tqdm import tqdm
from .dataloader import Dataset_soundscape
from argparse import Namespace, ArgumentParser, RawTextHelpFormatter
from .config import cfg

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


class Evaluate(object):
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

        embeds = {'sat_embeddings':None, 'audio_embeddings':None, 'text_embeddings':None}

        if self.hparams.metadata_type == 'lat_long':
            embeds['sat_embeddings']  = l2normalize(model.sat_encoder(batch['sat'].to(self.device),batch['lat_long'].to(self.device)))
        else:
            embeds['sat_embeddings']  = l2normalize(model.sat_encoder(batch['sat'].to(self.device)))

        if self.hparams.data_type == 'sat_audio':
            output = {}
            if self.hparams.saved_audio_embeds:
                output['audio_embeddings'] = batch['audio'].to(self.device)
            else:
                batch_audio = {}
                for key in batch['audio'].keys():
                    batch_audio[key] = batch['audio'][key].to(self.device)

                output['audio_embeddings'] = model.audio_encoder(batch_audio)
            
            embeds['audio_embeddings'] = l2normalize(output['audio_embeddings'])

        if self.hparams.data_type == 'sat_audio_text':
            output = {}
            if self.hparams.saved_audio_embeds:
                output['audio_embeddings'] = batch['audio']
            else:
                batch_audio = {}
                for key in batch['audio'].keys():
                    batch_audio[key] = batch['audio'][key].to(self.device) 
                output['audio_embeddings'] = model.audio_encoder(batch_audio)
            
            if self.hparams.saved_text_embeds:
                output['text_embeddings'] = batch['text']
            else:
                batch_text = {}
                for key in batch['text'].keys():
                    batch_text[key] = batch['text'][key].to(self.device)
                output['text_embeddings'] = model.text_encoder(batch_text)
            
            embeds['audio_embeddings'], embeds['text_embeddings'] = l2normalize(output['audio_embeddings']), l2normalize(output['text_embeddings'])

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
    def get_final_metrics(self):
        set_seed(56)
        geoclap, hparams  = self.get_geoclap()
        print(hparams)
        test_dataloader = self.get_dataloader(hparams)
        sat_embeddings = []
        audio_embeddings = []
        for i,batch in tqdm(enumerate(test_dataloader)):
            print("batch no:",str(i))
            embeds = self.get_embeddings(batch=batch,model=geoclap,hparams=hparams)
            sat_embeddings.append(embeds['sat_embeddings'])
            audio_embeddings.append(embeds['audio_embeddings'])
        sat_embeddings = torch.cat(sat_embeddings,axis=0).to(self.device)
        audio_embeddings = torch.cat(audio_embeddings,axis=0).to(self.device)
        print(sat_embeddings.shape, audio_embeddings.shape)
        results_i2s = get_retrevial_metrics(modality1_emb=sat_embeddings, modality2_emb=audio_embeddings, normalized=False)
        results_s2i = get_retrevial_metrics(modality1_emb=audio_embeddings, modality2_emb=sat_embeddings, normalized=False)
        return results_i2s, results_s2i


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = ArgumentParser(description='', formatter_class=RawTextHelpFormatter)
    parser.add_argument('--ckpt_path', type=str)
    args = parser.parse_args()
    #params
    set_seed(56)
    #configure evaluation
    evaluation = Evaluate(validation='test', ckpt_path=args.ckpt_path,device=device)
    results_i2s, results_s2i = evaluation.get_final_metrics()
    print("IMAGE TO SOUND RETREVIAL RESULTS:",results_i2s)
    print("SOUND TO IMAGE RETREVIAL RESULTS:",results_s2i)