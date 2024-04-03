import sys
from itertools import chain
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import torch.nn as nn
import torch
import os
import random
from argparse import ArgumentParser
from .config import cfg
from .models import SATMAE,AudioCLAP
from .dataloader import Dataset_soundscape
from .metrics import get_retrevial_metrics
import numpy as np
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

os.environ["WANDB__SERVICE_WAIT"] = "300"

def l2normalize(batch_embeddings):
    return batch_embeddings/batch_embeddings.norm(p=2,dim=-1, keepdim=True)

def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))

#computer cross entropy for the similarity matrix both rowwise and columnwise
def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    modality1_loss = contrastive_loss(similarity)
    modality2_loss = contrastive_loss(similarity.t())
    return (modality1_loss + modality2_loss) / 2.0

def get_loss(modality1_embeddings, modality2_embeddings,logit_scale):
    #similarity between moadality1 and modality2
    logits_per_modality1 = torch.matmul(modality1_embeddings,modality2_embeddings.t())*logit_scale
    #compute cross_entropy loss between the cross-modal similarities and hard gt
    loss_mod1mod2 = clip_loss(logits_per_modality1)

    return loss_mod1mod2, logits_per_modality1


class GeoCLAP(pl.LightningModule):
    def __init__(self, hparams):

        #save paramaters
        super().__init__()
        self.save_hyperparameters(hparams)
         #set path attributes
        self.train_path = cfg.train_csv
        self.vali_path = cfg.validate_csv
        self.test_path = cfg.test_csv
        self.valid_end_list =[]
        #Data modality: Satellite Image
        if self.hparams.sat_encoder == 'SatMAE':
            self.sat_encoder = SATMAE.SatMAE(pretrained_models_path=cfg.pretrained_models_path,device=self.device,fc_dim = self.hparams.fc_dim,metadata_type=self.hparams.metadata_type).to(self.device)     
        else:
            raise NotImplementedError("Only implemented Satellite image encoder is SATMAE")

        #Data modality: Audio and/or text
        if self.hparams.audio_encoder == 'AudioCLAP': #accepts either audio or text
            if 'audio' in self.hparams.data_type:
                if not self.hparams.saved_audio_embeds: # if frozen embeddings are NOT already saved 
                    self.audio_encoder = AudioCLAP.AudioCLAP_audiomodel(freeze=self.hparams.freeze_audio_model)
            if 'text' in self.hparams.data_type:
                if not self.hparams.saved_text_embeds: # if frozen embeddings are NOT already saved 
                    self.text_encoder = AudioCLAP.AudioCLAP_textmodel(freeze=self.hparams.freeze_text_model)
        
        if not self.hparams.saved_audio_embeds:
            if 'audio' in self.hparams.data_type and self.hparams.freeze_audio_model:
                self.audio_encoder.eval()
                for params in self.audio_encoder.parameters():
                    params.requires_grad=False

        if not self.hparams.saved_text_embeds:
            if 'text' in self.hparams.data_type and self.hparams.freeze_text_model:
                self.text_encoder.eval()
                for params in self.text_encoder.parameters():
                    params.requires_grad=False
            
        #trainable satellite image encoder to get embeddings for satellite image
        self.sat_encoder.train()

        self.temp_layer = AudioCLAP.temp_layer(self.hparams)
                
        self.temp_clip = self.hparams.temp_clip

    def get_embeds(self,batch):
        embeds = {'sat_embeddings':None, 'audio_embeddings':None, 'text_embeddings':None}
        if self.hparams.metadata_type == 'lat_long':
            embeds['sat_embeddings']  = l2normalize(self.sat_encoder(batch['sat'].to(self.device),batch['lat_long'].to(self.device)))
        else:
            embeds['sat_embeddings']  = l2normalize(self.sat_encoder(batch['sat'].to(self.device)))

        if self.hparams.data_type == 'sat_audio':
            output = {}
            if self.hparams.saved_audio_embeds:
                output['audio_embeddings'] = batch['audio'].to(self.device)
            else:
                batch_audio = {}
                for key in batch['audio'].keys():
                    batch_audio[key] = batch['audio'][key].to(self.device)

                output['audio_embeddings'] = self.audio_encoder(batch_audio)
            
            embeds['audio_embeddings'] = l2normalize(output['audio_embeddings'])

        if self.hparams.data_type == 'sat_audio_text':
            output = {}
            if self.hparams.saved_audio_embeds:
                output['audio_embeddings'] = batch['audio'].to(self.device)
            else:
                batch_audio = {}
                for key in batch['audio'].keys():
                    batch_audio[key] = batch['audio'][key].to(self.device) 
                output['audio_embeddings'] = self.audio_encoder(batch_audio)
            
            if self.hparams.saved_text_embeds:
                output['text_embeddings'] = batch['text'].to(self.device)
            else:
                batch_text = {}
                for key in batch['text'].keys():
                    batch_text[key] = batch['text'][key].to(self.device)
                output['text_embeddings'] = self.text_encoder(batch_text)
            
            embeds['audio_embeddings'], embeds['text_embeddings'] = l2normalize(output['audio_embeddings']), l2normalize(output['text_embeddings'])

        return embeds
    def forward(self, batch):
        embeds = self.get_embeds(batch)
        return embeds
    
    #clamp the temperature parameter
    def on_before_zero_grad(self, *args, **kwargs):
        self.temp_layer.logit_scale_ia.data = torch.clamp(self.temp_layer.logit_scale_ia.data, min=1.0, max=np.log(self.hparams.temp_clip))
        if self.hparams.data_type == 'sat_audio_text':
            self.temp_layer.logit_scale_it.data = torch.clamp(self.temp_layer.logit_scale_it.data, min=1.0, max=np.log(self.hparams.temp_clip))
            if not self.hparams.freeze_text_model:
                self.temp_layer.logit_scale_at.data = torch.clamp(self.temp_layer.logit_scale_at.data, min=1.0, max=np.log(self.hparams.temp_clip))
    
    def shared_step(self, batch):
        embeds = self(batch)

        audio_embeddings = embeds['audio_embeddings']
        sat_embeddings = embeds['sat_embeddings']
        text_embeddings = embeds['text_embeddings']

        #Calculate loss
        logit_scale_ia = self.temp_layer.logit_scale_ia.exp()
        loss_ia, logits_per_satImage_audio = get_loss(modality1_embeddings = sat_embeddings, 
                                                    modality2_embeddings=audio_embeddings,
                                                    logit_scale=logit_scale_ia)
            
        if self.hparams.data_type == 'sat_audio_text':
            logit_scale_it = self.temp_layer.logit_scale_it.exp()
            loss_SatText, logits_per_satImage_text = get_loss(modality1_embeddings = sat_embeddings, 
                                                            modality2_embeddings=text_embeddings,
                                                            logit_scale=logit_scale_it)
            if not self.hparams.freeze_text_model:
                logit_scale_at = self.temp_layer.logit_scale_at.exp() 
                loss_AudioText, logits_per_Audio_text = get_loss(modality1_embeddings = audio_embeddings, 
                                                            modality2_embeddings=text_embeddings,
                                                            logit_scale=logit_scale_at)     
                loss = (loss_ia + loss_SatText + loss_AudioText)/3
        
                return {'loss':loss,
                        'loss_ia':loss_ia,
                        'loss_it':loss_SatText,
                        'loss_at':loss_AudioText,
                        'logits_per_satImage_audio': logits_per_satImage_audio,
                        'normalized_audio_embeddings': audio_embeddings,
                        'normalized_satellite_embeddings': sat_embeddings
                        }
            else:
                loss = (1-self.hparams.text_loss_weight)*loss_ia + self.hparams.text_loss_weight*loss_SatText
                return {'loss':loss,
                        'loss_ia':loss_ia,
                        'loss_it':loss_SatText,
                        'logits_per_satImage_audio': logits_per_satImage_audio,
                        'normalized_audio_embeddings': audio_embeddings,
                        'normalized_satellite_embeddings': sat_embeddings
                        }
            
        else:
            return {'loss':loss_ia,
                'logits_per_satImage_audio': logits_per_satImage_audio,
                'normalized_audio_embeddings': audio_embeddings,
                'normalized_satellite_embeddings': sat_embeddings
                }
        
    
    def training_step(self, batch, batch_idx):
        outputs = self.shared_step(batch)
        if self.hparams.data_type == 'sat_audio':
            self.log('loss', outputs['loss'], sync_dist=True, batch_size=self.hparams.train_batch_size)
            self.log('logit_scale_ia',self.temp_layer.logit_scale_ia.data,sync_dist=True, batch_size=self.hparams.train_batch_size)
        if self.hparams.data_type == 'sat_audio_text':
            self.log('loss', outputs['loss'], sync_dist=True, batch_size=self.hparams.train_batch_size)
            self.log('loss_ia', outputs['loss_ia'], sync_dist=True, batch_size=self.hparams.train_batch_size)
            self.log('loss_it', outputs['loss_it'], sync_dist=True, batch_size=self.hparams.train_batch_size)
            self.log('logit_scale_ia',self.temp_layer.logit_scale_ia.data,sync_dist=True, batch_size=self.hparams.train_batch_size)
            self.log('logit_scale_it',self.temp_layer.logit_scale_it.data,sync_dist=True, batch_size=self.hparams.train_batch_size)
            if not self.hparams.freeze_text_model: 
                self.log('loss_at', outputs['loss_at'], sync_dist=True, batch_size=self.hparams.train_batch_size)
                self.log('logit_scale_at',self.temp_layer.logit_scale_at.data,sync_dist=True, batch_size=self.hparams.train_batch_size)
        
        return outputs['loss']
        
    
    def validation_step(self, batch, batch_idx):
        outputs = self.shared_step(batch)
        val_loss = outputs['loss']

        self.log('val_loss', val_loss, sync_dist=True, batch_size=self.hparams.val_batch_size, prog_bar=True)
        final_output = {'val_loss':outputs['loss'],'normalized_satellite_embeddings':outputs['normalized_satellite_embeddings'], 'normalized_audio_embeddings':outputs['normalized_audio_embeddings']}
        self.valid_end_list.append(final_output)
        return final_output


    #compute retrieval metrics for a random batch of validation 
    def on_validation_epoch_end(self):
        # import code;code.interact(local=dict(globals(), **locals()));
        outputs = self.valid_end_list 
        if len(outputs)==0:
            print('Skipping Validatiion Epoch End')
            pass
        else:
            random_batch = np.random.randint(0,len(outputs))
            validation_embeddings = outputs[random_batch]
            normalized_satellite_embeddings = validation_embeddings['normalized_satellite_embeddings']
            normalized_audio_embeddings = validation_embeddings['normalized_audio_embeddings']
            retrieval_results = get_retrevial_metrics(modality1_emb=normalized_satellite_embeddings, modality2_emb=normalized_audio_embeddings, normalized=True,k=10)
            self.log(f'R@10', retrieval_results['R@10'])
            self.log(f'Median Rank', retrieval_results['Median Rank'])
            self.valid_end_list = []
            return retrieval_results
    

    def train_dataloader(self):
        train_csv = cfg.train_csv
        trainloader = torch.utils.data.DataLoader(Dataset_soundscape(
                                                                    data_file=train_csv,
                                                                    is_train = True,
                                                                    sat_input_size= self.hparams.sat_input_size,
                                                                    sat_model= self.hparams.sat_encoder,
                                                                    audio_model= self.hparams.audio_encoder,
                                                                    data_type = self.hparams.data_type,
                                                                    metadata_type= self.hparams.metadata_type,
                                                                    saved_audio_embeds= self.hparams.saved_audio_embeds,
                                                                    saved_text_embeds= self.hparams.saved_text_embeds,
                                                                    sat_type = self.hparams.sat_type,
                                                                    text_type = self.hparams.text_type),
                                        num_workers=self.hparams.num_workers, batch_size=self.hparams.train_batch_size, shuffle=True, drop_last=False,pin_memory=True)
        return trainloader

    def val_dataloader(self):
        validate_csv = cfg.validate_csv
        validloader = torch.utils.data.DataLoader(Dataset_soundscape(
                                                                    data_file=validate_csv,
                                                                    is_train = False,
                                                                    sat_input_size= self.hparams.sat_input_size,
                                                                    sat_model= self.hparams.sat_encoder,
                                                                    audio_model= self.hparams.audio_encoder,
                                                                    data_type = self.hparams.data_type,
                                                                    metadata_type= self.hparams.metadata_type,
                                                                    saved_audio_embeds= self.hparams.saved_audio_embeds,
                                                                    saved_text_embeds= self.hparams.saved_text_embeds,
                                                                    sat_type = self.hparams.sat_type,
                                                                    text_type = self.hparams.text_type),
                                        num_workers=self.hparams.num_workers, batch_size=self.hparams.val_batch_size, shuffle=False, drop_last=False,pin_memory=True)
        return validloader

    def test_dataloader(self):
        test_csv = cfg.test_csv
        testloader = torch.utils.data.DataLoader(Dataset_soundscape(
                                                                    data_file=test_csv,
                                                                    is_train = False,
                                                                    sat_input_size= self.hparams.sat_input_size,
                                                                    sat_model= self.hparams.sat_encoder,
                                                                    audio_model= self.hparams.audio_encoder,
                                                                    data_type = self.hparams.data_type,
                                                                    metadata_type= self.hparams.metadata_type,
                                                                    saved_audio_embeds= self.hparams.saved_audio_embeds,
                                                                    saved_text_embeds= self.hparams.saved_text_embeds,
                                                                    sat_type = self.hparams.sat_type,
                                                                    text_type = self.hparams.text_type),
                                        num_workers=self.hparams.num_workers, batch_size=self.hparams.test_batch_size, shuffle=False, drop_last=False,pin_memory=True)
        return testloader

    def configure_optimizers(self):
        print(f'Initializing Learning rate {self.hparams.learning_rate}')
        if self.hparams.data_type == 'sat_audio':
            if self.hparams.saved_audio_embeds or self.hparams.freeze_audio_model:
                params = chain(self.sat_encoder.parameters(),self.temp_layer.parameters())
            else:
                params = chain(self.sat_encoder.parameters(),self.audio_encoder.parameters(),self.temp_layer.parameters())

        elif self.hparams.data_type == 'sat_audio_text':
            if self.hparams.saved_text_embeds or self.hparams.freeze_text_model:
                params = chain(self.sat_encoder.parameters(),self.temp_layer.parameters())
            else:
                params = chain(self.sat_encoder.parameters(),self.audio_encoder.parameters(),self.text_encoder.parameters(),self.temp_layer.parameters())
        self.optim = torch.optim.AdamW(params=params,
                    lr=self.hparams.learning_rate,
                    weight_decay=0.2,
                    betas=(0.9,0.98),
                    eps=1e-6
                    )
        
        
        self.warm_up_iterations = 2000
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer = self.optim,
            T_0 = self.warm_up_iterations
        )
        return {'optimizer': self.optim,
        'lr_scheduler': {
            'name':'train/lr',
            'scheduler': self.scheduler,
            'interval': 'step',
            'frequency': 1
        }
        }
    
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


def get_args():
    parser = ArgumentParser(description='')
    #training hparams
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--train_batch_size', type=int, default=256)
    parser.add_argument('--val_batch_size', type=int, default=256)
    parser.add_argument('--test_batch_size', type=int, default=256)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--mode', type=str, default='dev')                          #Options: dev, train
    parser.add_argument('--freeze_audio_model', type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument('--saved_audio_embeds',type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument('--freeze_text_model',type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument('--saved_text_embeds',type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument('--data_type', type=str, default='sat_audio')
    parser.add_argument('--sat_type', type=str, default='SoundingEarth')            #Options: [SoundingEarth, sentinel]
    parser.add_argument('--text_type', type=str, default='with_address')            #Options: [with_address, without_address, only_address]
    parser.add_argument('--metadata_type', type=str, default='none')                #Options: none, lat_long
    parser.add_argument('--text_loss_weight', type=float, default=0.5)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--project_name', type=str, default='GeoCLAP')
    parser.add_argument('--run_name', type=str, default='debug')
    parser.add_argument('--wandb_mode', type=str, default='disabled')
    parser.add_argument('--strategy', type=str, default='ddp_find_unused_parameters_false')
    parser.add_argument('--accelerator',type=str, default='gpu')
    parser.add_argument('--devices', type=int, default=1)
    parser.add_argument('--val_check_interval', type=int, default=1.0)

    # encoder types:
    parser.add_argument('--sat_encoder',type=str,default='SatMAE') 
    parser.add_argument('--audio_encoder',type=str,default='AudioCLAP')


    parser.add_argument('--fc_dim', type=int, default = 512)
    parser.add_argument('--sat_input_size', type=int, default= 224)

    #cilp specific hparams
    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--temp_clip',type=int, default =100)

    #logging hparams
    parser.add_argument('--ckpt_path',type=str, default ='none')
    parser.add_argument('--ckpt_mode',type=str, default ='hard')


    args = parser.parse_args()

    return args

if __name__ == '__main__':
    set_seed(56)
    args = get_args()
    #set learning rate logger
    print('Starting Training')
    print(args)
    #initliaze model
    geoclap_model = GeoCLAP(args)
    #initialize checkpoints and loggers
    lr_logger = LearningRateMonitor(logging_interval='step')
    wb_logger = WandbLogger(save_dir=cfg.log_dir,project=args.project_name, name=args.run_name, mode=args.wandb_mode)
    ckpt_monitors = ((
            ModelCheckpoint(monitor='val_loss', filename='{epoch}-{step}-{val_loss:.3f}', save_top_k = 5, every_n_epochs = 1,save_last=True,save_on_train_epoch_end=True)
        ))

    if args.mode == 'dev': 
        print('Development Test Run')
        trainer = pl.Trainer(precision=16,fast_dev_run=6, max_epochs=4, logger=wb_logger, strategy=args.strategy, num_sanity_val_steps=4,
        accelerator=args.accelerator, devices=args.devices, callbacks=[ckpt_monitors, lr_logger])
    elif args.mode == 'train':
        print('Training Run')
        trainer = pl.Trainer(precision=16, max_epochs=args.max_epochs, logger=wb_logger, strategy=args.strategy, num_sanity_val_steps=0, 
        accelerator=args.accelerator, devices=args.devices, callbacks=[ckpt_monitors, lr_logger], 
        val_check_interval=args.val_check_interval, log_every_n_steps=25)
    else:
        raise ValueError('Invalid value for mode')
    
    if args.ckpt_path.lower()=='none'.lower():
        trainer.fit(geoclap_model)
    else:
        if args.ckpt_mode.lower()=='hard':
            print('Hard Checkpoint Reload')
            trainer.fit(geoclap_model, ckpt_path=args.ckpt_path)
        elif args.ckpt_mode.lower()=='soft':
            print('Soft Checkpoint Reload')
            checkpoint = torch.load(args.ckpt_path)
            geoclap_model.load_state_dict(checkpoint['state_dict'])
            trainer.fit(geoclap_model)
