from yacs.config import CfgNode as CN
import os
cfg = CN()

cfg.DataRoot = '/storage1/fs1/jacobsn/Active/user_k.subash/data_archive/aporee'
cfg.pretrained_models_path = '/storage1/fs1/jacobsn/Active/user_k.subash/checkpoints/'

#paths
cfg.data_path = cfg.DataRoot
cfg.sat_audio_path = os.path.join(cfg.data_path,'raw_audio')
cfg.sat_audio_tensors_path = os.path.join(cfg.data_path,'raw_audio_tensors')
cfg.audio_duration = 10
cfg.embeddings_path = os.path.join(cfg.data_path,'embeddings')
cfg.sat_image_path = os.path.join(cfg.data_path,'images')
cfg.detailed_metadata_path = os.path.join(cfg.data_path,'final_metadata_with_captions.csv')
cfg.train_csv = os.path.join(cfg.data_path,'train_df.csv')
cfg.validate_csv = os.path.join(cfg.data_path,'validate_df.csv')
cfg.test_csv = os.path.join(cfg.data_path,'test_df.csv')

######################################## training hparams ##################################################
cfg.log_dir = '/storage1/fs1/jacobsn/Active/user_k.subash/projects/geoclap/logs'


####################################ESC10 subset############################################################
cfg.esc_data_path = "/storage1/fs1/jacobsn/Active/user_k.subash/data/ESC-50-master/"
cfg.esc_audio_tensors_path = os.path.join("/storage1/fs1/jacobsn/Active/user_k.subash/data/ESC-50-master","raw_audio_tensors")
cfg.esc10_classes = ["rain","sea_waves","chirping_birds","wind","siren","car_horn","engine","train","airplane","fireworks"]
cfg.esc_metapath = "/storage1/fs1/jacobsn/Active/user_k.subash/data/ESC-50-master/meta/esc50.csv"

cfg.heatmap_classes = ["car_horn","chirping_birds","fireworks","sea_waves"]
cfg.custom_heatmap_classes = ["car_horn","chirping_birds","animal_farm"]

########################### demo paths ###################################################################
cfg.best_model_ckpt = "/home/k.subash/GeoCLAP/logs/GeoClAP/w0k7jp9t/checkpoints/epoch=26-step=7506-val_loss=2.339.ckpt"
cfg.demo_path = "/home/k.subash/GeoCLAP/logs/demo" 
cfg.region_files_path = "/storage1/fs1/jacobsn/Active/proj_smart/GeoCLAP/region_files"
cfg.netherlands_images = "/storage1/fs1/jacobsn/Active/proj_smart/GeoCLAP/images_sentinel"
cfg.GeoCLAP_gallery_audio_embeds = '/storage1/fs1/jacobsn/Active/user_k.subash/data/aporee/embeddings/GeoCLAP_gallery_audio_embeds.pt'