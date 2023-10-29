Implementation of `GeoCLAP` as described in our BMVC 2023 paper titled **"Learning Tri-modal Embeddings for Zero-Shot Soundscape Mapping"**. \
[arxiv](https://arxiv.org/abs/2309.10667)

For reproducibility, we have provided the required metadata of the dataset and it's train/val/test split. We also provide the best checkpoints of `GeoCLAP` trained on sentinel2 as well as high resolution GoogleEarth imagery provided in SoundingEarth dataset. These files can be found in [this google drive folder](https://drive.google.com/drive/folders/1Qgh9TNuZ3VZjf6Y6ffMcX5WXL6AHzerP?usp=share_link).

1. Clone this repo
    ```
    git clone git@github.com:mvrl/geoclap.git
    cd geoclap
    ```
2. Setting up enviornment
    ```
    conda env create --file environment.yml
    conda activate geoclap
    ```
    Note: Despite having all the packages we need, for some reasons (yet to be diagnosed!) as discussed in [this issue](https://github.com/NVIDIA/TensorRT/issues/1747) one might get following error while running experiments `OSError: libcudnn.so.8: cannot open shared object file: No such file or directory`. The current solution is to reinstall pytorch as follows:
    ```
    conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
    ```

    Also, you might run into following error: `AttributeError: partially initialized module 'charset_normalizer' has no attribute 'md__mypyc' (most likely due to a circular import)` as dicussed in [this issue](https://github.com/microsoft/TaskMatrix/issues/242). To fix this:
    ```
    pip install --force-reinstall charset-normalizer==3.1.0
    ```
    Note: Instead of `conda` it could be easier to pull docker image `ksubash/geoclap:latest` for the project we provide using following steps:

    ```
    docker pull ksubash/geoclap:latest
    docker run -v $HOME:$HOME --gpus all --shm-size=64gb -it ksubash/geoclap
    source /opt/conda/bin/activate /opt/conda/envs/geoclap
    ```

3. Please refer to `./data_prep/README.md` for details on SoundingEarth and instructions on how to download Sentinel2 imagery. Some scipts for basic pre-processing steps required for experiments related to `GeoCLAP` are also provided there.

4. Check both `config.py` and `./data_prep/config.py` to setup relevant paths by manually creating relevant directories. Copy the pre-trained checkpoint of `SATMAE` named as `finetune-vit-base-e7.pth` provided in [this google drive folder](https://drive.google.com/drive/folders/1Qgh9TNuZ3VZjf6Y6ffMcX5WXL6AHzerP?usp=share_link) to the location pointed by `cfg.pretrained_models_path/SATMAE`. Similarly, copy all data related `.csv` files (`final_metadata_with_captions.csv`,`train_df.csv`,`validate_csv`) to the location pointed by `cfg.DataRoot`.

5. Now assuming that the data preperation is complete following steps 3 and 4, we are now ready to run experiments related to GeoCLAP. Change directory by one step in hierarchy so that `geoclap` can be run as a python module.
    ```
    cd ../
    ```
5. [Optional] It is advisable to pre-compute and save CLAP embeddings for audio and text so that while running experiments involving frozen CLAP encoders, we can fit larger batch size in memory and overall training is faster as well. To pre-compute and save CLAP embeddings run:
    ```
    python -m geoclap.miscs.clap_embeddings
    ```
    
    Note: We use  [wandb](https://wandb.ai/site) for logging our experiments. Therefore before launching experiments make sure you have `wandb` correctly setup. 
6. We can launch the GeoCLAP training as follows:
    ```
   python -m geoclap.train --data_type sat_audio_text \
                             --sat_type sentinel \
                             --text_type with_address \
                             --run_name sentinel_sat_audio_text \
                             --wandb_mode online \
                             --mode train \
                             --train_batch_size 128 \
                             --max_epochs 30 \
                             --freeze_audio_model False \
                             --saved_audio_embeds False \
                             --freeze_text_model False \
                             --saved_text_embeds False
    ```
    Note : Similarly, for all other experiments tabulated in the paper, refer to the document `experiments.txt`. 
7. Once the training is complete and we have decided on the appropriate checkpoint of the model, we can evaluate the cross-modal retrevial performance of the model using:
    ```
    python -m geoclap.evaluate --ckpt_path "path-to-your-geoclap-checkpoint"
    ```
8. Using the best checkpoint of GeoCLAP, audio embeddings for the test set can be pre-computed and saved as a single tensor: `GeoCLAP_gallery_audio_embeds.pt`. This will be used for sat-image to audio retrevial based demonstration.
    ```
    python -m geoclap.miscs.geoclap_audio_embeddings --ckpt_path "path-to-your-geoclap-checkpoint" 
    ```
9. Similarly, using the best checkpoint of GeoCLAP, sat embeddings for the images in the region of interest can be pre-computed using:
    ```
    python -m geoclap.miscs.geoclap_sat_embeddings --ckpt_path "path-to-your-geoclap-checkpoint" \
                                                   --region_file "path-to-your-region-csv" \
                                                   --sat_data_path "path-to-sat-images-for-region" \
                                                   --save_embeds_path "path-to-save-sat-embeds"
    ```
    Note: `geoclap.miscs.geoclap_sat_embeddings` assumes that the `region_file csv` consists of at least three fields:`key`, `latitude`, `longitude`. Also, it assumes that the satellite images of the dense grid over the region of interest are already downloaded following instructions in `./data_prep/CVGlobal/README.md`. The `keys` in `region_file csv` should match the corresponding filenames for images saved in directory pointed by `sat_data_path`.

10. Accordingly, as demonstrated in the main paper, for a region of interest (a `.csv` file containing (latitude,longitude) for all locations in a grid covering the region), we can compute cosine similarity of text and/or audio query with all satellite imagery over the region. Note for audio query, the script randomly selects audio from ESC50 dataset for a predefined set of classes in `cfg.heatmap_classes`.
    ```
    python -m geoclap.miscs.compute_similarity --ckpt_path "path-to-your-geoclap-checkpoint" \
                                               --region_file_path "path-to-region_file.csv" \
                                               --sat_data_path "path-to-satellite-images-for-the-region" \
                                               --text_query "animal farm;chirping birds;car horn" \
                                               --query_type "audio_text"
    ```

11. As demonstrated in supplementary materials of the paper, we provide a demo script to use the pre-trained GeoCLAP model to query with multiple textual prompts as well as to retrieve top audio from our test-set gallery (using the precomputed test-set audio embeddings from step 8.)
    ```
    python -m geoclap.miscs.demo --ckpt_path "path-to-checkpoint-of-the-best-model-trained-on-sat-imagery" \
                                 --region_file_path "path-to-region_file.csv" \
                                 --sat_data_path  "path-to-sat-images-for-the-region" \
                                 --query_type "audio_text" \
                                 --text_query "church bells;flowing river;animal farm;chirping birds;car horn;manufacturing factory" \
                                 --output_filename "demofile"
    ```
