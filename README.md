# GeoCLAP: Learning Tri-modal Embeddings for Zero-Shot Soundscape Mapping

PyTorch implementation of **GeoCLAP** from the BMVC 2023 paper ["Learning Tri-modal Embeddings for Zero-Shot Soundscape Mapping"](https://arxiv.org/abs/2309.10667).

GeoCLAP learns joint embeddings across satellite imagery, audio, and text to enable zero-shot soundscape mapping — predicting the acoustic environment of any location on Earth from overhead imagery alone.

For reproducibility, we provide the required dataset metadata and train/val/test splits. We also provide the best checkpoints of `GeoCLAP` trained on Sentinel-2 as well as high-resolution Google Earth imagery from the SoundingEarth dataset. These files can be found in [this Google Drive folder](https://drive.google.com/drive/folders/1Qgh9TNuZ3VZjf6Y6ffMcX5WXL6AHzerP?usp=share_link).

## 🔧 Installation

### Setup

1. Clone this repo:
    ```bash
    git clone git@github.com:mvrl/geoclap.git
    cd geoclap
    ```

2. Create and activate the conda environment:
    ```bash
    conda env create --file environment.yml
    conda activate geoclap
    ```

    > **Note:** If you encounter `OSError: libcudnn.so.8: cannot open shared object file: No such file or directory` (discussed in [this issue](https://github.com/NVIDIA/TensorRT/issues/1747)), reinstall PyTorch:
    > ```bash
    > conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
    > ```

    > **Note:** If you encounter `AttributeError: partially initialized module 'charset_normalizer' has no attribute 'md__mypyc'` (discussed in [this issue](https://github.com/microsoft/TaskMatrix/issues/242)), run:
    > ```bash
    > pip install --force-reinstall charset-normalizer==3.1.0
    > ```

### Docker (Alternative)

Instead of conda, you can pull the pre-built Docker image:

```bash
docker pull ksubash/geoclap:latest
docker run -v $HOME:$HOME --gpus all --shm-size=64gb -it ksubash/geoclap
source /opt/conda/bin/activate /opt/conda/envs/geoclap
```

## 📁 Data Preparation

1. Refer to `./data_prep/README.md` for details on the SoundingEarth dataset and instructions on how to download Sentinel-2 imagery. Basic pre-processing scripts for GeoCLAP experiments are also provided there.

2. Check both `geoclap/config.py` and `./data_prep/config.py` to set up relevant paths by manually creating the required directories.
   - Copy the pre-trained `SATMAE` checkpoint (`finetune-vit-base-e7.pth`) from [this Google Drive folder](https://drive.google.com/drive/folders/1Qgh9TNuZ3VZjf6Y6ffMcX5WXL6AHzerP?usp=share_link) to `cfg.pretrained_models_path/SATMAE`.
   - Copy all data-related `.csv` files (`final_metadata_with_captions.csv`, `train_df.csv`, `validate_csv`) to the location pointed to by `cfg.DataRoot`.

## 🏃 Training

> **Note:** We use [wandb](https://wandb.ai/site) for experiment logging. Make sure `wandb` is correctly set up before launching experiments.

### Pre-compute CLAP Embeddings (Optional)

Pre-computing and saving CLAP embeddings for audio and text allows for larger batch sizes and faster training when using frozen CLAP encoders:

```bash
python -m geoclap.miscs.clap_embeddings
```

### Launch Training

```bash
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

> **Note:** For all other experiments tabulated in the paper, refer to `experiments.txt`.

## 📈 Evaluation

Once training is complete, evaluate the cross-modal retrieval performance of the model:

```bash
python -m geoclap.evaluate --ckpt_path "path-to-your-geoclap-checkpoint"
```

## 🗺️ Soundscape Mapping

### Pre-compute Audio Embeddings

Using the best GeoCLAP checkpoint, pre-compute and save audio embeddings for the test set as `GeoCLAP_gallery_audio_embeds.pt` (used for satellite-image-to-audio retrieval demonstrations):

```bash
python -m geoclap.miscs.geoclap_audio_embeddings --ckpt_path "path-to-your-geoclap-checkpoint"
```

### Pre-compute Satellite Embeddings

Pre-compute satellite embeddings for images in a region of interest:

```bash
python -m geoclap.miscs.geoclap_sat_embeddings --ckpt_path "path-to-your-geoclap-checkpoint" \
                                               --region_file "path-to-your-region-csv" \
                                               --sat_data_path "path-to-sat-images-for-region" \
                                               --save_embeds_path "path-to-save-sat-embeds"
```

> **Note:** The `region_file` CSV must contain at least three fields: `key`, `latitude`, `longitude`. Satellite images for the dense grid over the region must be pre-downloaded following instructions in `./data_prep/CVGlobal/README.md`. The `key` values must match the corresponding image filenames in `sat_data_path`.

### Compute Similarity Heatmaps

For a region of interest, compute cosine similarity between text and/or audio queries and all satellite imagery over the region. For audio queries, the script randomly selects audio from the ESC50 dataset for a predefined set of classes in `cfg.heatmap_classes`:

```bash
python -m geoclap.miscs.compute_similarity --ckpt_path "path-to-your-geoclap-checkpoint" \
                                           --region_file_path "path-to-region_file.csv" \
                                           --sat_data_path "path-to-satellite-images-for-the-region" \
                                           --text_query "animal farm;chirping birds;car horn" \
                                           --query_type "audio_text"
```

### Demo

Query with multiple text prompts and retrieve top audio from the test-set gallery (using pre-computed audio embeddings from above):

```bash
python -m geoclap.miscs.demo --ckpt_path "path-to-checkpoint-of-the-best-model-trained-on-sat-imagery" \
                             --region_file_path "path-to-region_file.csv" \
                             --sat_data_path "path-to-sat-images-for-the-region" \
                             --query_type "audio_text" \
                             --text_query "church bells;flowing river;animal farm;chirping birds;car horn;manufacturing factory" \
                             --output_filename "demofile"
```

## 📝 Citation

If you find this code useful for your research, please cite:

```bibtex
@inproceedings{khanal2023soundscape,
  title = {Learning Tri-modal Embeddings for Zero-Shot Soundscape Mapping},
  author = {Khanal, Subash and Sastry, Srikumar and Dhakal, Aayush and Jacobs, Nathan},
  year = {2023},
  month = nov,
  booktitle = {British Machine Vision Conference (BMVC)},
}
```

Follow more works from our lab: [The Multimodal Vision Research Laboratory (MVRL)](https://mvrl.cse.wustl.edu)
