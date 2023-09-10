#!/bin/bash
# This requires GNU parallel and the `internetarchive` library to be installed on your system.
# You likely have the first one if you're running linux.
# For the second one, check https://archive.org/services/docs/api/internetarchive/installation.html
# Or simply run `pip install internetarchive`

cd /storage1/fs1/jacobsn/Active/user_k.subash/data/aporee
mkdir -p ./raw_audio
cd ./raw_audio
metadata_path="/storage1/fs1/jacobsn/Active/user_k.subash/data/aporee/metadata.csv"
tail -n +2 $metadata_path | cut -d',' -f 1 | parallel -j8 --joblog ../audio_download.log 'ia download {} --glob="*.mp3"'
