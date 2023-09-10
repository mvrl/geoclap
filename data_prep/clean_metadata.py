# This script attempts to perform post-processing of 'description' column 
# of metadata of the SoundingEarth dataset.
import re
import numpy as np
import pandas as pd
from geopy.geocoders import Nominatim
from geopy.point import Point
from cleantext import clean
import os
# from wordsegment import load, segment
# import nltk
from tqdm import tqdm
# nltk.download('words')
# words = set(nltk.corpus.words.words())
import sys
from config import cfg

geolocator = Nominatim(user_agent="openmapquest")
def reverse_geocoding(lat, lon):
    try:
        location = geolocator.reverse(Point(lat, lon),language='en')
        address = location.address
        return address
    except:
        address = None
        return address

# def splitwords(word):
#     load()
#     return ' '.join(segment(word))

def clean_description(description):
    sent = re.sub(r'(<br\s*/>)',' ',description)
    output = clean(sent,
        fix_unicode=True,               # fix various unicode errors
        to_ascii=True,                  # transliterate to closest ASCII representation
        lower=True,                     # lowercase text
        no_line_breaks= True,           # fully strip line breaks as opposed to only normalizing them
        no_urls=True,                   # replace all URLs with a special token
        no_emails=True,                 # replace all email addresses with a special token
        no_phone_numbers=True,          # replace all phone numbers with a special token
        no_numbers=False,               # replace all numbers with a special token
        no_digits=False,                # replace all digits with a special token
        no_currency_symbols=False,      # replace all currency symbols with a special token
        no_punct= True,                 # remove punctuations
        replace_with_punct="",          # instead of removing punctuations you may replace them
        replace_with_url="<URL>",
        replace_with_email="<EMAIL>",
        replace_with_phone_number="<PHONE>",
        replace_with_number="<NUMBER>",
        replace_with_digit="0",
        replace_with_currency_symbol="<CUR>",
        lang="en"                       # set to 'de' for German special handling
                )

    output = re.sub(r'\s+',' ',output)

    return output


def get_caption(i, lat, lon, title, description):
    if pd.notna(description):
        description = description
    else:
        description = title
    
    address = reverse_geocoding(lat=lat, lon=lon)
    if address != None:
        caption = clean_description(description + '. The location of the sound is: '+address+'.')
    else:
        caption = clean_description(description + '.')
    if i%100 == 0:
        print(str(i)+" done!")
    return caption


def get_detailed_metadata(data_path):
    meta_df = pd.read_csv(os.path.join(cfg.data_path,'metadata.csv'))
    print("Original data",len(meta_df))
    corrupt_ids = list(pd.read_csv(os.path.join(cfg.data_path,"corrupt_ids_final.csv"))['long_key']) #Ignore some IDs whose mp3 is found to be corrupt
    meta_df = meta_df[~meta_df['long_key'].isin(corrupt_ids)]
    print("data removing corrupt mp3",len(meta_df))

    #keep only the data with audio fs >= 16000
    meta_df = meta_df[meta_df['mp3samplerate']>=16000]
    print("data removing mp3 sampled by sr less than 16k",len(meta_df))
    audio_short_ids = list(meta_df.key)

    image_ids = os.listdir(cfg.sat_image_path)
    image_short_ids = [i.split('.jpg')[0] for i in image_ids]
    print("Total count of sat image samples in original dataset") 
    print(len(image_short_ids))
    meta_df.to_csv(os.path.join(cfg.data_path,"final_metadata.csv"))
    metadata = meta_df.fillna(np.nan)
    keys = list(metadata.key)
    lats = list(metadata.latitude)
    longs = list(metadata.longitude)
    titles = list(metadata.title)
    descriptions = list(metadata.description)

    captions = [get_caption(i, lats[i], longs[i], titles[i], descriptions[i]) for i in range(len(metadata))]
    metadata['caption'] = captions
    address = ["The location of the sound is" + caption.split("location of the sound is")[1] for caption in captions]
    metadata['address'] = address
    metadata.to_csv(os.path.join(data_path,'final_metadata_with_captions.csv'))

    print("description of metadata cleaned")

    return metadata

get_detailed_metadata(cfg.data_path)
