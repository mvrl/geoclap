# Download sentinel-cloudless for locations provided in a csv file: "source_balanced_geotagged_sounds.csv"
from argparse import ArgumentParser
import urllib
import pandas as pd
import os
import numpy as np
import pymap3d as pm
import urllib.request
from tqdm import tqdm

LIMIT = 100000 #Download limit per job

def bounding_box_from_circle(lat_center, lon_center, radius = 1000,
    disable_latitude_compensation=False):
  '''
  radius is in meters determined at the equator

  warning: doesn't handle the poles or the 180th meridian very well, it might loop give a bad bounding box
   should probably define a check to make sure the radius isn't too big
  '''
  
  thetas = np.linspace(0,2*np.pi, 5)
  x, y = radius*np.cos(thetas), radius*np.sin(thetas)


  if not disable_latitude_compensation:
    # use tangent plane boxes, defined in meters at location
    lat, lon, alt = pm.enu2geodetic(x, y, 0, lat_center, lon_center, 0)
  else:
    # use lat-lon boxes, defined in meters at equator
    lat, lon, alt = pm.enu2geodetic(x, y, 0, 0, 0, 0)
    lat = lat + lat_center
    lon = lon + lon_center

  b,t = lat[3], lat[1]
  l,r = lon[2], lon[0]

  return l,b,r,t

def download(url, out_file, redownload=False):
  '''
  Returns False if image didn't need to be downloaded.
  '''
  # print(f"checking: {out_file}")
  if not os.path.isfile(out_file) or redownload:
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    try:
      urllib.request.urlretrieve(url, out_file)
      # print(f"success: {url}")
    except KeyboardInterrupt :
      print('Shutting down.')
      return False
    except :
      print(f"failed: {url}")
    return True
  else:
    print(f"already exists")
    return True 
  
class Downloader():
  def __init__(self,save_path,layer='sentinel', radius=512,
      wms_uri='http://127.0.0.1:8080/wms', height=256, width=256,
      im_format="jpeg",disable_latitude_compensation=False):

    # radius: area of patch should be close to (2*radius)^2 meters 

    self.layer = layer
    self.radius = radius
    self.height = height
    self.width = width
    self.wms_uri = wms_uri
    self.im_format = im_format
    self.disable_latitude_compensation = disable_latitude_compensation
    self.save_path = save_path


  def process(self, rows):
    total_keys = 0
    total_downloads = 0
    total_keys_indb = 0

    for row in tqdm(rows):
      source, key, lat, lon = row
      total_keys += 1
      
      dir_path = os.path.join(self.save_path,source,"images",self.layer)
      if not os.path.exists(dir_path):
        os.makedirs(dir_path)

      filepath = os.path.join(dir_path,str(key)+"."+self.im_format)
      if not os.path.exists(filepath):
        l,b,r,t = bounding_box_from_circle(float(lat),float(lon),self.radius,
            disable_latitude_compensation=self.disable_latitude_compensation)

        image_url = f"{self.wms_uri}?service=WMS&version=1.1.1&request=GetMap&layers={self.layer}&styles=&width={self.width}&height={self.height}&srs=EPSG:4326&bbox={l},{b},{r},{t}&format=image/{self.im_format}"
        try:
          download(image_url,filepath,redownload=True)
          total_downloads += 1

        except urllib.error.HTTPError as e:
              # Return code error (e.g. 404, 501, ...)
              # ...
              print('HTTPError: {}'.format(e.code))
        except urllib.error.URLError as e:
              # Not an HTTP-specific error (e.g. connection refused)
              # ...
              print('URLError: {}'.format(e.reason))
        #file = open(filepath,'w')

      else:
        total_keys_indb += 1

    print(f"Total Keys (some may be duplicates): {total_keys}")
    print(f"Total Keys (already in database): {total_keys_indb}")
    print(f"Total Downloads (attempted): {total_downloads}")

def cli_main():

  parser = ArgumentParser()
  parser.add_argument('--save_path', type=str,default="/storage1/fs1/jacobsn/Active/user_k.subash/data/")
  parser.add_argument('--meta_path', default="/storage1/fs1/jacobsn/Active/user_k.subash/data/source_balanced_geotagged_sounds.csv",type=str)
  parser.add_argument('--layer', default='sentinel', type=str)
  parser.add_argument('--radius', default=512, type=int)
  parser.add_argument('--height', default=256, type=int)
  parser.add_argument('--width', default=256, type=int)
  parser.add_argument('--wms_uri', default='http://127.0.0.1:8080/wms', type=str)
  parser.add_argument('--im_format', default='jpeg', type=str)
  parser.add_argument('--disable_latitude_compensation','-dlc', action='store_true')
  parser.add_argument('--split_id', type=str,help='Split ID for blocks of $LIMIT sound IDs')
  
  args = parser.parse_args()

  meta_df = pd.read_csv(args.meta_path,low_memory=False)
  # Split sound IDs into blocks of $LIMIT
  split_id = int(args.split_id)
  start_index = (split_id - 1) * LIMIT
  end_index = split_id * LIMIT
  sub_df = meta_df.iloc[start_index:end_index]
  
  sources = list(sub_df['source'])
  keys = list(sub_df['key'])
  lats = list(sub_df['latitude'])
  lons = list(sub_df['longitude'])

  rows = [(str(sources[i]),str(keys[i]),lats[i],lons[i]) for i in range(len(keys))]

  downloader = Downloader(args.save_path,layer='sentinel', radius=args.radius,
                          wms_uri=args.wms_uri, height=args.height, width=args.width,
                          im_format=args.im_format,disable_latitude_compensation=False)
  
  downloader.process(rows)

if __name__ == '__main__':
  cli_main()
