#This is a demo to download one image.
from download_sentinel import bounding_box_from_circle, download
from argparse import ArgumentParser, RawTextHelpFormatter


parser = ArgumentParser(description='', formatter_class=RawTextHelpFormatter)
parser.add_argument('--wms_uri', default='http://127.0.0.1:8080/wms', type=str)
args = parser.parse_args()
lat = 52.49352874
lon = 13.4281294
layer='sentinel'
radius=1024
wms_uri=args.wms_uri
height=2048
width=2048
im_format="jpeg"
row_parser='latlon'
disable_latitude_compensation=False
lmdb_display_name = 'demo'

# build key from raw strings to avoid floating point issues 
key = f"{lat},{lon}"

key_bytes = key.encode("utf-8") 

l,b,r,t = bounding_box_from_circle(float(lat),float(lon),radius,
              disable_latitude_compensation=disable_latitude_compensation)
image_url = f"{wms_uri}?service=WMS&version=1.1.1&request=GetMap&layers={layer}&styles=&width={width}&height={height}&srs=EPSG:4326&bbox={l},{b},{r},{t}&format=image/{im_format}"
print(l,b,r,t)
print(f'{lmdb_display_name} key: "{key}" downloading: {image_url}')
out_file = "./logs/demo."+im_format
download(image_url,out_file,redownload=True)
