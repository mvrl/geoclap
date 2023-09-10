#This script just creates cache and some sub-directories that might be necessary to launch CVGlobal code jobs
import os

current_path = "./"
dirs = ["sentinel","terrain","terrain","osm","google_satellite"]

for dir in dirs:
    dirpath = os.path.join(current_path,"cache",dir)
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

files = ["access.log","error.log","nginx.pid"]
log_path = os.path.join(current_path,"logs")
if not os.path.exists(log_path):
        os.makedirs(log_path)
for f in files:
    fpath = os.path.join(log_path,f)
    if not os.path.exists(fpath):
       os.mknod(fpath)