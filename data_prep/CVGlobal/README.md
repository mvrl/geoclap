# Satellite imagery download using mappproxy
This a code taken from Dr.Nathan Jacobs' repo [CVGlobal](https://github.com/mvrl/CVGlobal). This can be used to download satellite imagery from different sources. In our project we will use it to download sentinel2-cloudless. To proceed, we require a `.csv` file with fields: `source, key, latitude, longitude`. Note: `source` is the source from where the geotagged audio sample was downloaded from. 

## Steps (local):
1. Make sure you are in right conda environment:
    ```
    conda activate geoclap
    ```
2. 
    ```
    cd path_to_CVGlobal_directory
    ```
3. Create necessary files and folders necessary before launching download job:
    ```
    python create_dirs.py
    ```
4. Make sure to change the paths in files:\
 `nginx.conf`, `mapproxy.yml`, and `launch_cache_servers.sh`\
 so that they point to your appropriate locations.
5. Launch the cache servers in the background:
    ```
    bash launch_cache_servers.sh &
    ```
6. Now, you should first check if everything is working by quickly downloading one image using the script:
    ```
    python demo.py
    ```
    If everything is working, you should see a sentinel image `demo.jpeg` downloaded to `./logs`.
7. Now, launch download job for desired split (for example: `--split_id 1`) of the csv file.
    ```
    python download_sentinel.py --meta_path path_to_latlong_csv --save_path path_to_save_images --split_id 1
    ```
    Note: The code currently splits the csv file into chunks of $100k$ rows. Update the script to change this default setting.

## Steps (LSF-like system):
1. Just like above (steps: 2,3,4), make sure the appropriate paths are created and the files `nginx.conf`, `mapproxy.yml`, and `launch_cache_servers.sh` are updated accordingly.
 
2. Launch cache servers using a docker image `docker(ksubash/sat2audio)`:
    ```
    LSF_DOCKER_PORTS="8080:8080 8081:8081 8082:8082 8083:8083 8084:8084 8085:8085" bsub \
        -q general -G compute-jacobsn \
        -R 'rusage[mem=20GB] select[port8080=1 && port8081=1 && port8082=1 && port8083=1 && port8084=1 && port8085=1]' \
        -M 40GB -a 'docker(ksubash/sat2audio)' \
        -o path_to_CVGlobal_directory/launch_cache.log \
        bash -c 'source /opt/conda/bin/activate sat2audio;cd path_to_CVGlobal_directory; bash launch_cache_servers.sh'
    ```
    Note: Please replace appropriate entries for your compute group (here `compute-jacobsn`) and `path_to_CVGlobal_directory`. 

    Check the log file `launch_cache.log` and make sure that the launch of cache servers is successful.

3. Look for the `ip-address` of the `EXEC_HOST` where the servers are running from step 2 by using the command:\
    ```bjobs -w```
4. Now launch the download job using:
    ```
    bsub -q general -G compute-jacobsn -R 'rusage[mem=60GB]' -M 80GB -o sentinel_download.log \
        -a 'docker(ksubash/sat2audio)' bash -c 'source /opt/conda/bin/activate sat2audio;cd path_to_CVGlobal_directory;python download_sentinel.py --wms_uri http://compute1-exec-165.ris.wustl.edu:8080/wms --split_id 1  --meta_path path_to_latlong_csv --save_path path_to_save_images'
    ```
    Note: Here `compute1-exec-165.ris.wustl.edu` is an example of `ip-address` of the `EXEC_HOST` where the servers are running. Change other fields (`path_to_CVGlobal_directory`, `path_to_latlong_csv`, `path_to_save_images`) accordingly.