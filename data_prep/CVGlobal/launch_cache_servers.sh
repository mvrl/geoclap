#!/usr/bin/env bash 

set -Eeuo pipefail

trap "kill 0" EXIT 

array=( 8081 8082 8083 8084 8085 )

for i in "${array[@]}"
do

  nice mapproxy-util serve-develop /storage1/fs1/jacobsn/Active/user_k.subash/projects/geoclap/data_prep/CVGlobal/mapproxy.yml -b :$i & 

done

nice nginx -c "$(readlink -f /storage1/fs1/jacobsn/Active/user_k.subash/projects/geoclap/data_prep/CVGlobal/nginx.conf)" &

wait

