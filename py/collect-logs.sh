#!/bin/bash -ex

if [ $# -lt 4 ]; then
  echo "Usage: $0 <config-dir> <job-uuid> <output-dir> <ssh-key>"
  exit 1
fi

config_dir=$1
job_uuid=$2
output_dir=$3
ssh_key=${4:-~/.ssh/id_rsa}
type=${5:-"normal"}

mkdir -p $output_dir

for tier in 0 1
do
  if [ ! -f "$config_dir/remote.tier$tier.conf" ]; then
    continue
  fi

  remotes=$(cat "$config_dir/remote.tier${tier}.conf" | cut -d' ' -f1)

  if [ ${type} == "faux" ]; then
    echo $remotes | tr ' ' '\n' | xargs -t -P16 -I% sh -c "scp -i ${ssh_key} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -r glinthawk@%:/tmp/stats_${job_uuid}_${tier}.csv ${output_dir}/stats_%_${tier}.csv || exit 0"

    echo $remotes | tr ' ' '\n' | xargs -t -P16 -I% sh -c "scp -i ${ssh_key} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -r glinthawk@%:/tmp/promptinfo_${job_uuid}_${tier}.csv ${output_dir}/promptinfo_%_${tier}.csv || exit 0"
  else

    echo $remotes | tr ' ' '\n' | xargs -t -P16 -I% scp -i ${ssh_key} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -r glinthawk@%:/tmp/stats_${job_uuid}_${tier}.csv $output_dir/stats_%_${tier}.csv

    echo $remotes | tr ' ' '\n' | xargs -t -P16 -I% scp -i ${ssh_key} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -r glinthawk@%:/tmp/promptinfo_${job_uuid}_${tier}.csv $output_dir/promptinfo_%_${tier}.csv
  fi

done
