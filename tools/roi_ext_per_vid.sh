dataset=$1
root=/home/forwchen/${dataset}/frames/
split=$2
gpuid=$3
outdir=/home/forwchen/${dataset}/roi_pkl
export GLOG_minloglevel=2

for vid in $(cat $split)
do
    echo $vid
    RET=1
    until [ ${RET} -eq 0 ]; do
        python roi_ext.py $gpuid $root/$vid $outdir/${vid}
        RET=$?
        echo 'script failed, restart...'
        sleep 1
    done
done
