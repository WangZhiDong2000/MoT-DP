export PYTHONPATH="$(dirname $0)/..":$PYTHONPATH

# conda run -n robodiff python tools/data_converter/nuscenes_converter.py nuscenes \
#     --root-path ../data/nuscenes \
#     --canbus ../data/nuscenes \
#     --out-dir ../data/infos/ \
#     --extra-tag nuscenes \
#     --version v1.0-mini

# /mnt/data2/nuscenes
python tools/data_converter/nuscenes_converter.py nuscenes \
    --root-path ../data/nuscenes \
    --canbus ../data/nuscenes \
    --out-dir ../data/infos/ \
    --extra-tag nuscenes \
    --version v1.0

