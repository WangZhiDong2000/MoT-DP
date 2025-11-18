export PYTHONPATH="$(dirname $0)/..":$PYTHONPATH

python tools/data_converter/nuscenes_converter.py nuscenes \
    --root-path ../data/nuscenes_mini \
    --canbus ../data/nuscenes_mini \
    --out-dir ../data/infos_mini/ \
    --extra-tag nuscenes \
    --version v1.0-mini

# /mnt/data2/nuscenes
# python tools/data_converter/nuscenes_converter.py nuscenes \
#     --root-path ../data/nuscenes \
#     --canbus ../data/nuscenes \
#     --out-dir ../data/infos/ \
#     --extra-tag nuscenes \
#     --version v1.0

