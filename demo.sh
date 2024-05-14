CUDA_ID=1
# SEQ_NAME=scene0608_00
# SEQ_NAME=scene0568_00
SEQ_NAME=scene0025_00
# SEQ_NAME=scene0050_00

echo [INFO] start mask clustering
CUDA_VISIBLE_DEVICES=$CUDA_ID python main.py --config scannet --debug --seq_name_list $SEQ_NAME
echo [INFO] finish mask clustering

echo [INFO] visualizing
python -m visualize.vis_scene --config scannet --seq_name $SEQ_NAME
echo [INFO] Please follow the instruction of pyviz to visualize the scene