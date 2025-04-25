PYTHON=${PYTHON:-"python"}

FILE=$1
JOB_NAME=$2
GPUS=${GPUS:-1}
PARTITION="VI_OP_1080TI"
GPUS_PER_NODE=4
CPUS_PER_TASK=5


srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    $PYTHON -u $FILE ${@:3}


cd /private_data/personal/pengliang/GUPNet_master/GUPNet-main/code

cp /private_data/personal/pengliang/dla34-ba72cf86.pth /root/.cache/torch/hub/checkpoints/
cd /pvc_user/pengliang/GUPNet_master/GUPNet-main/code
#CUDA_VISIBLE_DEVICES=0 python tools/train_val.py --config /pvc_user/pengliang/GUPNet_master/GUPNet-main/code/experiments/1025.yaml | tee log_reproduce_abla_base_noc_fix_depthBug_weighted_loss_nocUncern_RoIAlign_InsDepth_Uncern_removeOrgDepth_fixInferDepth_v2_removeGeo_300_decay200.txt
#CUDA_VISIBLE_DEVICES=0 python tools/train_val.py --config /pvc_user/pengliang/GUPNet_master/GUPNet-main/code/experiments/1025.yaml | tee log_reproduce_abla_base_noc_fix_depthBug_weighted_loss_nocUncern_RoIAlign_InsDepth_Uncern_removeOrgDepth_fixInferDepth_v2_removeGeo_300_LogDepth.txt
CUDA_VISIBLE_DEVICES=2 python tools/train_val.py --config /pvc_user/pengliang/GUPNet_master/GUPNet-main/code/experiments/1025.yaml | tee log_reproduce_abla_base_noc_fix_depthBug_weighted_loss_nocUncern_RoIAlign_InsDepth_Uncern_removeOrgDepth_fixInferDepth_v2_removeGeo_300_LogDepth_AugGT_3.txt

CUDA_VISIBLE_DEVICES=2 python tools/train_val.py --config /pvc_user/pengliang/GUPNet_master/GUPNet-main/code/experiments/1025.yaml | tee log_reproduce_abla_base_noc_fix_depthBug_weighted_loss_nocUncern_RoIAlign_InsDepth_Uncern_removeOrgDepth_fixInferDepth_v2_removeGeo_300_LogDepth_CaDDNDepth_1.txt

CUDA_VISIBLE_DEVICES=0,1 python tools/train_with_test.py --config experiments/1025.yaml