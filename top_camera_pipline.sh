set -e
DATA_VERSION="data-0715"
DEVICES="3"
PRETRAIN_WEIGHTS_PATH=''
CONFIG_NAME="solov2_r50_fpn_1x_top_camera"

CONFIG_ROOT="configs/potato/top_camera/"
CONFIG_FILE="${CONFIG_ROOT}/${CONFIG_NAME}.py"
LOG_DIR="${CONFIG_ROOT}/${DATA_VERSION}/${CONFIG_NAME}"
# TRAIN_OUTPUT="${CONFIG_ROOT}/${DATA_VERSION}/train"
# EXPORT_DIR="${CONFIG_ROOT}/${DATA_VERSION}/export"
# WEIGHTS="${TRAIN_OUTPUT}/${CONFIG_NAME}/best_model"
# EVAL_OUTPUT="${CONFIG_ROOT}/${DATA_VERSION}/eval"

# 模型训练
python tools/train.py ${CONFIG_FILE} \
--work-dir ${LOG_DIR} \
--auto-resume \
--gpu-id ${DEVICES} 

# # 模型推理
# python tools/test.py ${CONFIG_FILE} ${}\
# work_dirs/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_balloon.py/latest.pth \
# --work-dir configs/potato/top_camera/
# --eval bbox segm