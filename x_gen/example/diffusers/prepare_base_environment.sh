# 基础环境安装
cd $INFER_WORKSPACE/ascendx_video

pip install ascendx_video-*-none-any.whl

# 图灵算子环境安装
cd $INFER_WORKSPACE/AscendCloud/opp/turing
unzip CANN-cloud-turbo-ops-26.0.2-$MACHINE_TYPE-CANN8.5.1-torch2.9.0-py3.11.4-aarch64.zip
bash cloud_turbo_ops-26.0.2-$MACHINE_TYPE-CANN8.5.1-aarch64.run
export LD_LIBRARY_PATH=/usr/local/Ascend/cann-8.5.1/opp/vendors/turing_ascend_cloud/op_api/lib/:${LD_LIBRARY_PATH}
pip install cloud_turbo_ops-26.0.2+torch2.9.0.$MACHINE_TYPE-cp311-cp311-linux_aarch64.whl

# 安装MindIE-SD
cd $INFER_WORKSPACE/ascendx_video
git clone https://gitcode.com/Ascend/MindIE-SD.git && cd MindIE-SD
git checkout eaccd2142df164f37a8f622d02c493ad3ea0854d
python setup.py bdist_wheel
cd dist
pip install mindiesd-*.whl
cp ../mindiesd/plugin/libPTAExtensionOPS.so $INFER_WORKSPACE/ascendx_video

# 安装MindSpeed库
cd $INFER_WORKSPACE/ascendx_video
git clone https://gitcode.com/Ascend/MindSpeed.git
cd MindSpeed && git checkout $target_commit c357f9d2fd6f35365c9dc86f4792b8779420b667
pip install -e .

# 安装yunchang库
cd $INFER_WORKSPACE/ascendx_video
git clone https://github.com/feifeibear/long-context-attention.git
cd long-context-attention && git checkout 7a52abd669efb35e550680a239e1745b620b2bae
cp -f $INFER_WORKSPACE/AscendCloud/aigc_train/torch_npu/DiffSynth-Studio/wan2.2-longframe/yunchang.patch ./
git apply yunchang.patch
pip install -e .
