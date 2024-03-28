conda activate torch1.10
cd "/data/sihan.zhu/myfile/code/DA/DA/apdm/"
clear

python base_main.py --config "/data/sihan.zhu/myfile/code/DA/DA/apdm/train_test_file/base-train-config.yaml"
python infer.py --config "/data/sihan.zhu/myfile/code/DA/DA/apdm/train_test_file/base-test-config.yaml"