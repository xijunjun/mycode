import  os

import os
os.environ['CUDA_VISIBLE_DEVICES'] ='0'



yaml_path='/disks/disk1/Workspace/mycode/algotrain/Real-ESRGAN-master/trainyml/eye_sr_debug.yml'

# train_result_root='/disks/disk1/Workspace/mycode/exproot'
# expname=yaml_path.split('/')[-1].split('.')[0]
# exp_base_root=os.path.join(train_result_root,expname)

# os.system('python realesrgan/train.py -exp_base_root {} -opt {}'.format(exp_base_root,yaml_path))


os.system('python realesrgan/train.py  -opt {}'.format(yaml_path))

