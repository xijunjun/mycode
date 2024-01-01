import  os





yaml_path='/disks/disk1/Workspace/Project/Pytorch/TrainTemplate/project_cfg/e2e_test00.yaml'
train_result_root='/home/tao/disk1/Workspace/TrainResult/image2image'


expname=yaml_path.split('/')[-1].split('.')[0]
exp_base_root=os.path.join(train_result_root,expname)

os.system('python train.py --exp_base_root {} --yaml_path {}'.format(exp_base_root,yaml_path))


