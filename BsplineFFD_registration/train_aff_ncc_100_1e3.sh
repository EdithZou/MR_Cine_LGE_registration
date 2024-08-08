#!/bin/bash
#SBATCH --account=def-gawright
#SBATCH --gres=gpu:v100l:1
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --output=/home/fguo/projects/def-gawright/fguo/GuoLab_students/tzou/BsplineFFD_registration/Log_Affine/affine_train_1000_1e-4.out
module load cuda cudnn

source /home/fguo/projects/def-gawright/fguo/GuoLab_students/tzou/vm_torch/bin/activate

# training and testing
python  /home/fguo/projects/def-gawright/fguo/GuoLab_students/tzou/BsplineFFD_registration/affine_2D_train_test.py --train_dir /home/fguo/projects/def-gawright/fguo/GuoLab_students/tzou/FDMRI_3 --model_dir /home/fguo/projects/def-gawright/fguo/GuoLab_students/tzou/BsplineFFD_registration/Checkpoint_Affine --log_dir /home/fguo/projects/def-gawright/fguo/GuoLab_students/tzou/BsplineFFD_registration/Log_Affine --result_dir /home/fguo/projects/def-gawright/fguo/GuoLab_students/tzou/BsplineFFD_registration/Result_Affine --pattern train --lr 0.00001 --n_iter 1000
python /home/fguo/projects/def-gawright/fguo/GuoLab_students/tzou/BsplineFFD_registration/affine_2D_train_test.py  --test_dir  /home/fguo/projects/def-gawright/fguo/GuoLab_students/tzou/FDMRI_3 --model_dir /home/fguo/projects/def-gawright/fguo/GuoLab_students/tzou/BsplineFFD_registration/Checkpoint_Affine --log_dir /home/fguo/projects/def-gawright/fguo/GuoLab_students/tzou/BsplineFFD_registration/Log_Affine --result_dir /home/fguo/projects/def-gawright/fguo/GuoLab_students/tzou/BsplineFFD_registration/Result_Affine --checkpoint_path  /home/fguo/projects/def-gawright/fguo/GuoLab_students/tzou/BsplineFFD_registration/Checkpoint_Affine/affine.pth --pattern test
