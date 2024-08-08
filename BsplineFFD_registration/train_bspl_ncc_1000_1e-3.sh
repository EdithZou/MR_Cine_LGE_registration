#!/bin/bash
#SBATCH --account=def-gawright
#SBATCH --gres=gpu:v100l:1
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --output=/home/fguo/projects/def-gawright/fguo/GuoLab_students/tzou/BsplineFFD_registration/Log_Bspline/bspline_train.out
module load cuda cudnn

source /home/fguo/projects/def-gawright/fguo/GuoLab_students/tzou/vm_torch/bin/activate

# training and testing
python /home/fguo/projects/def-gawright/fguo/GuoLab_students/tzou/BsplineFFD_registration/bspline_2D_train_test.py  --train_dir  /home/fguo/projects/def-gawright/fguo/GuoLab_students/tzou/Xiehe_CineLGE_AHAsegment2d/train/base --model_dir /home/fguo/projects/def-gawright/fguo/GuoLab_students/tzou/BsplineFFD_registration/Checkpoint_Bspline --log_dir /home/fguo/projects/def-gawright/fguo/GuoLab_students/tzou/BsplineFFD_registration/Log_Bspline --result_dir /home/fguo/projects/def-gawright/fguo/GuoLab_students/tzou/BsplineFFD_registration/Result_Bspline --pattern train --lr 0.0001 --n_iter 2000  --alpha 1e-4 --beta 1e-3
#python /home/fguo/projects/def-gawright/fguo/GuoLab_students/tzou/BsplineFFD_registration/bspline_2D_train_test.py  --test_dir  /home/fguo/projects/def-gawright/fguo/GuoLab_students/tzou/Xiehe_CineLGE_AHAsegment2d/test/base --model_dir /home/fguo/projects/def-gawright/fguo/GuoLab_students/tzou/BsplineFFD_registration/Checkpoint_Bspline --log_dir /home/fguo/projects/def-gawright/fguo/GuoLab_students/tzou/BsplineFFD_registration/Log_Bspline --result_dir /home/fguo/projects/def-gawright/fguo/GuoLab_students/tzou/BsplineFFD_registration/Result_Bspline --checkpoint_path  /home/fguo/projects/def-gawright/fguo/GuoLab_students/tzou/BsplineFFD_registration/Checkpoint_Bspline/Xiehe_bspline_3000_train.pth --pattern test
