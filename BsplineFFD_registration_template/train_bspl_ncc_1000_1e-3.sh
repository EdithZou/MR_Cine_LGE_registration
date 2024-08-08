#!/bin/bash
#SBATCH --account=def-gawright
#SBATCH --gres=gpu:v100l:1
#SBATCH --mem=32G
#SBATCH --time=5:00:00
#SBATCH --output=/home/fguo/projects/def-gawright/fguo/GuoLab_students/tzou/Log_Bspline/bspline_train_ffd.out
module load cuda cudnn

source /home/fguo/projects/def-gawright/fguo/GuoLab_students/tzou/vm_torch/bin/activate

# training and testing
python /home/fguo24/projects/def-ouriadov/fguo24/GuoLab_students/tzou/BsplineFFD_registration/bspline_2D_train_test.py  --train_dir  /home/fguo24/projects/def-ouriadov/fguo24/GuoLab_students/tzou/Xiehe_CineLGE_AHAsegment2d/train/base --model_dir /home/fguo24/projects/def-ouriadov/fguo24/GuoLab_students/tzou/Checkpoint_Bspline --log_dir /home/fguo24/projects/def-ouriadov/fguo24/GuoLab_students/tzou/Log_Bspline --result_dir /home/fguo24/projects/def-ouriadov/fguo24/GuoLab_students/tzou/Result_Bspline --pattern train --lr 0.00001 --n_iter 200  --alpha 10
#python /home/fguo/projects/def-gawright/fguo/GuoLab_students/tzou/BsplineFFD_registration/bspline_2D_train_test.py  --test_dir  /home/fguo/projects/def-gawright/fguo/GuoLab_students/tzou/Xiehe_CineLGE_AHAsegment2d/test/base --model_dir /home/fguo/projects/def-gawright/fguo/GuoLab_students/tzou/Checkpoint_Bspline --log_dir /home/fguo/projects/def-gawright/fguo/GuoLab_students/tzou/Log_Bspline --result_dir /home/fguo/projects/def-gawright/fguo/GuoLab_students/tzou/Result_Bspline --checkpoint_path  /home/fguo/projects/def-gawright/fguo/GuoLab_students/tzou/Checkpoint_Bspline/bspline_2000_dice.pth --pattern test
