#!/bin/bash
#SBATCH --account=def-ouriadov
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=30G
#SBATCH --time=20:00:00
#SBATCH --output=/home/fguo24/projects/def-ouriadov/fguo24/GuoLab_students/tzou/BsplineFFD_registration_template/Log_PixelMorph/bspline_Tongji_temp_train_1000_1e-1.out
module load cuda cudnn
source /home/fguo24/projects/def-ouriadov/fguo24/GuoLab_students/tzou/py3.10_pytorch2.1.2/bin/activate

# train
python /home/fguo24/projects/def-ouriadov/fguo24/GuoLab_students/tzou/BsplineFFD_registration_template/bspline_2D_Tongji_train_test.py --train_dir /home/fguo24/projects/def-ouriadov/fguo24/GuoLab_students/tzou/Tongji_CineLGE_2D --model_dir /home/fguo24/projects/def-ouriadov/fguo24/GuoLab_students/tzou/BsplineFFD_registration_template/Checkpoint_PixelMorph --log_dir /home/fguo24/projects/def-ouriadov/fguo24/GuoLab_students/tzou/BsplineFFD_registration_template/Log_Temp --result_dir  /home/fguo24/projects/def-ouriadov/fguo24/GuoLab_students/tzou/BsplineFFD_registration_template/Result_Temp --csv_train_dir /home/fguo24/projects/def-ouriadov/fguo24/GuoLab_students/tzou/Tongji_CineLGE_2D/Tongji_CineLGE_2D_fguo24.csv --pattern train --lr 0.0001 --n_iter 800 --alpha 1e-1 --beta 1e-2
#test
#python /home/fguo24/projects/def-ouriadov/fguo24/GuoLab_students/tzou/BsplineFFD_registration_template/bspline_2D_Tongji_train_test.py --test_dir /home/fguo24/projects/def-ouriadov/fguo24/GuoLab_students/tzou/Tongji_CineLGE_2D --model_dir /home/fguo24/projects/def-ouriadov/fguo24/GuoLab_students/tzou/BsplineFFD_registration_template/Checkpoint_Temp --log_dir /home/fguo24/projects/def-ouriadov/fguo24/GuoLab_students/tzou/BsplineFFD_registration_template/Log_Temp --result_dir /home/fguo24/projects/def-ouriadov/fguo24/GuoLab_students/tzou/BsplineFFD_registration_template/Result_Temp --csv_test_dir /home/fguo24/projects/def-ouriadov/fguo24/GuoLab_students/tzou/Tongji_CineLGE_2D/Tongji_CineLGE_2D_test.csv --checkpoint_path /home/fguo24/projects/def-ouriadov/fguo24/GuoLab_students/tzou/BsplineFFD_registration_template/Checkpoint_PixelMorph/tongji_train.pth --pattern test
