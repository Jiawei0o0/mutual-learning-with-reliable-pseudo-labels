python ./code/train_Mutual_Reliable.py --dataset_name LA --model mine3d_v1 --exp mine --labelnum 8 --gpu 1 &&\
python ./code/train_Mutual_Reliable.py --dataset_name LA --model mine3d_v1 --exp mine --labelnum 16 --gpu 1 &&\
python ./code/test_3d.py --dataset_name LA --model mine3d_v1 --exp mine --labelnum 8 --gpu 1 &&\
python ./code/test_3d.py --dataset_name LA --model mine3d_v1 --exp mine --labelnum 16 --gpu 1