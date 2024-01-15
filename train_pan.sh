python ./code/train_Mutual_Reliable.py --dataset_name Pancreas_CT --model mine3d_v1 --exp mine --labelnum 6 --gpu 0 &&\
# python ./code/train_Mutual_Reliable.py --dataset_name Pancreas_CT --model mine3d_v1 --exp mine --labelnum 12 --gpu 0 &&\
python ./code/test_3d.py --dataset_name Pancreas_CT --model mine3d_v1 --exp mine --labelnum 6 --gpu 0 
# python ./code/test_3d.py --dataset_name Pancreas_CT --model mine3d_v1 --exp mine --labelnum 12 --gpu 0