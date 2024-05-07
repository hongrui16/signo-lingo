
## CNNLSTM DEBUG
# python train_final.py --batch_size 2 --n_frames 10 --debug
# python train_final.py --batch_size 2
# python train_final.py --batch_size 2 --n_frames 10 --debug --decoder_types TransformerEncoderCls
# python train_final.py --batch_size 2 --n_frames 10 --debug --decoder_types TransformerFull
# python train_final.py --batch_size 50  --dataset_name WLASL300 --load_kpts_only #--debug 
python train_final.py --batch_size 200  --dataset_name WLASL100 --load_kpts_only --decoder_types TransformerFull #--debug 
# python train_final.py --batch_size 2 --debug  --dataset_name WLASL100 --load_kpts_only
