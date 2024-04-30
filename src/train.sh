
## CNNLSTM DEBUG
# python train_final.py --batch_size 2 --n_frames 10 --debug
# python train_final.py --batch_size 2
# python train_final.py --batch_size 2 --n_frames 10 --debug --decoder_types TransformerEncoderCls
python train_final.py --batch_size 2 --n_frames 10 --debug --decoder_types TransformerFull
