# LMAN Small-SCALE model (x2) 
# python main.py --model LMAN --scale 2 --n_resgroups 4 --n_feats 64 --patch_size 96 --save LMAN_base_x2 --reset 

# LMAN Small-SCALE model (x3) - from LMAN Small-SCALE model (x2)
# python main.py --model LMAN --scale 3 --n_resgroups 4 --n_feats 64 --patch_size 144 --save LMAN_base_x3 --reset --pre_train ../experiment/LMAN_base_x2/model/model_best.pt

# LMAN Small-SCALE model (x4) - from LMAN Small-SCALE model (x2)
# python main.py --model LMAN --scale 4 --n_resgroups 4 --n_feats 64 --patch_size 192 --save LMAN_base_x4 --reset --pre_train ../experiment/LMAN_base_x2/model/model_best.pt


# LMAN Small-SCALE model (x8) - from LMAN Small-SCALE model (x2)
# python main.py --model LMAN --scale 8 --n_resgroups 4 --n_feats 64 --patch_size 384 --save LMAN_base_x8 --reset --pre_train ../experiment/LMAN_base_x2/model/model_best.pt

# LMAN in the paper (x2)
# python main.py --model LMAN --scale 2 --n_resgroups 16 --n_feats 64 --patch_size 96 --save LMAN_x2_16 --reset

# LMAN in the paper (x3) - from LMAN (x2)
# python main.py --model LMAN --scale 3 --n_resgroups 16 --n_feats 64 --patch_size 144 --save LMAN_x3_16 --reset --pre_train ../experiment/LMAN_x2_16/model/model_best.pt

# LMAN in the paper (x4) - from LMAN (x2)
# python main.py --model LMAN --scale 4 --n_resgroups 16 --n_feats 64 --patch_size 192 --save LMAN_x4_16 --reset --pre_train ../experiment/LMAN_x2_16/model/model_best.pt

# LMAN in the paper (x8) - from LMAN (x2)
python main.py --model LMAN --scale 8 --n_resgroups 16 --n_feats 64 --patch_size 384 --save LMAN_x8_16 --reset --pre_train ../experiment/LMAN_x2_16/model/model_best.pt


# Standard benchmarks (Ex. LMAN_x2)
# python main.py --model LMAN --save test_base_x2 --data_test Set5+Set14+B100+Urban100+Manga109 --scale 2 --n_resgroups 4 --n_feats 64 --pre_train ../experiment/model_base_x2.pt --test_only --save_results --save_gt  
# python main.py --model LMAN --save test_x2 --data_test Set5+Set14+B100+Urban100+Manga109 --scale 2 --n_resgroups 16 --n_feats 64 --pre_train ../experiment/model_x2.pt --test_only --save_results --save_gt  

# Standard benchmarks (Ex. LMAN_x3)
# python main.py --model LMAN --save test_base_x3 --data_test Set5+Set14+B100+Urban100+Manga109 --scale 3 --n_resgroups 4 --n_feats 64 --pre_train ../experiment/model_base_x3.pt --test_only #--save_results --save_gt 
# python main.py --model LMAN --save test_x3 --data_test Set5+Set14+B100+Urban100+Manga109 --scale 3 --n_resgroups 16 --n_feats 64 --pre_train ../experiment/model_x3.pt --test_only #--save_results --save_gt 

# Standard benchmarks (Ex. LMAN_x4)
# python main.py --model LMAN --save test_base_x4 --data_test Set5+Set14+B100+Urban100+Manga109 --scale 4 --n_resgroups 4 --n_feats 64 --pre_train ../experiment/model_base_x4.pt --test_only # --save_results --save_gt
# python main.py --model LMAN --save test_x4 --data_test Set5+Set14+B100+Urban100+Manga109 --scale 4 --n_resgroups 16 --n_feats 64 --pre_train ../experiment/model_x4.pt --test_only # --save_results --save_gt

# Standard benchmarks (Ex. LMAN_x8)
# python main.py --model LMAN --save test_base_x8 --data_test Set5_x8+Set14_x8+B100_x8+Urban100_x8+Manga109_x8 --scale 8 --n_resgroups 4 --n_feats 64 --pre_train ../experiment/model_base_x8.pt  --test_only #--save_results --save_gt
# python main.py --model LMAN --save test_x8 --data_test Set5_x8+Set14_x8+B100_x8+Urban100_x8+Manga109_x8 --scale 8 --n_resgroups 16 --n_feats 64 --pre_train ../experiment/model_x8.pt  --test_only #--save_results --save_gt

