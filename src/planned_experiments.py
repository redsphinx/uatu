# ------------------------------------------------------------------------------------
# experiments images -> siamese_cnn_image.py
# ------------------------------------------------------------------------------------
# 0	CLR vs. LR with decay on [viper, grid, prid450, cuhk01]
#
# no CLR
# 0_0	no CLR: lr=0.001, decay=0.95
# 0_1	no CLR: lr=0.0001, decay=0.95
# 0_2	no CLR: lr=0.00001, decay=0.95
# 0_3	no CLR: lr=0.000001, decay=0.95
#
# with CLR
# 0_4	with CLR: min=0.000001, max=0.00001
# 0_5	with CLR: min=0.00001, max=0.0001
# 0_6	with CLR: min=0.0001, max=0.001
# 0_7	with CLR: min=0.00005, max=0.001 [BASELINE]
#
# ------------------------------------------------------------------------------------
#
# 1	neural layers: type of merging on [viper, grid, prid450, cuhk01]
#
# 1_0	neural_distance=absolute
# 1_1	neural_distance=subtract
# 1_2	neural_distance=concatenate
# 1_3	neural_distance=divide
# 1_4	neural_distance=add
# 1_5	neural_distance=multiply
#
# ------------------------------------------------------------------------------------
#
# 2	non-neural vs. neural on [viper, grid, prid450, cuhk01]
#
# 2_0	cost_module_type=euclidean, lr=0.00001
# 2_1 cost_module_type=cosine, lr=0.00001
#
# ------------------------------------------------------------------------------------
#
# 3	training: single dataset
#
# 3_0	no batchnorm
# (others can be found in experiments 0 with CLR)
#
# ------------------------------------------------------------------------------------
#
# 4	training: mix all datasets including test
#
# 4_0	no batchnorm
# 4_1 with batchnorm
#
# ------------------------------------------------------------------------------------
#
# 5	training: train on all mixed, exclude test. Then retrain trained network on the test.
#
# 5_0	no batch_norm
# 5_1	with batch_norm
#
# ------------------------------------------------------------------------------------
#
# 6	training: train on all ordered for subset={viper, grid, prid450}
#
# 6_0 train order: grid, prid450, viper
# 6_1 train order: prid450, grid, viper
# 6_2 train order: grid, viper, prid450
# 6_3 train order: viper, grid, prid450
# 6_4 train order: viper, prid450, grid
# 6_5 train order: prid450, viper, grid
#
# ------------------------------------------------------------------------------------
#
# 7   priming
#
# 7_0 TODO
#
#
# ------------------------------------------------------------------------------------
# experiments video -> siamese_cnn_video.py
# ------------------------------------------------------------------------------------
#
# 8	3D convolution vs. cnn_lstm (single dataset)
#
# 8_0	video_head_type=3d_convolution, no batchnorm
# 8_1	video_head_type=3d_convolution, with batchnorm
# 8_2	video_head_type=cnn_lstm
#
# ------------------------------------------------------------------------------------
#
# 9	training: mixing all datasets, including test
#
# 9_0	3d_conv, no batchnorm
# 9_1	3d_conv, with batchnorm
# 9_2	cnn_lstm
#
# ------------------------------------------------------------------------------------
#
# 10 	training: retrain network on test
#
# 10_0	3d_conv, no batchnorm
# 10_1	3d_conv, with batchnorm
# 10_2	cnn_lstm
#
#
#
