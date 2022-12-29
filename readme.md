# 1 generat gemm 
/FasterTransformer_master/build/bin/t5_gemm 1 1 32 768 12 64 2048 768 12 64 2048 81920 1
/FasterTransformer_master/build/bin/t5_gemm 2 1 32 768 12 64 2048 768 12 64 2048 81920 1
/FasterTransformer_master/build/bin/t5_gemm 4 1 32 768 12 64 2048 768 12 64 2048 81920 1
/FasterTransformer_master/build/bin/t5_gemm 8 1 32 768 12 64 2048 768 12 64 2048 81920 1
merge  gemm.in to one file

# 2 pytorch:
python3 mt5_translate.py \
--ft_model_location \
all_models/t5/bnenfilhiidlomsthurvizh/1 \
--test_ft \
--data_type \
fp16 \
--lib_path \
/data/mt/hbl/FasterTransformer_master/build/lib/libth_t5.so

# 3 triton client:
python3 triton_ft_client.py


