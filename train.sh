COMMAND_ARG=$@
: ${VLLM:=0}
: ${NPROC:=1}
NUM_PROC=$((NPROC - VLLM))
accelerate launch --config_file configs/deepspeed_zero2.yaml --deepspeed_config_file configs/ds_config.json --mixed_precision=${FP:=fp16} --num_processes ${NUM_PROC} $COMMAND_ARG
