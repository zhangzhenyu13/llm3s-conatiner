cd rlhf-ppo

export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=1,2

OUTPUT="$HOME/SharedData/RLHF/models/bloomz-0.56b-rlhf"

mkdir -p $OUTPUT
export MASTER_PORT=1269

# DeepSpeed Team
ACTOR_MODEL_PATH="$HOME/SharedData/RLHF/models/bloomz-0.56b-sft"

# CRITIC_MODEL_PATH="$HOME/SharedData/RLHF/models/bloomz-0.56b-reward"
CRITIC_MODEL_PATH="$HOME/CommonModels/your-org/rewardS1/"

ACTOR_ZERO_STAGE=2
CRITIC_ZERO_STAGE=2


Num_Padding_at_Beginning=0 # this is model related

Actor_Lr=9.65e-6
Critic_Lr=5e-6

export MASTER_PORT=1269
deepspeed --master_port $MASTER_PORT train-rlhf.py \
   --data_path "datapath/xz-rlhf" \
   --data_split 2,4,4 \
   --data_output_path $OUTPUT/tmp-data \
   --actor_model_name_or_path $ACTOR_MODEL_PATH \
   --critic_model_name_or_path $CRITIC_MODEL_PATH \
   --num_padding_at_beginning $Num_Padding_at_Beginning \
   --padding_side right \
   --per_device_train_batch_size 1 \
   --per_device_mini_train_batch_size 1 \
   --generation_batch_numbers 1 \
   --ppo_epochs 1 \
   --max_answer_seq_len 64 \
   --max_prompt_seq_len 64 \
   --actor_lora_dim 128 \
   --actor_lora_module_name transformer.h. \
   --actor_learning_rate ${Actor_Lr} \
   --critic_learning_rate ${Critic_Lr} \
   --num_train_epochs 1 \
   --lr_scheduler_type cosine \
   --gradient_accumulation_steps 1 \
   --disable_actor_dropout \
   --num_warmup_steps 100 \
   --deepspeed --seed 1234 \
   --enable_hybrid_engine \
   --actor_zero_stage $ACTOR_ZERO_STAGE \
   --critic_zero_stage $CRITIC_ZERO_STAGE \
   --enable_ema \
   --output_dir $OUTPUT 
