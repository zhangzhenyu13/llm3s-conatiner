# num_tokens: maximum tokens of one prompt and its input
# num_prompt_instructions: how many new instructions to generate
# num_prompt_instructions: the fewshot demonstrations used to build one input prompt

task=TaskName
proxyServer=${server ip running the chatGPT-server.py}

python generate_instruction.py generate_instruction_following_data \
    --request_batch_size=4 \
    --num_instructions_to_generate=1  \
    --output_dir="./data/${task}/" \
    --num_prompt_instructions=3 \
    --num_tokens=128 \
    --api="chat" \
    --model_name="gpt-3.5-turbo" \
    --service_host=$proxyServer \
    --service_port=1238 \
    --service_method="ouryx05private" \
    --seed_tasks_path="./data/${task}/seeds.json" \
    --prompt_path="./data/${task}/prompt_prior.txt"

