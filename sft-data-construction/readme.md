## create an log dir
mkdir -p logs

## Start-Server in a foreign machine
`nohup bash startService.sh > logs/server.log &`


## for test purpose
`python test.py`

some generated texts are in `response.txt`



## for generation purpose
### 1. prepare your prior rules for generation, where an example in `prompt_cn.txt`

### 2. prepare your seed_tasks that are used to generate new instructions and datasets, where an example is in `zh_seed_tasks.json`

### 3. we recommend you to prepare separate tasks seeds (e.g. `seed_rewrite.json`, `seed_nli.json`), because we are going to inject instructions back to our training data. Each seed task should contain fewer than 50 seeds.

### 4. finally run the following
`bash run.sh`

### Some generated are in `genInst/regen.json`
