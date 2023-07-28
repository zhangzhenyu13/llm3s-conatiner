for fn in $(ls data)
do
echo data/$fn/regen.json
cat data/$fn/regen.json >> task_seeds.json

done