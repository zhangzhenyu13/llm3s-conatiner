# hosts="dev84 dev85 dev86 dev88 dev89 v100"
hosts="v100 v100-140"
dataPath=/SharedData/Bloom7Bz/data_dir/train-sess.json

configPath=/Projects/llm3s-conatiner/

for h in $hosts
do
{
echo $h
echo $dataPath $h:$(dirname $dataPath)
scp ~/$dataPath $h:~/$(dirname $dataPath) 

# echo $configPath $h:$(dirname $configPath)
# scp -r ~/$configPath $h:~/$(dirname $configPath) 

} &

done
wait

echo "deployed all"
