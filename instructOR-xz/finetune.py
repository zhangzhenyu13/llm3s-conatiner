from datasets import load_dataset
from dataloader import load_dataset_local
import os
from uniem.finetuner import FineTuner
# med_dataset_dict = load_dataset('vegaviazhang/Med_QQpairs')
med_dataset_dict = load_dataset_local('vegaviazhang/Med_QQpairs')

model_hub_local = os.path.join(os.environ['HOME'], "CommonModels")

print(med_dataset_dict['train'][0])
print(med_dataset_dict['train'][1])

dataset = med_dataset_dict['train']
dataset = dataset.rename_columns({'question1': 'sentence1', 'question2': 'sentence2'})

#  Med_QQpairs只有训练集，我们需要手动划分训练集和验证集
dataset = dataset.train_test_split(test_size=0.1, seed=42)
dataset['validation'] = dataset.pop('test')

finetuner = FineTuner.from_pretrained(
    os.path.join(model_hub_local,'moka-ai/m3e-base'),
    dataset=dataset
)
fintuned_model = finetuner.run(epochs=3)
