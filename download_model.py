from transformers import AutoModelForCausalLM, AutoTokenizer
import os
modelid= "bigscience/bloomz-3b"
savePath= os.path.join(os.environ["HOME"], "OpenModels", modelid)
tokenizer = AutoTokenizer.from_pretrained(modelid)
model = AutoModelForCausalLM.from_pretrained(modelid)
tokenizer.save_pretrained(savePath)
model.save_pretrained(savePath)
