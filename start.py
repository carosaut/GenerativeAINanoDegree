from transformers import TrainingArguments, Trainer, AutoModelForSequenceClassification
from datasets import load_dataset
import torch
from peft import LoraConfig, get_peft_model
from transformers import Trainer, GPT2Tokenizer, AutoTokenizer
import numpy as np


# fetch dataset 

dataset = load_dataset("amazon_polarity")

# Select train and test data 
train_dataset = dataset['train'].shuffle(seed=42).select(range(1000))
test_dataset = dataset['test'].shuffle(seed=42).select(range(1000))


#get tokenized 
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

def process_text(example):
    return tokenizer(example["content"], truncation=True, padding="max_length")

tokenized_train_dataset = train_dataset.map(process_text, batched=True)
tokenized_test_dataset = test_dataset.map(process_text, batched=True)


#model
model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', 
id2label={0: "NEGATIVE", 1: "POSITIVE"},
label2id={"NEGATIVE": 0, "POSITIVE": 1},
num_labels=2)


training_args = TrainingArguments(
    output_dir="./results",
    per_device_eval_batch_size=8,
    per_device_train_batch_size=8,
    evaluation_strategy="epoch",
    num_train_epochs=5,
    do_eval=True,
    save_strategy='epoch',
)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {'accuracy': (predictions == labels).mean()}

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=tokenized_test_dataset,
    compute_metrics=compute_metrics,
    train_dataset=tokenized_train_dataset

)

initial_evaluation = trainer.evaluate()
print("Initial Evaluation:", initial_evaluation) # {'eval_loss': 6.4715752601623535, 'eval_model_preparation_time': 0.007, 'eval_accuracy': 0.21666666666666667, 'eval_runtime': 260.3621, 'eval_samples_per_second': 1.152, 'eval_steps_per_second': 1.152}


config = LoraConfig(target_modules=["classifier"],)
lora_model = get_peft_model(model, config)
lora_model.print_trainable_parameters()


ft_training_args = TrainingArguments(
    output_dir="./lora_results",
    per_device_eval_batch_size=8,
    per_device_train_batch_size=8,
    evaluation_strategy="epoch",
    num_train_epochs=5,
    do_eval=True,
    save_strategy='epoch',
)

ft_trainer = Trainer(
    model=lora_model,
    args=ft_training_args,
    eval_dataset=tokenized_test_dataset,
    compute_metrics=compute_metrics,
    train_dataset=tokenized_train_dataset

)

ft_trainer.train()

lora_model.save_pretrained("./fine-tuned")
final_model = AutoModelForSequenceClassification.from_pretrained("./fine-tuned")

final_trainer = Trainer(
	args=ft_training_args,
	compute_metrics=compute_metrics,
	model=final_model,
	eval_dataset=tokenized_test_dataset,
	train_dataset=tokenized_train_dataset,
)

final_evaluation = final_trainer.evaluate()
print("final evaluation:", final_evaluation)
