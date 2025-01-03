from transformers import BertTokenizer, GPT2ForSequenceClassification, TrainingArguments, Trainer, AutoModelForCausalLM, AutoModelForSequenceClassification
from datasets import load_dataset
import torch
from peft import LoraConfig, get_peft_model
from transformers import Trainer, EvalPrediction, GPT2Tokenizer, AutoTokenizer
import numpy as np


config = LoraConfig()

# fetch dataset 
# dataset = load_dataset("allenai/scifact_entailment")
# dataset = load_dataset("imdb")
# dataset = load_dataset('abullard1/steam-reviews-constructiveness-binary-label-annotations-1.5k', 'main_data')
dataset = load_dataset("amazon_polarity")

# Select train and test data 
train_dataset = dataset['train'].shuffle(seed=42).select(range(500))
# test_dataset = dataset['validation'].shuffle(seed=42).select(range(300))
test_dataset = dataset['test'].shuffle(seed=42).select(range(500))


#get tokenized claim, abstract and title
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
# tokenizer.pad_token = tokenizer.eos_token

def process_text(example):
    # return tokenizer(example["claim"], example["abstract"], truncation=True, padding="max_length", return_tensors="pt")
    return tokenizer(example["content"], truncation=True, padding="max_length")



# label_mapping = {"SUPPORT": 0, "CONTRADICT": 1, "NEI": 2}

# label_mapping = {"Negative": 0, "Positive": 1}

tokenized_train_dataset = train_dataset.map(process_text, batched=True)
tokenized_test_dataset = test_dataset.map(process_text, batched=True)


# def encode_labels(example):
#     example["labels"] = label_mapping[example["verdict"]]
#     return example

# tokenized_train_dataset = tokenized_train_dataset.map(encode_labels)
# tokenized_test_dataset = tokenized_test_dataset.map(encode_labels)

#model
model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', 
id2label={0: "NEGATIVE", 1: "POSITIVE"},
label2id={"NEGATIVE": 0, "POSITIVE": 1},
num_labels=2)

# model.config.pad_token_id = tokenizer.pad_token_id


training_args = TrainingArguments(
    output_dir="./results",
    per_device_eval_batch_size=4,
    per_device_train_batch_size=4,
    evaluation_strategy="epoch",
    num_train_epochs=1,
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
    per_device_eval_batch_size=4,
    per_device_train_batch_size=4,
    evaluation_strategy="epoch",
    num_train_epochs=4,
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

lora_model.save_pretrained("/final-save")
final_model = AutoModelForSequenceClassification.from_pretrained("/final-save")

final_trainer = Trainer(
	args=ft_training_args,
	compute_metrics=compute_metrics,
	model=final_model,
	eval_dataset=tokenized_test_dataset,
	train_dataset=tokenized_train_dataset,
)

final_evaluation = final_trainer.evaluate()
print("final evaluation:", final_evaluation)
