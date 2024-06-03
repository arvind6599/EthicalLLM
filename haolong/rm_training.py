from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, AutoConfig, AutoModelForSequenceClassification
from datasets import load_dataset
from trl import ModelConfig, RewardConfig, RewardTrainer, get_kbit_device_map, get_peft_config, get_quantization_config
import torch







if __name__ == "__main__":
    model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    ##################
    # data
    ##################
    # NOTE: change this later
    dataset = load_dataset("Anthropic/hh-rlhf", data_dir="harmless-base")
    small_sample_train = dataset['train'].shuffle(seed=42).select(range(20))
    small_sample_test = dataset['test'].shuffle(seed=42).select(range(10))
    
    def preprocess_function(examples, tokenizer=tokenizer):
        new_examples = {
            "input_ids_chosen": [],
            "attention_mask_chosen": [],
            "input_ids_rejected": [],
            "attention_mask_rejected": [],
        }
        for chosen, rejected in zip(examples["chosen"], examples["rejected"]):
            tokenized_chosen = tokenizer(chosen)
            tokenized_rejected = tokenizer(rejected)

            new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
            new_examples["attention_mask_chosen"].append(tokenized_chosen["attention_mask"])
            new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])
            new_examples["attention_mask_rejected"].append(tokenized_rejected["attention_mask"])

        return new_examples
    
    small_sample_train = small_sample_train.map(
    preprocess_function,
    batched=True,
    num_proc=4,
    )
    small_sample_test = small_sample_test.map(
        preprocess_function,
        batched=True,
        num_proc=4,
    )
    
    ###############
    # train
    ###############
    cmd_args = [
    "--per_device_train_batch_size=32",
    "--model_name_or_path=facebook/opt-350m",
    "--output_dir=reward_modeling_anthropic_hh",
    "--num_train_epochs=1",
    "--gradient_accumulation_steps=16",
    "--gradient_checkpointing=True",
    "--learning_rate=1.41e-5",
    "--report_to=wandb",
    "--remove_unused_columns=False",
    "--optim=adamw_torch",
    "--logging_steps=10",
    "--evaluation_strategy=steps",
    "--max_length=56",
    ]
    ################
    # Config parsing
    ################
    parser = HfArgumentParser((RewardConfig, ModelConfig))
    reward_config, model_config = parser.parse_args_into_dataclasses(args=cmd_args)
    reward_config.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    # print(reward_config.per_device_train_batch_size)

    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    
    # model conversion
    config = AutoConfig.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", num_labels=2)
    model_for_classification = AutoModelForSequenceClassification.from_config(config)

    # copy the original weights from the original model to the new model, except the classification head
    model_for_classification.base_model = model.base_model
    model_for_classification.to(device)
    
    # train
    trainer = RewardTrainer(
        model=model_for_classification,
        tokenizer=tokenizer,
        args=reward_config,
        train_dataset=small_sample_train,
        eval_dataset=small_sample_test,
        peft_config=get_peft_config(model_config),
    )
    trainer.train()
    
    