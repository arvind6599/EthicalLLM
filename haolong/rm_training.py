from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, AutoConfig, AutoModelForSequenceClassification
from datasets import load_dataset
from trl import ModelConfig, RewardConfig, RewardTrainer, get_kbit_device_map, get_peft_config, get_quantization_config
import torch
import os
import time
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict
)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
    # old code
    # model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", device_map=device_map)
    
    
    # new code: 直接加载用于序列分类的模型，配置自动匹配分类任务
    quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
    )
    
    config = AutoConfig.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", num_labels=2)
    model_for_classification = AutoModelForSequenceClassification.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.2", 
        config=config,
        quantization_config=quantization_config,
        device_map=device_map
    )
    
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", max_length=100, padding="max_length", truncation=True)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
        model_for_classification.resize_token_embeddings(len(tokenizer))
    # tokenizer.model_max_length = 100 #NOTE: remember to add truncation=True
    model_for_classification.config.pad_token_id = model_for_classification.config.eos_token_id
    # model_for_classification.to(device)
    # print("Model on device:", device)
    
    # lora config
    LORA_R = 8
    LORA_ALPHA = 32
    LORA_DROPOUT = 0.1
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=LORA_R, # LoRA中低秩近似的秩
        lora_alpha=LORA_ALPHA, # 见上文中的低秩矩阵缩放超参数
        lora_dropout=LORA_DROPOUT, # LoRA层的dropout
        bias='none'
    )
    
    
    print("Model loaded!")

    ##################
    # data
    ##################
    # NOTE: change this later
    dataset = load_dataset("Tachi67/rm_data_action")
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
            tokenized_chosen = tokenizer(chosen, truncation=True)
            tokenized_rejected = tokenizer(rejected, truncation=True)
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
    
    print("Data loaded!")
    
    ###############
    # train
    ###############
    cmd_args = [
    "--per_device_train_batch_size=16",
    # "--model_name_or_path=mistralai/Mistral-7B-Instruct-v0.2",
    "--output_dir=reward_modeling_action",
    "--num_train_epochs=3",
    "--gradient_accumulation_steps=16",
    "--gradient_checkpointing=True",
    "--learning_rate=1.41e-5",
    "--report_to=tensorboard",
    "--remove_unused_columns=False",
    "--optim=adamw_torch",
    "--logging_steps=10",
    "--evaluation_strategy=steps",
    "--max_length=100",
    "--save_total_limit=1",
    "--load_best_model_at_end=True",
    "--fp16=True",
    "--fp16_opt_level=O1",
    ]
    ################
    # Config parsing
    ################
    parser = HfArgumentParser((RewardConfig, ModelConfig))
    reward_config, model_config = parser.parse_args_into_dataclasses(args=cmd_args)
    reward_config.gradient_checkpointing_kwargs = dict(use_reentrant=False)

    # quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        # quantization_config=quantization_config,
    )
    # old code
    # # model conversion
    # config = AutoConfig.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", num_labels=2)
    # model_for_classification = AutoModelForSequenceClassification.from_config(config)
    # # copy the original weights from the original model to the new model, except the classification head
    # model_for_classification.base_model = model.base_model
    # model_for_classification.to(device)
    
    # train
    trainer = RewardTrainer(
        model=model_for_classification,
        tokenizer=tokenizer,
        args=reward_config,
        train_dataset=small_sample_train,
        eval_dataset=small_sample_test,
        peft_config=lora_config,
    )
    
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated(device=device)
    
    start = time.time()
    print("Training...")
    trainer.train()
    print("Completed!")
    print("Time taken for rm training:", time.time() - start)
    print("Pushing to hub...")
    trainer.model.push_to_hub("Tachi67/EthcalLLM-RM-action")
    print("Done!")

    