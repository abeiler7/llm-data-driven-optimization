import os
import argparse
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
    default_data_collator,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    GPTQConfig,
)
from transformers.trainer_utils import get_last_checkpoint
from datasets import load_from_disk
import torch
import shutil

import data_prep


import bitsandbytes as bnb
from huggingface_hub import login, HfFolder


def parse_arge():
    """Parse the arguments."""
    parser = argparse.ArgumentParser()
    # add model id and dataset path argument
    parser.add_argument(
        "--model_id",
        type=str,
        help="Model id to use for training.",
    )
    parser.add_argument(
        "--dataset", type=str, default=None, help="HuggingFace Dataset ID"
    )
    parser.add_argument(
        "--train_dataset_path", type=str, default=None, help="Path to training dataset."
    )
    parser.add_argument(
        "--val_dataset_path", type=str, default=None, help="Path to validation dataset."
    )
    parser.add_argument(
        "--output_data_path", type=str, default="lm_dataset", help="Path to store output data."
    )
    parser.add_argument(
        "--hf_token", type=str, default=HfFolder.get_token(), help="Hugging Face Token."
    )
    # add training hyperparameters for epochs, batch size, learning rate, and seed
    parser.add_argument(
        "--epochs", type=int, default=3, help="Number of epochs to train for."
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
        help="Batch size to use for training.",
    )
    parser.add_argument(
        "--lr", type=float, default=5e-5, help="Learning rate to use for training."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed to use for training."
    )
    parser.add_argument(
        "--gradient_checkpointing",
        type=bool,
        default=True,
        help="Path to deepspeed config file.",
    )
    parser.add_argument(
        "--bf16",
        type=bool,
        default=True if torch.cuda.get_device_capability()[0] == 8 else False,
        help="Whether to use bf16.",
    )
    parser.add_argument(
        "--merge_weights",
        type=bool,
        default=True,
        help="Whether to merge LoRA weights with base model.",
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=64,
        help="LoRA Config - lora_r Value",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=16,
        help="LoRA Config - lora_alpha Value",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.1,
        help="LoRA Config - lora_r Value",
    )
    
    # Push to Hub Parameters
    parser.add_argument("--push_to_hub", type=bool, default=True)
    parser.add_argument("--hub_model_id", type=str, default=None)
    parser.add_argument("--hub_strategy", type=str, default=None)
    parser.add_argument("--hub_token", type=str, default=None)

    args, _ = parser.parse_known_args()

    if args.hf_token:
        print(f"Logging into the Hugging Face Hub with token {args.hf_token[:10]}...")
        login(token=args.hf_token)

    return args


# COPIED FROM https://github.com/artidoro/qlora/blob/main/qlora.py
def print_trainable_parameters(model, use_4bit=False):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    if use_4bit:
        trainable_params /= 2
    print(
        f"all params: {all_param:,d} || trainable params: {trainable_params:,d} || trainable%: {100 * trainable_params / all_param}"
    )


# COPIED FROM https://github.com/artidoro/qlora/blob/main/qlora.py
def find_all_linear_names(model):
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, bnb.nn.Linear4bit):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


def create_peft_model(model, args, gradient_checkpointing=True, bf16=True):
    from peft import (
        get_peft_model,
        LoraConfig,
        TaskType,
        prepare_model_for_kbit_training,
    )
    from peft.tuners.lora import LoraLayer

    # prepare int-4 model for training
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=gradient_checkpointing
    )
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # get lora target modules
    # modules = find_all_linear_names(model)
    # print(f"Found {len(modules)} modules to quantize: {modules}")
    lora_modules = ["q_proj","v_proj","k_proj","o_proj",]

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=lora_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, peft_config)

    # pre-process the model by upcasting the layer norms in float 32 for
    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            if bf16:
                module = module.to(torch.bfloat16)
        if "norm" in name:
            module = module.to(torch.float32)
        if "lm_head" in name or "embed_tokens" in name:
            if hasattr(module, "weight"):
                if bf16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)

    model.print_trainable_parameters()
    return model

def print_files(path):
    # import python modules
    import os
    print(f"*** {path} ***")
    # Get list of all files only in the given directory
    fun = lambda x : os.path.isfile(os.path.join(path,x))
    files_list = filter(fun, os.listdir(path))
    
    # Create a list of files in directory along with the size
    size_of_file = [
        (f,os.stat(os.path.join(path, f)).st_size)
        for f in files_list
    ]
    # Iterate over list of files along with size
    # and print them one by one.
    for f,s in size_of_file:
        print("{} : {}MB".format(f, round(s/(1024*1024),3)))

def training_function(args):
    # set seed
    set_seed(args.seed)

    if args.dataset:
        train_dataset = data_prep.data_prep(args.dataset)
        # train_dataset = data["train"]
        # val_dataset = data["val"]
    else:
        train_dataset = load_from_disk(args.train_dataset_path)
        # val_dataset = load_from_disk(args.val_dataset_path)

    # val_set_size = len(val_dataset)
    # load model from the hub with a bnb config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        use_cache=False
        if args.gradient_checkpointing
        else True,  # this is needed for gradient checkpointing
        device_map="auto",
        quantization_config=bnb_config,
    )

    # create peft config
    model = create_peft_model(
        model, args, gradient_checkpointing=args.gradient_checkpointing, bf16=args.bf16
    )

    # Define training args
    output_dir = "/opt/ml/checkpoints"
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        bf16=args.bf16,  # Use BF16 if available
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        gradient_checkpointing=args.gradient_checkpointing,
        # adding evaluation to model
        # evaluation_strategy="steps" if val_set_size > 0 else "no",
        # eval_steps=200 if val_set_size > 0 else None,
        # logging strategies
        logging_dir=f"{args.output_data_path}/logs",
        logging_strategy="steps",
        logging_steps=10,
        save_strategy="no",
        # push to hub parameters
        push_to_hub=args.push_to_hub,
        hub_strategy=args.hub_strategy,
        hub_model_id=args.hub_model_id,
        hub_token=args.hub_token,
    )

    # Create Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        # eval_dataset=val_dataset,
        data_collator=default_data_collator,
    )

    if get_last_checkpoint(output_dir) is not None:
        print("***** continue training *****")
        last_checkpoint = get_last_checkpoint(output_dir)
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        trainer.train()
    
    # if val_set_size > 0:
    #     # evaluate model
    #     eval_result = trainer.evaluate(eval_dataset=val_dataset)

    #     # writes eval result to file which can be accessed later in s3 ouput
    #     with open(os.path.join(args.output_data_path, "eval_results.txt"), "w") as writer:
    #         print(f"***** Eval results *****")
    #         for key, value in sorted(eval_result.items()):
    #             writer.write(f"{key} = {value}\n")
    #             print(f"{key} = {value}\n")
    try:
        trainer.create_model_card(model_name=args.hub_model_id)
        trainer.push_to_hub()
    except Exception as e:
        print(f"Unable to Push to Hugging Face Hub. Err: {e}")

    sagemaker_save_dir="/opt/ml/model/"
    if args.merge_weights:
        print("***MERGING***")
        # merge adapter weights with base model and save
        # save int 4 model
        trainer.model.save_pretrained(output_dir, safe_serialization=False)
        # clear memory
        del model
        del trainer
        torch.cuda.empty_cache()

        from peft import AutoPeftModelForCausalLM

        # load PEFT model in fp16
        model = AutoPeftModelForCausalLM.from_pretrained(
            output_dir,
            low_cpu_mem_usage=True,
            device_map="auto", # Added from 28_train_llms_with_qlora example
            torch_dtype=torch.float16,
            trust_remote_code=True, # Added from 28_train_llms_with_qlora example
        )  
        # Merge LoRA and base model and save
        model = model.merge_and_unload()        
        model.save_pretrained(
            sagemaker_save_dir, safe_serialization=True, max_shard_size="2GB"
        )

        try:
            print("***Pushing Model to Hub***")
            model.push_to_hub(repo_id=args.hub_model_id)
        except Exception as e:
            print(f"Unable to Push to Hugging Face Hub. Err: {e}")
    else:
        print("NOT MERGING")
        trainer.model.save_pretrained(
            sagemaker_save_dir, safe_serialization=True
        )

    # save tokenizer for easy inference
    print("***Retreiving & Saving Tokenizer***")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.save_pretrained(sagemaker_save_dir)

    try:
        print("***Pushing Tokenizer to Hub***")
        tokenizer.push_to_hub(repo_id=args.hub_model_id)
    except Exception as e:
        print(f"Unable to Push to Hugging Face Hub. Err: {e}")

    # # copy inference script
    # os.makedirs("/opt/ml/model/code", exist_ok=True)
    # shutil.copyfile(
    #     os.path.join(os.path.dirname(__file__), "inference.py"),
    #     "/opt/ml/model/code/inference.py",
    # )
    # shutil.copyfile(
    #     os.path.join(os.path.dirname(__file__), "requirements.txt"),
    #     "/opt/ml/model/code/requirements.txt",
    # )

    print_files("/opt/ml/model")
    # print_files("/opt/ml/checkpoints")
    print_files("/opt/ml/output")

    # # import os
    import glob

    files = glob.glob('/opt/ml/model/*')
    for f in files:
        os.remove(f)
    # files = glob.glob('/opt/ml/checkpoints/*')
    # print("Copying Files")
    # for f in files:
    #     # os.remove(f)
    #     shutil.copyfile(
    #         os.path.join('/opt/ml/checkpoints/', os.path.basename(f).split('/')[-1]),
    #         os.path.join('/opt/ml/model/', os.path.basename(f).split('/')[-1])
    #     )
    
def main():
    args = parse_arge()
    try:
        training_function(args)
    except Exception as e:
        print(f"***training_function() FAILED - Removing Model***")
        print(f"Error: {e}")

        import glob
        files = glob.glob('/opt/ml/model/*')
        for f in files:
            os.remove(f)
        
        raise e


if __name__ == "__main__":
    main()
    
