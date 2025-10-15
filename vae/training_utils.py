from transformers import Trainer, TrainerCallback
import pathlib
import os
import torch
import random
import gc

import math
import wandb
from peft import (
    LoraConfig,
)
from tqdm import tqdm
from model_vae import VAE

import torch.distributed as dist

local_rank = int(os.environ["LOCAL_RANK"])
device = torch.device(local_rank)

def run_inference(model, lines):
    model.eval()
    outputs = []
    print("Running inference")
    with torch.no_grad():
        for line in tqdm(lines):
            #print("=========================== START ============================")
            #print("Current input: ", line.split("\n")[-3:])
            # Tokenize input text
            tokenized_text = model.tokenizer(line, truncation=True,
                                          max_length=5120, padding=False,
                                          return_attention_mask=False)
            # Generate compressed outputs
            input_ids = torch.LongTensor([tokenized_text['input_ids']]).to(device)
            #print("input_ids shape: ", input_ids.size())
            memory_slots = model._compress(input_ids).to(device, torch.bfloat16)
            print("memory_slots shape: ", memory_slots.shape)

            decompress_memory_slots = model.decompress_layer(memory_slots)

            # prompt_output = model.tokenizer(data['prompt'], add_special_tokens=False, padding=False)
            prompt_ids = torch.LongTensor([[model.ae_token_id]]).to(device)
            #print("prompt_ids shape: ", prompt_ids.size())

            prompt_answer_embs = model.tokens_to_embeddings(prompt_ids)
            #print("prompt_answer_embs shape: ", prompt_answer_embs.size())
            #print("prompt_answer_embs shape: ", prompt_answer_embs.shape)

            memory_slots = decompress_memory_slots.to(prompt_answer_embs)
                        
            # Concatenate and clone input embeddings
            decoder_input_embeddings = torch.cat((memory_slots.unsqueeze(0), prompt_answer_embs), dim=1)
            #print("decoder_input_embeddings shape: ", decoder_input_embeddings.size())

            output = decoder_input_embeddings.clone()
            #print("output shape: ", output.size())

            generate_text = []
            past_key_values = None

            # Generate text output
            for i in range(100):
                with model.icae.disable_adapter():   # no independent decoder; use self.icae
                    out = model.icae(inputs_embeds=output, past_key_values=past_key_values, use_cache=True)
                logit = out.logits[:, -1, :model.vocab_size-1]
                past_key_values = out.past_key_values

                next_token_id = torch.argmax(logit, dim=-1)
                # print(next_token_id)
                
                if next_token_id.item() == 2:   # eos
                    break

                output = model.icae.get_base_model().model.embed_tokens(next_token_id).unsqueeze(1).to(device)
                generate_text.append(next_token_id.item())

            generated_text = model.tokenizer.decode(generate_text, skip_special_tokens=True)
            outputs.append(generated_text)

            #print("generated_text:", generated_text)

            #print("=========================== END ============================")

    return outputs

class InferenceCallback(TrainerCallback):
    def __init__(self, lines, interval=2500):
        self.lines = lines
        #self.run = run
        self.interval = interval

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.interval == 0:
            print(f"Running custom inference at step {state.global_step}")
            outputs = run_inference(kwargs['model'], self.lines)
            my_outputs = []
            correct = 0
            train_correct = 0
            idx = 0
            for i, j in zip(self.lines, outputs):
                print("=========================================================================")
                print("GT:", i.split("\n")[-3:])
                print(state.global_step, "generated_output:", j)
                print("=========================================================================")
                my_outputs.append([i, j])

                gt_string = "\n".join(i.split("\n")[-3:])
                if (gt_string == j):
                    correct += 1
                    if (idx < 200):
                        train_correct += 1
                
                idx += 1

            acc = correct / len(outputs)
            train_acc = train_correct / 200
    
            my_table = wandb.Table(
                columns=["input", "output"],
                data=my_outputs
            )

            if not dist.is_initialized() or dist.get_rank() == 0:
                # Before saving, gather the full parameters:
                with kwargs['model'].summon_full_params(recurse=True):
                    full_state_dict = kwargs['model'].state_dict()
                    ckpt_path = f"{args.output_dir}/{args.run_name}_ckpt_{state.global_step}.pth"
                    torch.save(full_state_dict, ckpt_path)

            wandb.log(
                {
                    f"infer_table_step_{state.global_step}": my_table,
                    "accuracy": acc,
                    "train_accuracy": train_acc
                }
            )
        return control

def train_model(args, notes, model, train_dataset, eval_dataset, model_args, training_args, lines, data_collator=None):
    if max(training_args.per_device_train_batch_size, training_args.per_device_eval_batch_size) == 1:
        data_collator = None
        
    # print training_args at local_rank 0
    local_rank = int(os.getenv('LOCAL_RANK', '0'))
    if local_rank == 0:
        print(training_args)
    
    # run = wandb.init(
    #     project="icae_game24",
    #     tags=["new"],
    #     config={
    #         "model_args": vars(model_args),
    #         "training_args": vars(args)
    #     },
    #     name=training_args.run_name,
    #     notes=notes
    # )

    print("run_name:", training_args.run_name)
    training_args.output_dir = os.path.join(training_args.output_dir, training_args.run_name)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        #callbacks=[InferenceCallback(lines, interval=args.eval_interval)]
    )

    # checkpoint = "./ckpts/cd_500k_dim16/checkpoint-6000"
    
    # print(f"Loaded from the checkpoint: {checkpoint}")
    if list(pathlib.Path(training_args.output_dir).glob('checkpoint*')):
        print('Resume from last checkpoint')
        trainer.train(resume_from_checkpoint=True)
    else:
        print('Training from scratch!')
        trainer.train()
    # # trainer.save_model()
    # trainer.log_metrics("train", train_result.metrics)
    # metrics = trainer.evaluate()
    # trainer.log_metrics("eval", metrics)
    # trainer.save_metrics("eval", metrics)

    torch.save(trainer.model.state_dict(), f"{training_args.output_dir}/{training_args.run_name}/model_weights_final.pth")

    #### 🚀 **Free Up GPU Memory BEFORE Re-Loading Model** ####
    print("🚀 Clearing VRAM before inference...")

    # Move model to CPU first (prevents GPU tensors lingering)
    model.to("cpu")
    del model  # Delete model object

    # Delete Trainer & Free Memory
    del trainer
    gc.collect()  # Garbage collection
    torch.cuda.empty_cache()  # Clear VRAM

    #### 🚀 **Now Reload the Model for Inference** ####
    print("🚀 Re-loading model for inference...")

    # EVALUATION
    lora_config = LoraConfig(
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = VAE(model_args, training_args, lora_config)
    print(f"Loading trained checkpoint from {training_args.output_dir}")
    model.load_state_dict(torch.load(f"{training_args.output_dir}/{training_args.run_name}/model_weights_final.pth"), strict=False)
    model = model.to(device)

    outputs = run_inference(model, lines)

    my_outputs = []
    correct = 0
    train_correct = 0
    idx = 0
    for i, j in zip(lines, outputs):
        print("=========================================================================")
        print("GT:\n", i)
        print("final", "generated_output:", j)
        print("=========================================================================")
        my_outputs.append([i, j])

        if (i == j):
            correct += 1
            if (idx < 200):
                train_correct += 1
        
        idx += 1

    acc = correct / len(outputs)
    train_acc = train_correct / 200

    my_table = wandb.Table(
        columns=["input", "output"],
        data=my_outputs
    )
    run.log(
        {
            "final": my_table,
            "accuracy": acc,
            "train_accuracy": train_acc
        }
    )
    
    run.finish()

def text_extraction(input_ids, length, lm_ratio=0.0):
    
    input_len = len(input_ids)
    assert input_len >= 1, f"Error: invalid input length ({input_len})"
    
    # ae
    if random.random() >= lm_ratio: 
        if input_len <= length: # if shorter, keep the complete text
            return input_ids, []
        else:
            last_start = input_len - length
            random_start = random.randint(0, last_start)
            return input_ids[random_start: random_start+length], []
    
    # lm    
    # What happens when either a or b (in first if case) is very small? If b is very small, it'll be an AE task. If a is small, still lm task.
    # What happens in the second else case if b is very small? It's treated as an AE task.
    if input_len <= length:
        r = random.randint(0, input_len-1)
        return input_ids[:r+1], input_ids[r+1:]
    else:
        last_start = input_len - length
        random_start = random.randint(0, last_start)
        return input_ids[random_start: random_start+length], input_ids[random_start+length:]

def _compute_num_segments(total_length, mem_size, mean_compression_rate):
    assert total_length > 0
    num_segments = math.ceil(total_length / (mem_size * mean_compression_rate))  # 128 * 4 -> 1 * (128*4)
    return num_segments

def pretrain_tokenize_function(
    examples, 
    tokenizer,
    model_max_length,
    mem_size,
    min_tokens_for_lm,
    mean_compression_rate,
    add_special_token_for_lm,
    leave_tokens_for_lm,
    ae_token_id,
    eos_id,
    #lm_token_id,
    mem, 
    input_type, 
    lm_ratio=0.0
):
    mid_point = len(examples[input_type])
    all_texts = examples[input_type] + examples["chain_of_thought"]

    # Batch tokenize together (single call)
    tokenized_outputs = tokenizer(
        all_texts, 
        truncation=False, 
        padding=False, 
        return_attention_mask=False
    )

    # Split back into input & CoT tokenized outputs
    text_output = {key: val[:mid_point] for key, val in tokenized_outputs.items()}  # First half
    reasoning_trace_output = {key: val[mid_point:] for key, val in tokenized_outputs.items()}  # Second half

    text_output['prompt_answer_ids'] = []
    text_output['labels'] = []
    
    max_len = model_max_length  # heuristic

    # Each data point in the batch can be AE or LM.
    for idx in range(len(text_output["input_ids"])):        
        ae = True
        a, b = text_extraction(text_output["input_ids"][idx], max_len, lm_ratio=lm_ratio)
        length_a = len(a)
        num_segments = _compute_num_segments(length_a, mem_size, mean_compression_rate)
        total_mem_length = num_segments * mem_size

        # Make sure that it is lm task iff it has at least min tokens for lm (64).
        if len(b) > min_tokens_for_lm:  # avoid too few tokens for lm, which is a waste of computing
            ae = False
            b = b[:max_len]

        text_output['input_ids'][idx] = a

        # decoder part: note that in v2, we add mem_tokens to the prompt_ids for easy implementation; which is different from v1 implementation where mem tokens are not in the prompt_ids
        if ae:  # autoencoding objective
            # Why is it mem[0]? Filler memory token, will be overwritten during forward pass.
            prompt_ids = mem + [ae_token_id]
            answer_ids = reasoning_trace_output['input_ids'][idx] + [eos_id]    # if ae, eos token
        else:   # lm objective
            print("No need to use LM objective!!!")
            # prompt_ids = [mem[0]] * total_mem_length
            # if add_special_token_for_lm:
            #     prompt_ids += [lm_token_id]
            # answer_ids = b   # if lm, no eos token

        text_output['prompt_answer_ids'].append(prompt_ids + answer_ids)
        if ae:
            labels = [-100] * len(prompt_ids) + answer_ids
        else:
            labels = [-100] * len(prompt_ids) + [-100] * leave_tokens_for_lm + answer_ids[leave_tokens_for_lm:] # no loss for leave_tokens_for_lm
        text_output['labels'].append(labels)
        assert len(text_output['prompt_answer_ids'][-1]) == len(labels)
        
    return text_output


class DataCollatorForDynamicPadding:
    def __init__(self, pad_token_id, pad_to_multiple_of=None):
        self.pad_token_id = pad_token_id
        self.pad_to_multiple_of = pad_to_multiple_of
    def __call__(self, examples):
        input_ids = [torch.tensor(example["input_ids"], dtype=torch.long) for example in examples]
        labels = [torch.tensor(example["labels"], dtype=torch.long) for example in examples]
        prompt_answer_ids = [torch.tensor(example["prompt_answer_ids"], dtype=torch.long) for example in examples]
        input_ids = self.dynamic_padding(input_ids, fill_value=self.pad_token_id)
        prompt_answer_ids = self.dynamic_padding(prompt_answer_ids, fill_value=self.pad_token_id)
        labels = self.dynamic_padding(labels)
        batch = {"input_ids": input_ids, "labels": labels, "prompt_answer_ids": prompt_answer_ids}
        return batch
    def dynamic_padding(self, sequences, fill_value=-100):
        max_length = max(len(x) for x in sequences)
        if self.pad_to_multiple_of:
            max_length = ((max_length - 1) // self.pad_to_multiple_of + 1) * self.pad_to_multiple_of
        padded_sequences = torch.full((len(sequences), max_length), fill_value, dtype=torch.long)
        for i, seq in enumerate(sequences):
            padded_sequences[i, :len(seq)] = seq
        return padded_sequences