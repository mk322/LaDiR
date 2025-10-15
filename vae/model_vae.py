"""
VAE that supports multi span concat
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn as nn
from typing import Optional
from peft import get_peft_model
import math
from safetensors.torch import load_file

    
def print_trainable_parameters(model):
    trainable_parameters = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_parameters += param.numel()
    print(f"trainable params: {trainable_parameters} || all params: {all_param} || trainable%: {100 * trainable_parameters / all_param}")


def freeze_model(model):
    for _, param in model.named_parameters():
        param.requires_grad = False


class VAE(torch.nn.Module):
    """
    Variational Autoencoder for text compression and reconstruction.
    
    This VAE model:
    - Compresses text sequences into latent representations
    - Supports multi-segment processing for long sequences
    - Uses LoRA for parameter-efficient fine-tuning
    - Includes KL divergence regularization
    
    Args:
        model_args: Model configuration arguments
        training_args: Training configuration arguments  
        lora_config: LoRA configuration for fine-tuning
    """
    
    def __init__(self, model_args, training_args, lora_config):
        super().__init__()
        self.model_args = model_args
        self.training_args = training_args
        self.model_name = model_args.model_name_or_path
        self.icae = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.bfloat16)
        
        self.training = self.model_args.train    
        
        if self.training:    # independent model for gradient checkpointing
            self.decoder = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.bfloat16)

        self.vocab_size = self.icae.config.vocab_size + 1    # [PAD] token
        self.pad_token_id = self.vocab_size - 1
        self.mean_compression_rate = training_args.mean_compression_rate

        self.dim = 128

        self.beta = model_args.beta

        self.mean = nn.Linear(in_features=self.icae.config.hidden_size, out_features=self.dim, dtype=torch.bfloat16)
        self.log_var = nn.Linear(in_features=self.icae.config.hidden_size, out_features=self.dim, dtype=torch.bfloat16)

        self.decompress_layer = nn.Linear(in_features=self.dim, out_features=self.icae.config.hidden_size, dtype=torch.bfloat16)
        
        # tunable
        self.mem_size = self.training_args.fixed_mem_size
        self.vocab_size_with_mem = self.vocab_size + self.mem_size # so, the mem tokens are in the range [self.vocab_size, self.vocab_size + self.mem_size)

        # special tokens in addition to mem and length tokens
        self.ae_token_id = self.vocab_size_with_mem + 0       

        self.icae.resize_token_embeddings(self.vocab_size_with_mem + 1) 
        
        # special tokens for Llama-2/Mistral tokenizer
        self.bos_id = 1
        self.eos_id = 2
        
        #self.dim = self.icae.config.hidden_size
        self.icae = get_peft_model(self.icae, lora_config)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.memory_token_embed = nn.Embedding(self.mem_size + 1, self.dim, padding_idx=None)
        #self.ae_token_embed = nn.Embedding(1, self.dim, padding_idx=None)

        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)
        self.append_sequence = torch.arange(self.vocab_size, self.vocab_size + self.mem_size, dtype=torch.long, device=self.device).unsqueeze(0)   # mem tokens
        
        if self.training:
            self.init()

    def encoder_mean(self, input_ids_enc, *kargs):
        hidden_state = self.encoder(input_ids_enc, *kargs)
        shapes = hidden_state.shape
        mean = self.fc_mean(hidden_state).view(-1, shapes[2], shapes[1])
        if self.h_tanh:
            mean = torch.tanh(mean)
        return mean

    def init(self):
        print("Freezing the decoder...")
        if self.training:
            freeze_model(self.decoder)
            self.decoder.eval()
            #self.decoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        print_trainable_parameters(self)
        if self.training_args.restore_from is not None and self.training_args.restore_from != "":
            print(f"Loading from the pretrained checkpoint: {self.training_args.restore_from}...")
            state_dict = load_file(self.training_args.restore_from)
            self.load_state_dict(state_dict)
            print(f"Finished loading from {self.training_args.restore_from}")
        #print("Enabling gradient checkpointing...")
        # self.icae.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                
        
    def compute_num_segments(self, total_length):
        assert total_length > 0
        num_segments = math.ceil(total_length / (self.mem_size * self.mean_compression_rate))  # 128 * 4 -> 1 * (128*4)
        return num_segments

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        prompt_answer_ids: torch.LongTensor = None,
        labels: Optional[torch.LongTensor] = None,
    ):
        # encoder part
        #print("input_ids shape: ", input_ids.size())
        #print("prompt_answer_ids shape: ", prompt_answer_ids.size())
        #print("labels shape: ", labels.size())
        
        batch_size = input_ids.size(0)
        total_length = input_ids.size(1)
        prompt_answer_ids = prompt_answer_ids.reshape(batch_size, -1)
        num_segments = self.compute_num_segments(total_length)
        segment_length = math.ceil(total_length / num_segments)

        prompt_answer_embs = self.icae.get_base_model().model.embed_tokens(prompt_answer_ids)

        max_compressed_length = num_segments * self.mem_size
        #print("max_compressed_length: ", max_compressed_length)
        
        compress_outputs = torch.zeros((batch_size, max_compressed_length, self.dim)).to(prompt_answer_embs)
        #print("compress_outputs shape: ", compress_outputs.size())
        
        total_kl_loss = torch.zeros(1).to(prompt_answer_embs)

        for segment_idx in range(num_segments):
           # print(f"===============Segment {segment_idx}=======================")
            
            start_idx = segment_idx * segment_length
            end_idx = min((segment_idx + 1) * segment_length, total_length)
            #print(f"start_idx: {start_idx} | end_idx: {end_idx}")
            segment_input_ids = input_ids[:, start_idx:end_idx]

            segment_input_ids = torch.cat([segment_input_ids, self.append_sequence.repeat(batch_size, 1)], dim=1)
            #print("segment_input_ids shape after concat: ", segment_input_ids.size())
            mem_flag = segment_input_ids >= self.vocab_size
            #print("mem_flag shape: ", mem_flag.size())

            segment_input_embedding = self.icae.get_base_model().model.embed_tokens(segment_input_ids)
            #print("segment_input_embedding shape: ", segment_input_embedding.size())
            segment_input_embedding[mem_flag] = self.decompress_layer(self.memory_token_embed(segment_input_ids[mem_flag] - self.vocab_size).to(segment_input_embedding))
            #print("Populated segment_input_embedding memory tokens")
            
            # compress the current segment
            segment_compress_outputs = self.icae(inputs_embeds=segment_input_embedding, output_hidden_states=True)
            segment_compress_outputs = segment_compress_outputs.hidden_states[-1]
            batch_memory_tokens = segment_compress_outputs[mem_flag].view(batch_size, self.mem_size, -1)
            mean_vae = self.mean(batch_memory_tokens)
            log_var_vae  = self.log_var(batch_memory_tokens)

            batch_memory_tokens = self.reparameterize(mean_vae, log_var_vae)

            if self.training:
                batch_memory_tokens += torch.randn_like(batch_memory_tokens) * 0.3

            kl_segment = -0.5 * torch.mean(
                1 + log_var_vae - mean_vae.pow(2) - log_var_vae.exp()
            )
            total_kl_loss += kl_segment

            # collect memory tokens
            compress_outputs[:, segment_idx*self.mem_size: self.mem_size*(segment_idx+1)] = batch_memory_tokens
            
            del segment_input_ids, segment_input_embedding
            torch.cuda.empty_cache()

            #print(f"===============Segment {segment_idx} END=======================")
        
        # decoder part
        kl_loss = total_kl_loss / num_segments

        decoder_mem_flag = (prompt_answer_ids >= self.vocab_size) & (prompt_answer_ids < self.vocab_size + self.mem_size)   # only mem tokens
        #print("decoder_mem_flag shape: ", decoder_mem_flag.size())

        decompress_outputs = self.decompress_layer(compress_outputs)
        
        prompt_answer_embs[decoder_mem_flag] = decompress_outputs.view(-1, decompress_outputs.size(-1))

        special_prompt = prompt_answer_ids >= self.vocab_size_with_mem

        prompt_answer_embs[special_prompt] = self.decompress_layer(self.memory_token_embed(prompt_answer_ids[special_prompt] - self.vocab_size)).view(-1, decompress_outputs.size(-1)).to(prompt_answer_embs)    # replace special token's embedding from self.memory_token_embed
        
        if self.training:   # has an independent self.decoder
            decoder_outputs = self.decoder(inputs_embeds=prompt_answer_embs, output_hidden_states=True)
        else:
            with self.icae.disable_adapter():   # no independent decoder; use self.icae
                decoder_outputs = self.icae(inputs_embeds=prompt_answer_embs, output_hidden_states=True)

        logits = decoder_outputs.logits
        #print("decoder_outputs logits shape: ", logits.size())

        effective_logits = logits[:,:-1,:].reshape(-1, logits.size(-1))  # Why are we skipping the last generated logit? It's probably the eos token.
        #print("effective_logits shape: ", effective_logits.size())
        target_ids = labels[:,1:].reshape(-1)  # Why does it take from the first index onwards?
        #print("target_ids shape: ", target_ids.size())

        ce_loss = self.loss_fct(effective_logits, target_ids)
        loss = ce_loss + self.beta * kl_loss
        return {"loss": loss, "logits": logits, "kl_loss": kl_loss, "ce_loss": ce_loss}

    def decoder_loss(self, memory_slots, prompt_answer_ids, labels):
        batch_size = memory_slots.size(0)
                    
        max_compressed_length = self.mem_size
        prompt_answer_embs = self.icae.get_base_model().model.embed_tokens(prompt_answer_ids)
        compress_outputs = torch.zeros((batch_size, max_compressed_length, self.dim)).to(prompt_answer_embs)

        # Convert memory_slots to the same dtype as prompt_answer_embs
        memory_slots = memory_slots.to(dtype=prompt_answer_embs.dtype)
        compress_outputs[:, :self.mem_size] = memory_slots

        decoder_mem_flag = (prompt_answer_ids >= self.vocab_size) & (prompt_answer_ids < self.vocab_size + self.mem_size)   # only mem tokens

        # Decompress to model's hidden size
        decompress_outputs = self.decompress_layer(compress_outputs)
        
        # Ensure the embeddings are in the correct shape for the model
        prompt_answer_embs = prompt_answer_embs.to(dtype=decompress_outputs.dtype)
        prompt_answer_embs[decoder_mem_flag] = decompress_outputs.view(-1, decompress_outputs.size(-1))

        special_prompt = prompt_answer_ids >= self.vocab_size_with_mem

        #print("self.memory_token_embed(prompt_answer_ids[special_prompt] - self.vocab_size).dtype", self.memory_token_embed(prompt_answer_ids[special_prompt] - self.vocab_size).dtype)

        prompt_answer_embs[special_prompt] = self.decompress_layer(self.memory_token_embed(prompt_answer_ids[special_prompt] - self.vocab_size).to(prompt_answer_embs.dtype)).view(-1, decompress_outputs.size(-1))    # replace special token's embedding from self.memory_token_embed
        prompt_answer_embs = prompt_answer_embs.reshape(batch_size, -1, decompress_outputs.size(-1))

        print("prompt_answer_embs.shape", prompt_answer_embs.shape)

        if self.training:   # has an independent self.decoder
            decoder_outputs = self.decoder(inputs_embeds=prompt_answer_embs, output_hidden_states=True)
        else:
            with self.icae.disable_adapter():   # no independent decoder; use self.icae
                decoder_outputs = self.icae(inputs_embeds=prompt_answer_embs, output_hidden_states=True)

        logits = decoder_outputs.logits

        effective_logits = logits[:,:-1,:].reshape(-1, logits.size(-1))  
        target_ids = labels[:,1:].reshape(-1)  

        ce_loss = self.loss_fct(effective_logits, target_ids)

        return {"loss": ce_loss}

    def tokens_to_embeddings(self, token_ids):   # input_tokens can be either normal tokens and special tokens
        embeddings = self.icae.get_base_model().model.embed_tokens(token_ids)
        special_flags = token_ids >= self.vocab_size
        embeddings[special_flags] = self.decompress_layer(self.memory_token_embed(token_ids[special_flags] - self.vocab_size).to(embeddings))    # replace special token's embedding from self.memory_token_embed
        return embeddings

    def _compress(
        self,
        input_ids: torch.LongTensor = None,
        return_sample = None
    ):  # for inference; compress a fixed length of input into memory slots

        batch_size = input_ids.size(0)
        total_length = input_ids.size(1)
        num_segments = self.compute_num_segments(total_length)
        segment_length = math.ceil(total_length / num_segments)
        
        max_compressed_length = num_segments * self.mem_size
        compress_outputs = torch.zeros((batch_size, max_compressed_length, self.dim))
        
        for segment_idx in range(num_segments):
            start_idx = segment_idx * segment_length
            end_idx = min((segment_idx + 1) * segment_length, total_length)
            segment_input_ids = input_ids[:, start_idx:end_idx]
            segment_input_ids = torch.cat([segment_input_ids, self.append_sequence.repeat(batch_size, 1)], dim=1)
            mem_flag = segment_input_ids >= self.vocab_size

            segment_input_embedding = self.icae.get_base_model().model.embed_tokens(segment_input_ids)
            segment_input_embedding[mem_flag] = self.decompress_layer(self.memory_token_embed(segment_input_ids[mem_flag] - self.vocab_size).to(segment_input_embedding))

            # compress the current segment
            segment_compress_outputs = self.icae(inputs_embeds=segment_input_embedding, output_hidden_states=True)
            segment_compress_outputs = segment_compress_outputs.hidden_states[-1]
            batch_memory_tokens = segment_compress_outputs[mem_flag].view(batch_size, self.mem_size, -1)

            if return_sample == "parameters":
                mean_vae = self.mean(batch_memory_tokens)
                log_var_vae  = self.log_var(batch_memory_tokens)
                compress_outputs = (mean_vae, log_var_vae)

            elif return_sample == "sample":
                log_var_vae = self.log_var(batch_memory_tokens)
                mean_vae = self.mean(batch_memory_tokens)
                batch_memory_tokens = self.reparameterize(mean_vae, log_var_vae) 
                batch_memory_tokens += torch.randn_like(batch_memory_tokens) * 0.3
                compress_outputs[:, segment_idx*self.mem_size: self.mem_size*(segment_idx+1)] = batch_memory_tokens

            else: 
                batch_memory_tokens = self.mean(batch_memory_tokens)
                # collect memory tokens
                compress_outputs[:, segment_idx*self.mem_size: self.mem_size*(segment_idx+1)] = batch_memory_tokens

            del segment_input_ids, segment_input_embedding
            torch.cuda.empty_cache()

        return compress_outputs
    
    def run_inference(self, text: str):
        self.eval()
        with torch.no_grad():
            tokenized_text = self.tokenizer(text, truncation=True,
                                          max_length=5120, padding=False,
                                          return_attention_mask=False)
            # Generate compressed outputs
            input_ids = torch.LongTensor([tokenized_text['input_ids']]).to(self.device)
            memory_slots = self._compress(input_ids)
            prompt_ids = torch.LongTensor([[self.ae_token_id]]).to(self.device)

            prompt_answer_embs = self.tokens_to_embeddings(prompt_ids)
            memory_slots = memory_slots.to(self.device, prompt_answer_embs)
            
            decompress_memory_slots = self.decompress_layer(memory_slots)
            decoder_input_embeddings = torch.cat((decompress_memory_slots.unsqueeze(0), prompt_answer_embs), dim=1)
            output = decoder_input_embeddings.clone()

            generate_text = []
            past_key_values = None

            # Generate text output
            for i in range(100):
                with self.icae.disable_adapter():   # no independent decoder; use self.icae
                    out = self.icae(inputs_embeds=output, past_key_values=past_key_values, use_cache=True)
                logit = out.logits[:, -1, :self.vocab_size-1]
                past_key_values = out.past_key_values

                next_token_id = torch.argmax(logit, dim=-1)
                # print(next_token_id)
                
                if next_token_id.item() == 2:   # eos
                    break

                output = self.icae.get_base_model().model.embed_tokens(next_token_id).unsqueeze(1).to(self.device)
                generate_text.append(next_token_id.item())

            generated_text = self.tokenizer.decode(generate_text)

        return generated_text

    def encode_text(self, text, return_sample=""):
        self.eval()
        with torch.no_grad():
            tokenized_text = self.tokenizer(text, truncation=True,
                                          max_length=5120, padding=False,
                                          return_attention_mask=False)
            # Generate compressed outputs
            input_ids = torch.LongTensor([tokenized_text['input_ids']]).to(self.device)
            memory_slots = self._compress(input_ids, return_sample=return_sample)
        return memory_slots
    
    def encode_batch_text(self, text_list, return_sample=""):
        self.eval()
        with torch.no_grad():
            tokenized_text = self.tokenizer(text_list, truncation=True,
                                          max_length=5120, padding=True,
                                          return_attention_mask=False)
            # Generate compressed outputs
            input_ids = torch.LongTensor(tokenized_text['input_ids']).to(self.device)
            memory_slots = self._compress(input_ids, return_sample=return_sample)
        return memory_slots
    
    def decode_text(self, memory_slots):
        with torch.no_grad():
            prompt_ids = torch.LongTensor([[self.ae_token_id]]).to(self.device).repeat(memory_slots.shape[0], 1)

            prompt_answer_embs = self.tokens_to_embeddings(prompt_ids)

            memory_slots = memory_slots.to(device=self.device, dtype=prompt_answer_embs.dtype)
                        
            # Concatenate and clone input embeddings
            decompress_memory_slots = self.decompress_layer(memory_slots)
            decoder_input_embeddings = torch.cat((decompress_memory_slots, prompt_answer_embs), dim=1)

            output = decoder_input_embeddings.clone()

            generate_text = []
            past_key_values = None

            # Generate text output
            for i in range(100):
                with self.icae.disable_adapter():   # no independent decoder; use self.icae
                    out = self.icae(inputs_embeds=output, past_key_values=past_key_values, use_cache=True)
                logit = out.logits[:, -1, :self.vocab_size-1]
                past_key_values = out.past_key_values

                next_token_id = torch.argmax(logit, dim=-1)
                
                if next_token_id.item() == 2:   # eos
                    break

                output = self.icae.get_base_model().model.embed_tokens(next_token_id).unsqueeze(1).to(self.device)
                generate_text.append(next_token_id.item())

            generated_text = self.tokenizer.decode(generate_text, skip_special_tokens=True)

        return generated_text

    def decode_text_batch(self, memory_slots):
        """
        Decode multiple memory slots in parallel to generate text for each slot.
        
        Args:
            memory_slots: Tensor of shape (batch_size, memory_size, hidden_dim)
            
        Returns:
            List of generated texts, one for each memory slot in the batch
        """
        with torch.no_grad():
            batch_size = memory_slots.shape[0]
            prompt_ids = torch.LongTensor([[self.ae_token_id]]).to(self.device).repeat(batch_size, 1)

            prompt_answer_embs = self.tokens_to_embeddings(prompt_ids)

            memory_slots = memory_slots.to(device=self.device, dtype=prompt_answer_embs.dtype)
                        
            # Concatenate and clone input embeddings
            decompress_memory_slots = self.decompress_layer(memory_slots)
            decoder_input_embeddings = torch.cat((decompress_memory_slots, prompt_answer_embs), dim=1)

            output = decoder_input_embeddings.clone()

            # Initialize lists to store generated tokens for each sequence in batch
            generate_text = [[] for _ in range(batch_size)]
            # Track which sequences have finished generating
            finished_sequences = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
            past_key_values = None

            # Generate text output
            for i in range(100):
                with self.icae.disable_adapter():   # no independent decoder; use self.icae
                    out = self.icae(inputs_embeds=output, past_key_values=past_key_values, use_cache=True)
                logit = out.logits[:, -1, :self.vocab_size-1]
                past_key_values = out.past_key_values

                next_token_ids = torch.argmax(logit, dim=-1)
                
                # Update finished sequences
                finished_sequences = finished_sequences | (next_token_ids == 2)  # 2 is EOS token
                
                # If all sequences are finished, break
                if finished_sequences.all():
                    break

                # Only process tokens for sequences that haven't finished
                active_sequences = ~finished_sequences
                if active_sequences.any():
                    # Get embeddings for next tokens
                    next_token_embeddings = self.icae.get_base_model().model.embed_tokens(next_token_ids).unsqueeze(1).to(self.device)
                    
                    # Update output only for active sequences
                    output = next_token_embeddings
                    
                    # Store generated tokens for active sequences
                    for idx in range(batch_size):
                        if active_sequences[idx]:
                            generate_text[idx].append(next_token_ids[idx].item())

            # Decode all sequences
            generated_texts = [self.tokenizer.decode(text, skip_special_tokens=True) for text in generate_text]

        return generated_texts