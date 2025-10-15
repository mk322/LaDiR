import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import random
import json
from torch.utils.checkpoint import checkpoint
from transformers.configuration_utils import PretrainedConfig

from fm_noise_scheduler import FlowMatchEulerDiscreteScheduler

def count_parameters(model: nn.Module) -> int:
    """
    Counts the total number of trainable parameters in a PyTorch module.

    Args:
        model: A PyTorch nn.Module.

    Returns:
        The total number of trainable parameters as an integer.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

def rescale(z, shift, scale, mode="encode"):
    if mode == "encode":
        return (z - shift) * scale 
    else:
        return 1 / scale * z + shift

def rescale_only(z, shift, scale, mode="encode"):
    if mode == "encode":
        return z * scale 
    else:
        return 1 / scale * z
    
def freeze_module(module):
    for param in module.parameters():
        param.requires_grad = False


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class Projector(nn.Module):
    """MLP Projector layer."""

    def __init__(self, dim, mlp_dim=None, out_dim=None):
        super(Projector, self).__init__()
        self.fc1 = nn.Linear(dim, mlp_dim or dim)
        self.fc2 = nn.Linear(mlp_dim or dim, out_dim or dim)
        self.activation = nn.SiLU()

    def forward(self, x) -> torch.Tensor:
        return self.fc2(self.activation(self.fc1(x)))
    
class TimeCondEmbed(nn.Module):
    """Time-Condition embedding layer."""

    def __init__(self, cond_dim, embed_dim, freq_dim=256):
        super(TimeCondEmbed, self).__init__()
        self.timestep_proj = Projector(freq_dim, embed_dim, embed_dim)
        self.condition_proj = Projector(cond_dim, embed_dim, embed_dim)
        self.freq_dim, self.time_freq = freq_dim, None

    def get_freq_embed(self, timestep, dtype) -> torch.Tensor:
        if self.time_freq is None:
            dim, log_theta = self.freq_dim // 2, 9.210340371976184  # math.log(10000)
            freq = torch.arange(dim, dtype=torch.float32, device=timestep.device)
            self.time_freq = freq.mul(-log_theta / dim).exp().unsqueeze(0)
        emb = timestep.unsqueeze(-1).float() * self.time_freq
        return torch.cat([emb.cos(), emb.sin()], dim=-1).to(dtype=dtype)

    def get_t_embed(self, timestep, dtype):
        return self.timestep_proj(self.get_freq_embed(timestep, dtype))
        
    def forward(self, timestep, z) -> torch.Tensor:
        t = self.timestep_proj(self.get_freq_embed(timestep, z.dtype))
        #print("t.shape", t.shape)
        #print("self.condition_proj(z)", self.condition_proj(z).shape)
        #return self.condition_proj(z).add_(t)
        return z.add_(t.unsqueeze_(1) if t.dim() == 2 else t)
        #return self.condition_proj(z).add_(t.unsqueeze_(1) if t.dim() == 2 else t)


class LMFusionConfig(PretrainedConfig):
    def __init__(self, hidden_size=768, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        #self.num_layers = num_layers

class LMFusionModel(nn.Module):
    """
    LMFusion model that combines language models with VAE compression and diffusion generation.
    
    This model:
    - Uses a pre-trained language model (LLaMA) for text processing
    - Compresses text into latent thought tokens using a VAE
    - Generates diverse thought tokens using diffusion processes
    - Supports classifier-free guidance for controlled generation
    
    Args:
        text_llama: Pre-trained language model for text processing
        thought_llama: Language model for thought processing (currently unused)
        autoencoder: VAE model for text compression
        tokenizer: Text tokenizer
        model_config: Model configuration
        hidden_dim: Hidden dimension size (default: 4096)
        freeze_text: Whether to freeze text model parameters (default: True)
    """

    def __init__(self, 
                 text_llama,
                 thought_llama,
                 autoencoder,
                 tokenizer,
                 model_config,
                 hidden_dim: int = 4096,   # e.g. LLaMA hidden
                 freeze_text: bool = True,
):
        super().__init__()

        self.model_config = model_config

        self.tht_token_dim = 128
        self.text_llama = text_llama
        self.text_llama.model.config.use_cache = False
        self.autoencoder = autoencoder
        self.tokenizer = tokenizer

        # Freeze the autoencoder
        freeze_module(self.autoencoder)

        self.ae_to_latent = nn.Linear(self.tht_token_dim, hidden_dim, dtype=text_llama.dtype)
        self.output_projection = FinalLayer(hidden_dim, self.tht_token_dim)
        self.special_image_token_embeddings = nn.Linear(hidden_dim, 2, bias=False, dtype=text_llama.dtype)

        self.num_train_timesteps = 1000

        self.noise_scheduler = FlowMatchEulerDiscreteScheduler(1000)
        self.sample_scheduler = FlowMatchEulerDiscreteScheduler(1000)

        self.text_drop_prob = 0.1
        self.fake_input_embed = nn.Parameter(torch.zeros(1, hidden_dim))
        torch.nn.init.normal_(self.fake_input_embed, std=.02)
        
        self.time_embed = TimeCondEmbed(cond_dim=self.tht_token_dim, embed_dim=hidden_dim)
        self.t_embedder = TimeCondEmbed(cond_dim=hidden_dim, embed_dim=hidden_dim)

        self.diffusion_batch_mul = 8
        self.gradient_checkpointing = False
        self.prev_icae_params = {}

        self.initialize_weights()

    def initialize_weights(self):
        """
        Initialize model weights for better training stability.
        - Applies Xavier uniform initialization to linear layers
        - Initializes embedding layers with normal distribution
        - Zeros out output projection layers for stable training start
        """
        assert not hasattr(self, "llama")

        # Initialize transformer layers and linear projections
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        
        self.apply(_basic_init)
        
        # Initialize special token embeddings
        torch.nn.init.normal_(self.special_image_token_embeddings.weight, std=0.02)
        
        # Make sure fake input embedding is properly initialized
        if hasattr(self, 'fake_input_embed'):
            torch.nn.init.normal_(self.fake_input_embed, std=0.02)
 

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs):
        """
        Enable gradient checkpointing for this model.
        """
        self.gradient_checkpointing = True

    def replace_text_before_tht_start(
        self,
        inputs_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        video_start_token_id: int,
    ) -> torch.Tensor:
        """
        Replaces all text token embeddings before the first occurrence of the
        `video_start_token_id` with `null_embed`, excluding the very first token
        (BOS) at index 0.

        Args:
            inputs_embeds (torch.Tensor): [batch_size, seq_len, hidden_dim]
            input_ids (torch.Tensor):     [batch_size, seq_len]
            video_start_token_id (int):   the token ID that signifies start of video
            null_embed (torch.Tensor):    [hidden_dim] tensor to use for 'null' replacement

        Returns:
            torch.Tensor: The updated inputs_embeds after replacement.
        """
        batch_size, seq_len, hidden_dim = inputs_embeds.shape

        for b in range(batch_size):
            # Find all occurrences of the video_start_token_id in this sequence
            if random.random() < self.text_drop_prob:
                video_positions = (input_ids[b] == video_start_token_id).nonzero(as_tuple=True)[0]
                
                # If there's at least one occurrence, take the first one
                if len(video_positions) > 0:
                    first_video_idx = video_positions[0].item()

                    # Replace from index 1 up to (but not including) first_video_idx
                    # We skip index 0 in order to exclude the BOS token
                    if first_video_idx > 1: 
                        inputs_embeds[b, 1:first_video_idx] = self.fake_input_embed

        return inputs_embeds

    def build_bidirectional_tht_mask(
        self,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Build a custom 3D attention mask [B, L, L] where:
        - By default tokens follow a causal pattern (token i can attend 0..i).
        - PAD tokens are blocked everywhere.
        - THT tokens can attend each other fully (bidirectional).
        """

        B, L = input_ids.shape
        device = input_ids.device

        # 1) Start with a causal lower-triangular mask if desired,
        #    otherwise make it fully 1's if you want everything visible.
            # shape [L, L]
        base_mask_2d = torch.tril(torch.ones(L, L, dtype=torch.bool, device=device))
        # else:
        #     base_mask_2d = torch.ones(L, L, dtype=torch.bool, device=device)

        # 2) Expand to [B, L, L] to have one per batch example
        attn_mask = base_mask_2d.unsqueeze(0).expand(B, L, L).clone()

        # 3) Block out columns that are PAD tokens. If a position j is PAD,
        #    no one can attend to it. (Set attn_mask[:, :, j] = False).
        #    You may also want to block out the row dimension if you don't
        #    want to let PAD tokens attend *from* themselves, but typically
        #    either is fine or both is safer.
        pad_positions = (input_ids == self.tokenizer.pad_token_id)  # shape [B, L]
        # For each batch b, for each position j that is PAD, set attn_mask[b, :, j] = False
        attn_mask[pad_positions.unsqueeze(1).expand(B, L, L)] = False

        # 4) "Open up" the sub-block for THT tokens so they can see each other
        #    in a fully bidirectional manner, overriding any causal constraint.
        for b in range(B):
            # Collect all positions of THT tokens
            tht_positions = (input_ids[b] == self.tokenizer.tht_token_id).nonzero(as_tuple=True)[0]
            if len(tht_positions) > 1:
                # Turn on attn_mask[b, i, j] = True for every i,j in THT positions
                # even if i < j or i > j
                for i in tht_positions:
                    for j in tht_positions:
                        attn_mask[b, i, j] = True

        attn_mask = attn_mask.unsqueeze(1)

        return attn_mask.bfloat16()
    
    def get_dict_list(self, pred_thought_tokens_list, gt_thought_tokens, input, output, timestep_list=None):
        dict_list = []
        gt_cot_text = self.autoencoder.decode_text(gt_thought_tokens)
        if gt_cot_text in output:
            reward = 1.0
        else:
            reward = 0.0
        dict_list.append({"input": input, "solution": output, "infer_steps": "gt", "text": gt_cot_text, "reward": float(reward), "lst_reward": float(1), "tokens": gt_thought_tokens.reshape(1, 3, -1).float().tolist()})
        
        #print(pred_thought_tokens_list.shape, pred_thought_tokens_list[-1].shape)

        for bs in range(len(pred_thought_tokens_list)):
            lst_cot_text = self.autoencoder.decode_text(pred_thought_tokens_list[bs][-1].unsqueeze(0))
            if lst_cot_text in output:
                lst_reward = 1.0
            else:
                lst_reward = 0.0
            
            #for t in range(len(pred_thought_tokens_list[bs])):
            start_time = time.time()
            #print(f"decoding batch {bs}")
            cot_text_list = self.autoencoder.decode_text_batch(pred_thought_tokens_list[bs])
            end_time = time.time()
            print(f"decoding batch {bs} time: {end_time - start_time}s")
            for t in range(len(pred_thought_tokens_list[bs])):
                if cot_text_list[t] in output:
                    reward = 1.0
                else:
                    reward = 0.0
                dict_list.append({"input": input, "solution": output, "infer_steps": float(timestep_list[t]), "text": cot_text_list[t], "reward": float(reward), "lst_reward": float(lst_reward), "tokens": pred_thought_tokens_list[bs][t].reshape(1, 3, -1).float().tolist()})

        return dict_list
    
    def generate_data(self, input_ids_q, input, output, gt_thought: str):

        with torch.no_grad():
            #bs = input_ids_q.shape[0]
            bs = 20
            input_ids_q = input_ids_q.repeat(bs, 1)
            #q_embeds = self.text_llama.get_input_embeddings()(input_ids_q)
            gt_thought_tokens = self.autoencoder.encode_text(gt_thought)
            bs_gt_thought_tokens = gt_thought_tokens.repeat(bs, 1, 1)
            bs_gt_thought_tokens = bs_gt_thought_tokens.reshape(bs, -1, self.tht_token_dim).to(dtype=torch.bfloat16, device=input_ids_q.device)

            pred_thought_tokens_list, timestep_list = self.denoise_debug(input_ids_q, bs_gt_thought_tokens, guidance_scale=1)
            dict_list = self.get_dict_list(pred_thought_tokens_list, gt_thought_tokens, input, output, timestep_list)
            self.save_dicts_to_file(dict_list, "lrm/data/train_diffusion_generated_data_bs20.jsonl")


    def generate_debug(self, input_ids_q, gt_thought: str):

        with torch.no_grad():
            bs = input_ids_q.shape[0]
            #q_embeds = self.text_llama.get_input_embeddings()(input_ids_q)
            gt_thought_tokens = self.autoencoder.encode_text(gt_thought)

            #gt_thought_tokens = self.of_token
            gt_thought_tokens = gt_thought_tokens.reshape(bs, -1, self.tht_token_dim).to(dtype=torch.bfloat16, device=input_ids_q.device)

            pred_thought_tokens_list = self.denoise_debug(input_ids_q, gt_thought_tokens)
            dict_list = self.get_dict_list(pred_thought_tokens_list, gt_thought_tokens, input, output)
            self.save_dicts_to_file(dict_list, "lrm/data/cd4_lrm_diffusion_sampled.jsonl")

            #print("MSE:", F.mse_loss(input=pred_thought_tokens, target=thought_tokens))
            #thought_token_embeds = torch.randn(thought_token_embeds.shape, dtype=thought_token_embeds.dtype, device=thought_token_embeds.device)

            #mse = F.mse_loss(input=pred_thought_tokens.to(input_ids_q.device), target=gt_thought_tokens.to(input_ids_q.device))

            #torch.save(pred_thought_tokens, '4gpus_512tokens_pred.pt')
            #torch.save(gt_thought_tokens, '4gpus_512tokens_gt.pt')

            #print("pred_thought_tokens.shape", pred_thought_tokens.shape)

            # dict_list = []
            # gt_cot_text = self.autoencoder.decode_text(gt_thought_tokens)
            # dict_list.append({"text": gt_cot_text, "tokens": gt_thought_tokens})

            # print(f"gt cot text:\n{gt_cot_text}")

            # for t in range(len(pred_thought_tokens_list)):
            #     cot_text = self.autoencoder.decode_text(pred_thought_tokens_list[t])
            #     print(f"generated cot text with infer steps={t}\n", cot_text)
            # #proj_pred_thought_token = self.ae_to_latent(pred_thought_tokens)[0]

            #cot_text = self.autoencoder.decode_text(pred_thought_tokens)


            # LLM generate: 
            # special_image_token_embeddings_weight = self.special_image_token_embeddings.weight

            # concat_embeds = torch.concat(tensors=[q_embeds, special_image_token_embeddings_weight[0].reshape(bs, 1, -1), pred_thought_tokens, special_image_token_embeddings_weight[1].reshape(bs, 1, -1)], dim=1)
            
            # concat_embeds = concat_embeds.to(self.text_llama.dtype)

            # attention_mask = torch.ones(concat_embeds.shape[:2], dtype=torch.long, device=concat_embeds.device)

            # outputs = self.text_llama.generate(
            #     inputs_embeds=concat_embeds,  
            #     attention_mask=attention_mask,
            #     max_new_tokens=400,
            #     top_p=1.0,
            #     temperature=0.6,
            #     #eos_token_id=self.tokenizer.eos_token_id,      
            #     do_sample=True,             
            #     use_cache=True                
            # )

            # text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        return cot_text, None


    def generate_batch(self, input_ids_q, gt_thought: str, cfg=1):

        with torch.no_grad():
            bs = input_ids_q.shape[0]
            #q_embeds = self.text_llama.get_input_embeddings()(input_ids_q)
            gt_thought_tokens = self.autoencoder.encode_text(gt_thought)

            #gt_thought_tokens = self.of_token
            gt_thought_tokens = gt_thought_tokens.reshape(bs, -1, self.tht_token_dim).to(dtype=torch.bfloat16, device=input_ids_q.device)

            extend_bs = 100
            input_ids_q = input_ids_q.repeat(extend_bs, 1)
            gt_thought_tokens = gt_thought_tokens.repeat(extend_bs, 1, 1)

            pred_thought_tokens_list = self.denoise(input_ids_q, gt_thought_tokens, guidance_scale=cfg)
    
            #print("MSE:", F.mse_loss(input=pred_thought_tokens, target=thought_tokens))
            #thought_token_embeds = torch.randn(thought_token_embeds.shape, dtype=thought_token_embeds.dtype, device=thought_token_embeds.device)

            #mse = F.mse_loss(input=pred_thought_tokens.to(input_ids_q.device), target=gt_thought_tokens.to(input_ids_q.device))

            #torch.save(pred_thought_tokens, '4gpus_512tokens_pred.pt')
            #torch.save(gt_thought_tokens, '4gpus_512tokens_gt.pt')

            
            #print("pred_thought_tokens.shape", pred_thought_tokens.shape)
            gt_cot_text = self.autoencoder.decode_text(gt_thought_tokens[0].unsqueeze(0))
            cot_text_list = []
            print(f"gt cot text:\n{gt_cot_text}")
            # for t in range(0, len(pred_thought_tokens_list), 25):
            for bs in range(extend_bs):
                cot_text = self.autoencoder.decode_text(pred_thought_tokens_list[bs].unsqueeze(0))
                cot_text_list.append(cot_text)
                print(f"generated cot text with bs={bs}\n", cot_text)
    
            #cot_text = self.autoencoder.decode_text(pred_thought_tokens_list[t].mean(dim=0))

            #proj_pred_thought_token = self.ae_to_latent(pred_thought_tokens)[0]

            #cot_text = self.autoencoder.decode_text(pred_thought_tokens)


            # LLM generate: 
            # special_image_token_embeddings_weight = self.special_image_token_embeddings.weight

            # concat_embeds = torch.concat(tensors=[q_embeds, special_image_token_embeddings_weight[0].reshape(bs, 1, -1), pred_thought_tokens, special_image_token_embeddings_weight[1].reshape(bs, 1, -1)], dim=1)
            
            # concat_embeds = concat_embeds.to(self.text_llama.dtype)

            # attention_mask = torch.ones(concat_embeds.shape[:2], dtype=torch.long, device=concat_embeds.device)

            # outputs = self.text_llama.generate(
            #     inputs_embeds=concat_embeds,  
            #     attention_mask=attention_mask,
            #     max_new_tokens=400,
            #     top_p=1.0,
            #     temperature=0.6,
            #     #eos_token_id=self.tokenizer.eos_token_id,      
            #     do_sample=True,             
            #     use_cache=True                
            # )

            # text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return cot_text_list, None

    def save_dicts_to_file(self, dict_list, filename):
        """
        Save a list of dicts with text and tensor to a file line by line.

        Args:
            dict_list (list): List of dicts like {"text": str, "tokens": torch.Tensor}
            filename (str): Output file path
        """
        with open(filename, "a") as f:
            for item in dict_list:
                f.write(json.dumps(item) + "\n")


    def generate_batch_with_best_of_n(self, input_ids_q, gt_thought: str, cfg=1, lrm=None, reward_guidance_scale=1.0, lrm_input_ids=None):

        with torch.no_grad():
            bs = input_ids_q.shape[0]
            gt_thought_tokens = self.autoencoder.encode_text(gt_thought)

            gt_thought_tokens = gt_thought_tokens.reshape(bs, -1, self.tht_token_dim).to(dtype=torch.bfloat16, device=input_ids_q.device)

            extend_bs = 100
            input_ids_q = input_ids_q.repeat(extend_bs, 1)
            gt_thought_tokens = gt_thought_tokens.repeat(extend_bs, 1, 1)
            lrm_input_ids = lrm_input_ids.repeat(extend_bs, 1)
            
            pred_thought_tokens_list = self.denoise(input_ids_q, gt_thought_tokens, guidance_scale=cfg)
    
            reward = lrm.get_reward(lrm_input_ids, pred_thought_tokens_list, guidance_scale=reward_guidance_scale) 

            gt_cot_text = self.autoencoder.decode_text(gt_thought_tokens[0].unsqueeze(0))
            cot_text_list = []
            print(f"gt cot text:\n{gt_cot_text}")
            # for t in range(0, len(pred_thought_tokens_list), 25):
            for bs in range(extend_bs):
                cot_text = self.autoencoder.decode_text(pred_thought_tokens_list[bs].unsqueeze(0))
                cot_text_list.append(cot_text)
                print(f"generated cot text with bs={bs}\n", cot_text, f"reward={reward[bs]}")
            
            best_bs = torch.argmax(reward)
            best_cot_text = cot_text_list[best_bs]
            print(f"\n\nbest cot text:\n{best_cot_text}, reward={reward[best_bs]}")


        return cot_text_list, reward
    

    def generate_batch_with_lrm(self, input_ids_q, gt_thought: str, cfg=1, lrm=None, lrm_scale=1.0, reward_guidance_scale=1.0, lrm_input_ids=None):

        bs = input_ids_q.shape[0]
        #q_embeds = self.text_llama.get_input_embeddings()(input_ids_q)
        with torch.no_grad():
            gt_thought_tokens = self.autoencoder.encode_text(gt_thought)

        #gt_thought_tokens = self.of_token
        gt_thought_tokens = gt_thought_tokens.reshape(bs, -1, self.tht_token_dim).to(dtype=torch.bfloat16, device=input_ids_q.device)

        extend_bs = 1
        input_ids_q = input_ids_q.repeat(extend_bs, 1)
        lrm_input_ids = lrm_input_ids.repeat(extend_bs, 1)
        gt_thought_tokens = gt_thought_tokens.repeat(extend_bs, 1, 1)

        pred_thought_tokens_list, reward = self.denoise_with_reward_guidance(input_ids_q, gt_thought_tokens, lrm, lrm_input_ids, lrm_scale=lrm_scale, reward_guidance_scale=reward_guidance_scale, guidance_scale=cfg)

        #print("MSE:", F.mse_loss(input=pred_thought_tokens, target=thought_tokens))
        #thought_token_embeds = torch.randn(thought_token_embeds.shape, dtype=thought_token_embeds.dtype, device=thought_token_embeds.device)

        #mse = F.mse_loss(input=pred_thought_tokens.to(input_ids_q.device), target=gt_thought_tokens.to(input_ids_q.device))

        #torch.save(pred_thought_tokens, '4gpus_512tokens_pred.pt')
        #torch.save(gt_thought_tokens, '4gpus_512tokens_gt.pt')

        #print("pred_thought_tokens.shape", pred_thought_tokens.shape)
        with torch.no_grad():
            gt_cot_text = self.autoencoder.decode_text(gt_thought_tokens[0].unsqueeze(0))
            
        cot_text_list = []
        print(f"gt cot text:\n{gt_cot_text}")
        # for t in range(0, len(pred_thought_tokens_list), 25):
        for bs in range(extend_bs):
            cot_text = self.autoencoder.decode_text(pred_thought_tokens_list[bs].unsqueeze(0))
            cot_text_list.append(cot_text)
            print(f"generated cot text with bs={bs}\n", cot_text)
            
        #cot_text = self.autoencoder.decode_text(pred_thought_tokens_list[t].mean(dim=0))

        #proj_pred_thought_token = self.ae_to_latent(pred_thought_tokens)[0]

        #cot_text = self.autoencoder.decode_text(pred_thought_tokens)


        # LLM generate: 
        # special_image_token_embeddings_weight = self.special_image_token_embeddings.weight

        # concat_embeds = torch.concat(tensors=[q_embeds, special_image_token_embeddings_weight[0].reshape(bs, 1, -1), pred_thought_tokens, special_image_token_embeddings_weight[1].reshape(bs, 1, -1)], dim=1)
        
        # concat_embeds = concat_embeds.to(self.text_llama.dtype)

        # attention_mask = torch.ones(concat_embeds.shape[:2], dtype=torch.long, device=concat_embeds.device)

        # outputs = self.text_llama.generate(
        #     inputs_embeds=concat_embeds,  
        #     attention_mask=attention_mask,
        #     max_new_tokens=400,
        #     top_p=1.0,
        #     temperature=0.6,
        #     #eos_token_id=self.tokenizer.eos_token_id,      
        #     do_sample=True,             
        #     use_cache=True                
        # )

        # text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        return cot_text_list, reward

    def distance_repulsion_grad(self, z: torch.Tensor, sigma):
        """
        Compute the repulsive gradient for a tensor of shape (B, S, D).

        Args
        ----
        z     : Tensor, shape (batch, seq_len, dim)
        sigma : scalar or 1‑D tensor – bandwidth of the RBF kernel

        Returns
        -------
        grad  : Tensor, shape (batch, seq_len, dim)
        """
        B, S, D = z.shape                    # batch, sequence, hidden‑dim
        z_flat  = z.reshape(B * S, D)        # (BS, D)

        diff   = z_flat[:, None, :] - z_flat[None, :, :]      # (BS, BS, D)
        dist2  = (diff ** 2).sum(-1, keepdim=True)            # (BS, BS, 1)

        # weight = exp(-‖diff‖² / σ²) · (1 − ‖diff‖² / σ²)
        w      = torch.exp(-dist2 / sigma ** 2) * (1.0 - dist2 / sigma ** 2)

        # Optional: ignore self‑interaction (diagonal)
        w.diagonal(dim1=0, dim2=1).zero_()

        grad_flat = (w * diff).sum(1) * 2                     # (BS, D)
        grad      = grad_flat.view(B, S, D)                   # (B, S, D)
        return grad

    def generate_batch_diversity(self, input_ids_q, gt_thought: str, cfg=1, diversity_scale=0.10, lrm_input_ids=None):

        bs = input_ids_q.shape[0]
        #q_embeds = self.text_llama.get_input_embeddings()(input_ids_q)
        with torch.no_grad():
            gt_thought_tokens = self.autoencoder.encode_text(gt_thought)

        #gt_thought_tokens = self.of_token
        gt_thought_tokens = gt_thought_tokens.reshape(bs, -1, self.tht_token_dim).to(dtype=torch.bfloat16, device=input_ids_q.device)

        extend_bs = 100
        input_ids_q = input_ids_q.repeat(extend_bs, 1)
        #lrm_input_ids = lrm_input_ids.repeat(extend_bs, 1)
        gt_thought_tokens = gt_thought_tokens.repeat(extend_bs, 1, 1)

        pred_thought_tokens_list = self.denoise_with_reward_guidance_and_diversity(input_ids_q, gt_thought_tokens, guidance_scale=cfg, diversity_sigma=50, max_diversity_scale=diversity_scale)

        #print("MSE:", F.mse_loss(input=pred_thought_tokens, target=thought_tokens))
        #thought_token_embeds = torch.randn(thought_token_embeds.shape, dtype=thought_token_embeds.dtype, device=thought_token_embeds.device)

        #mse = F.mse_loss(input=pred_thought_tokens.to(input_ids_q.device), target=gt_thought_tokens.to(input_ids_q.device))

        #torch.save(pred_thought_tokens, '4gpus_512tokens_pred.pt')
        #torch.save(gt_thought_tokens, '4gpus_512tokens_gt.pt')

        #print("pred_thought_tokens.shape", pred_thought_tokens.shape)
        with torch.no_grad():
            gt_cot_text = self.autoencoder.decode_text(gt_thought_tokens[0].unsqueeze(0))
            
        cot_text_list = []
        print(f"gt cot text:\n{gt_cot_text}")
        # for t in range(0, len(pred_thought_tokens_list), 25):
        for bs in range(extend_bs):
            cot_text = self.autoencoder.decode_text(pred_thought_tokens_list[bs].unsqueeze(0))
            cot_text_list.append(cot_text)
            print(f"generated cot text with bs={bs}\n", cot_text)
            
        #cot_text = self.autoencoder.decode_text(pred_thought_tokens_list[t].mean(dim=0))

        #proj_pred_thought_token = self.ae_to_latent(pred_thought_tokens)[0]

        #cot_text = self.autoencoder.decode_text(pred_thought_tokens)


        # LLM generate: 
        # special_image_token_embeddings_weight = self.special_image_token_embeddings.weight

        # concat_embeds = torch.concat(tensors=[q_embeds, special_image_token_embeddings_weight[0].reshape(bs, 1, -1), pred_thought_tokens, special_image_token_embeddings_weight[1].reshape(bs, 1, -1)], dim=1)
        
        # concat_embeds = concat_embeds.to(self.text_llama.dtype)

        # attention_mask = torch.ones(concat_embeds.shape[:2], dtype=torch.long, device=concat_embeds.device)

        # outputs = self.text_llama.generate(
        #     inputs_embeds=concat_embeds,  
        #     attention_mask=attention_mask,
        #     max_new_tokens=400,
        #     top_p=1.0,
        #     temperature=0.6,
        #     #eos_token_id=self.tokenizer.eos_token_id,      
        #     do_sample=True,             
        #     use_cache=True                
        # )

        # text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        return cot_text_list


    def generate(self, input_ids_q, gt_thought: str, cfg=1):

        with torch.no_grad():
            bs = input_ids_q.shape[0]
            #q_embeds = self.text_llama.get_input_embeddings()(input_ids_q)
            gt_thought_tokens = self.autoencoder.encode_text(gt_thought)

            #gt_thought_tokens = self.of_token
            gt_thought_tokens = gt_thought_tokens.reshape(bs, -1, self.tht_token_dim).to(dtype=torch.bfloat16, device=input_ids_q.device)

            pred_thought_tokens = self.denoise(input_ids_q, gt_thought_tokens, guidance_scale=cfg)
    
            #print("MSE:", F.mse_loss(input=pred_thought_tokens, target=thought_tokens))
            #thought_token_embeds = torch.randn(thought_token_embeds.shape, dtype=thought_token_embeds.dtype, device=thought_token_embeds.device)

            mse = F.mse_loss(input=pred_thought_tokens.to(input_ids_q.device), target=gt_thought_tokens.to(input_ids_q.device))

            #torch.save(pred_thought_tokens, '4gpus_512tokens_pred.pt')
            #torch.save(gt_thought_tokens, '4gpus_512tokens_gt.pt')

            #print("pred_thought_tokens.shape", pred_thought_tokens.shape)
            gt_cot_text = self.autoencoder.decode_text(gt_thought_tokens)

            print(f"gt cot text:\n{gt_cot_text}")
            cot_text = self.autoencoder.decode_text(pred_thought_tokens)

            #proj_pred_thought_token = self.ae_to_latent(pred_thought_tokens)[0]

            #cot_text = self.autoencoder.decode_text(pred_thought_tokens)


            # LLM generate: 
            # special_image_token_embeddings_weight = self.special_image_token_embeddings.weight

            # concat_embeds = torch.concat(tensors=[q_embeds, special_image_token_embeddings_weight[0].reshape(bs, 1, -1), pred_thought_tokens, special_image_token_embeddings_weight[1].reshape(bs, 1, -1)], dim=1)
            
            # concat_embeds = concat_embeds.to(self.text_llama.dtype)

            # attention_mask = torch.ones(concat_embeds.shape[:2], dtype=torch.long, device=concat_embeds.device)

            # outputs = self.text_llama.generate(
            #     inputs_embeds=concat_embeds,  
            #     attention_mask=attention_mask,
            #     max_new_tokens=400,
            #     top_p=1.0,
            #     temperature=0.6,
            #     #eos_token_id=self.tokenizer.eos_token_id,      
            #     do_sample=True,             
            #     use_cache=True                
            # )

            # text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        return cot_text, mse

    def denoise_debug(self, q_input_ids, gt_thought_tokens, guidance_scale=1, generator=None):
        """Run diffusion denoising process."""

        bs, seq, token_hidden_dim = gt_thought_tokens.shape

        x = torch.randn(bs, seq, token_hidden_dim, dtype=torch.bfloat16).to(q_input_ids.device)
        
        self.sample_scheduler._step_index = None
        self.sample_scheduler.set_timesteps(num_inference_steps=50)
        
        ret_list = []
        timestep_list = []


        #print(self.noise_scheduler.timesteps)
        for t in self.sample_scheduler.timesteps:
            ret_list.append(x.clone())
            timestep_list.append(t)
            proj_x = self.ae_to_latent(x)
            timestep = torch.as_tensor(t, device=x.device).expand(bs)

            proj_model_pred = self.inference_step(q_input_ids, proj_x, timestep, infer_cfg=guidance_scale) # dim=1024

            x = self.sample_scheduler.step(model_output=proj_model_pred, timestep=t, sample=x, generator=generator).prev_sample # dim=1024

        ret_list.append(x.clone())
        timestep_list.append(0)
        return torch.stack(ret_list, dim=0).transpose(0, 1), timestep_list
    
    def denoise(self, q_input_ids, gt_thought_tokens, guidance_scale=1, generator=None) -> torch.Tensor:
        """Run diffusion denoising process."""

        bs, seq, token_hidden_dim = gt_thought_tokens.shape

        # if guidance_scale > 1:
        #     x = torch.randn(bs * 2, seq, token_hidden_dim, dtype=torch.bfloat16).to(q_input_ids.device)
        # else:
        x = torch.randn(bs, seq, token_hidden_dim, dtype=torch.bfloat16).to(q_input_ids.device) * 2.5
        
        if self.sample_scheduler.config.prediction_type == "flow":
            self.sample_scheduler._step_index = None
        self.sample_scheduler.set_timesteps(num_inference_steps=50)
        
        #print(self.noise_scheduler.timesteps)
        for t in self.sample_scheduler.timesteps:

            proj_x = self.ae_to_latent(x)
            timestep = torch.as_tensor(t, device=x.device).expand(bs)

            proj_model_pred = self.inference_step(q_input_ids, proj_x, timestep, infer_cfg=guidance_scale) # dim=1024

            x = self.sample_scheduler.step(model_output=proj_model_pred, timestep=t, sample=x, generator=generator).prev_sample # dim=1024
            
            print(f"step {t}, x.norm={x.norm()}")
            #x = rescale(x, self.model_config.ae.shift_factor, self.model_config.ae.scale_factor, mode="decode")

            mse_loss = nn.functional.mse_loss(x.float(), gt_thought_tokens.float())
            #print(f"mse_loss at steps {t}={mse_loss}")
        
        print(f"final before rescale, x.norm={x.norm()}")

        #x = rescale(x, self.model_config.ae.shift_factor, self.model_config.ae.scale_factor, mode="decode")
        #print(f"final after rescale, x.norm={x.norm()}")
        return x

    # ✱ 1.  distance–repulsion gradient     f(u)=u·exp(−u/σ²)
    def distance_grad(self, z: torch.Tensor, sigma: float) -> torch.Tensor:
        """
        z      : [B, N, D]    latent tokens at the current step
        sigma  : controls the 'sweet‑spot' distance  (≈ median ||z_i−z_j||)
        return : ∇_z R  with same shape as z
        """
        # pairwise differences along the token/row axis (N)
        diff   = z.unsqueeze(2) - z.unsqueeze(1)          # [B, N, N, D]
        dist2  = (diff ** 2).sum(-1, keepdim=True)        # [B, N, N, 1]
        w      = torch.exp(-dist2 / sigma ** 2) * (1.0 - dist2 / sigma ** 2)
        grad   = (w * diff).sum(2) * 2.0                  # sum_j 2 w_ij (z_i − z_j)
        return grad

    def median_bandwidth_latent(self, z: torch.Tensor) -> float:
        """
        z : [B, N, D]  current latent tokens  (float32/F16 okay)
        σ_t = median ||z_i − z_j||₂   over all pairs (i<j)
        """
        B, N, D = z.shape
        flat = z.reshape(B * N, D).float()          # [B*N, D]
        dist = torch.cdist(flat, flat, p=2)         # [B*N, B*N]
        return dist[dist > 0].median().item() + 1e-6

    def get_diversity_scale(self, t, T=1000,
                            base_scale: float = 1.0,
                            min_scale: float  = 0.1,
                            power: float      = 2.0):

        alpha = (t / T).clamp(min=0.0, max=1.0)
        return min_scale + (base_scale - min_scale) * (alpha ** power)

    def denoise_with_reward_guidance_and_diversity(
        self,
        q_input_ids,
        gt_thought_tokens,
        # lrm,
        # lrm_input_ids,
        # reward_guidance_scale: float = 1.0,
        guidance_scale:       float = 1.0,
        max_diversity_scale:      float = 0.10,   # ✱ γ  in the pseudo‑code
        diversity_sigma = None,
        generator=None
    ):
        """
        Same I/O as before, plus:
            diversity_scale  – weight γ for the repulsion kick
            diversity_sigma  – σ in the objective; if None we auto‑estimate
        """
        bs, seq, token_dim = gt_thought_tokens.shape
        device = q_input_ids.device

        # initialise latent
        x = torch.randn(bs, seq, token_dim, dtype=torch.bfloat16, device=device)

        if self.sample_scheduler.config.prediction_type == "flow":
            self.sample_scheduler._step_index = None
        self.sample_scheduler.set_timesteps(num_inference_steps=50)

        # # ✱ choose σ if the caller did not
        # if diversity_sigma is None:
        #     diversity_sigma = self.median_bandwidth_latent(x)
            #diversity_sigma = float(token_dim) ** 0.5  # heuristic; adjust if needed

        #print(f"diversity_sigma={diversity_sigma}")

        for t in self.sample_scheduler.timesteps:
            # ----- 1. model prediction (classifier‑free guidance) -----
            with torch.no_grad():
                proj_x  = self.ae_to_latent(x)                       # [B, N, D_lat]
                timestep = torch.full((bs,), t, device=device)
                model_pred = self.inference_step(
                    q_input_ids,
                    proj_x,
                    timestep,
                    infer_cfg=guidance_scale
                )                                                   # [B, N, token_dim]

            # # ----- 2. reward gradient guidance -----
            # _, reward_grad = lrm.get_reward_gradient(
            #     lrm_input_ids, model_pred, timestep,
            #     guidance_scale=reward_guidance_scale
            # )                                                       # [B, N, token_dim]
        #if diversity_sigma is None:
            diversity_sigma = self.median_bandwidth_latent(model_pred)
            print(f"diversity_sigma={diversity_sigma}")
            # ----- 3. diversity kick (ascent on R)  ✱ -----
            repulsion_grad = self.distance_grad(
                model_pred.float(), sigma=diversity_sigma
            ).to(model_pred.dtype)


            diversity_scale = self.get_diversity_scale(torch.tensor(t), T=1000, base_scale=max_diversity_scale, min_scale=0.1, power=2.0)
            print(f"diversity_scale={diversity_scale}")
            # combine:   y = base + reward + γ · ∇R
            guided_pred = model_pred + diversity_scale * repulsion_grad

            # ----- 4. scheduler step -----
            x = self.sample_scheduler.step(
                model_output=guided_pred,
                timestep=t,
                sample=x,
                generator=generator
            ).prev_sample

            # optional debug
            mse_loss = F.mse_loss(x.float(), gt_thought_tokens.float())
            print(f"t={int(t):3d} | MSE={mse_loss:.4f}")

        return x


    def denoise_with_reward_guidance(self, q_input_ids, gt_thought_tokens, lrm, lrm_input_ids, lrm_scale=1.0, reward_guidance_scale=1.0, guidance_scale=1, generator=None) -> torch.Tensor:
        """
        Run diffusion denoising process with reward guidance.
        
        Args:
            q_input_ids: Input token IDs [B, T]
            gt_thought_tokens: Ground truth thought tokens [B, N, D]
            reward_guidance_scale: Scale factor for reward gradient guidance
            guidance_scale: Scale factor for classifier-free guidance
            generator: Random number generator for noise
            
        Returns:
            torch.Tensor: Denoised thought tokens [B, N, D]
        """
        bs, seq, token_hidden_dim = gt_thought_tokens.shape
        x = torch.randn(bs, seq, token_hidden_dim, dtype=torch.bfloat16).to(q_input_ids.device)
        
        if self.sample_scheduler.config.prediction_type == "flow":
            self.sample_scheduler._step_index = None
        self.sample_scheduler.set_timesteps(num_inference_steps=50)

        timestep = self.sample_scheduler.timesteps[0]
        self.sample_scheduler._init_step_index(timestep)

        # Get the timestep differences for the gradient computation
        #dts = self.sample_scheduler.sigma[:-1] - self.sample_scheduler.sigma[1:]
        #dts = torch.as_tensor(dts, dtype=torch.bfloat16, device=x.device)
        #dts = dts.unsqueeze(0).expand(bs, -1)  # Expand to [bs, num_timesteps]  

        #print("dts", dts.shape, dts.dtype, len(self.sample_scheduler.timesteps))

        for i, t in enumerate(self.sample_scheduler.timesteps):
            # Get the current prediction
            with torch.no_grad():
                proj_x = self.ae_to_latent(x)  # [B, N, hidden_dim]
                timestep = torch.as_tensor(t, dtype=torch.bfloat16, device=x.device).expand(bs)
                # Get the base model prediction
                model_pred = self.inference_step(q_input_ids, proj_x, timestep, infer_cfg=guidance_scale)  # [B, N, tht_token_dim]
            
            # Get reward gradient and project it to match model prediction shape
            _, reward_grad = lrm.get_reward_gradient(lrm_input_ids, x, timestep, lrm_scale=lrm_scale) 
            
            #reward_grad = torch.clamp(reward_grad, -0.05, 0.05)
            #print(f"model_pred={model_pred}")
            
            #print(f"self.sample_scheduler._step_index={self.sample_scheduler._step_index+1}")

            sigma = self.sample_scheduler.sigmas[self.sample_scheduler._step_index+1]

            #print(f"sigmas={sigma}, coeff={sigma / (1 - sigma)}")

            # Combine the base prediction with reward guidance

            guided_pred = model_pred - (sigma / (1 - sigma) * reward_guidance_scale * reward_grad)
            #x = x - dts[:, i] * v_pred


            # guided_pred = model_pred + sigmas / (1 - sigmas) * reward_grad
            
            # Take a step with the guided prediction
            x = self.sample_scheduler.step(
                model_output=guided_pred,
                timestep=t,
                sample=x,
                generator=generator
            ).prev_sample
            
            # Print progress
            mse_loss = nn.functional.mse_loss(x.float(), gt_thought_tokens.float())
            print(f"mse_loss at step {t}={mse_loss}")

        reward = lrm.get_reward(lrm_input_ids, x, rm_cfg=lrm_scale) 
        #print(f"final reward={reward}")

        return x, reward


    def guidance(model, reward_model, prompt_embeds, latents, timesteps, reward_weight):
        """
        # model: Flow model that predicts velocity given latents and conditions
        # reward_model: Model that evaluates the quality of latents based on prompt
        embeddings and timestep
        # prompt_embeds: Embeddings of the text prompts
        # latents: Initial noise
        # timesteps: Sequence of timesteps
        # reward_weight: weighting coefficient of multi-dimensional rewards
        # guidance_scale: scale factor for reward guidance
        """
        dts = timesteps[:-1] - timesteps[1:]
        for i, t in enumerate(timesteps):
            v_pred = model(latents, prompt_embeds, t)
            latents = latents.detach().requires_grad_(True)
            reward = reward_model(latents, prompt_embeds, t)
            reward = reward * reward_weight
            reward_guidance = torch.autograd.grad(reward, latents)
            v_pred = v_pred - guidance_scale * t / (1 - t) * reward_guidance
            latents = latents - dts[i] * v_pred
        return latents


    def inference_step(self, input_ids, thought_token_embeds, timestep, infer_cfg=1):

        B, T = input_ids.shape
        _, N, D = thought_token_embeds.shape
        device = input_ids.device

        inputs_embeds = self.text_llama.get_input_embeddings()(input_ids).to(torch.bfloat16)

        bot_token_embeds = self.special_image_token_embeddings.weight[0].unsqueeze(0).unsqueeze(1).expand(B, 1, -1).to(inputs_embeds.dtype)
        timestep_embeds = self.time_embed.get_t_embed(timestep=timestep, dtype=inputs_embeds.dtype).reshape(B, 1, -1)

        inputs_embeds = torch.cat([inputs_embeds, bot_token_embeds, timestep_embeds, thought_token_embeds], dim=1)

        # 1) Create the [BOT, TIMESTEP, THT*N, EOT] block for each example
        #    so we can cat them to the original input_ids per row.
        #    That block has length = N + 2 (because BOT + EOT around N THT).
        appended_tokens = torch.tensor(
            [self.tokenizer.bot_token_id] + 
            [self.tokenizer.time_token_id] +
            [self.tokenizer.tht_token_id]*N,
            device=device
        )  # shape (N+2,)
        

        appended_tokens = appended_tokens.unsqueeze(0).expand(B, -1)  # shape (B, N+2)

        new_input_ids = torch.cat([input_ids, appended_tokens], dim=1)  # shape (B, T + N + 2)

        attention_mask = self.build_bidirectional_tht_mask(
            input_ids=new_input_ids, 
        )

        outputs = self.text_llama(
                        inputs_embeds=inputs_embeds,
                        attention_mask=attention_mask,
                        #position_ids=position_ids,
                        output_hidden_states=True,
                        return_dict=True,
                        use_cache=False,
                    )
        outputs_hidden_states = outputs.hidden_states[-1] 
        model_pred = outputs_hidden_states[:, -N:, :].reshape(B, N, -1)
        t_embedder_pred = self.t_embedder.get_t_embed(timestep, model_pred.dtype).reshape(B, -1)
        model_pred = self.output_projection(model_pred, t_embedder_pred)
        #model_pred = self.output_projection(model_pred)

        if infer_cfg > 1:
            null_inputs_embeds = inputs_embeds.clone()
            
            for b in range(B):
                bot_positions = (new_input_ids[b] == self.tokenizer.bot_token_id).nonzero(as_tuple=True)[0]
                
                # If there's at least one occurrence, take the first one
                if len(bot_positions) > 0:
                    first_video_idx = bot_positions[0].item()

                    # Replace from index 1 up to (but not including) first_video_idx
                    # We skip index 0 in order to exclude the BOS token
                    if first_video_idx > 1: 
                        null_inputs_embeds[b, 1:first_video_idx] = self.fake_input_embed
                else:
                    print("didn't find!!!")

            null_outputs = self.text_llama(
                            inputs_embeds=null_inputs_embeds,
                            attention_mask=attention_mask,
                            #position_ids=position_ids,
                            output_hidden_states=True,
                            return_dict=True,
                            use_cache=False,
                        )
            outputs_hidden_states = null_outputs.hidden_states[-1] 
            null_model_pred = outputs_hidden_states[:, -N:, :].reshape(B, N, -1)

            null_model_pred = self.output_projection(null_model_pred, t_embedder_pred)
            #null_model_pred = self.output_projection(null_model_pred)
            model_pred = null_model_pred + infer_cfg * (model_pred - null_model_pred)

        return model_pred 
    
    def prepare_input(self, input_ids, thought_token_embeds, timestep, cfg_train=False, infer_cfg=1):

        B, T = input_ids.shape
        _, N, D = thought_token_embeds.shape

        device = input_ids.device

        # 1) Create the [BOT, TIMESTEP, THT*N, EOT] block for each example
        #    so we can cat them to the original input_ids per row.
        #    That block has length = N + 2 (because BOT + EOT around N THT).
        appended_tokens = torch.tensor(
            [self.tokenizer.bot_token_id] + 
            [self.tokenizer.time_token_id] +
            [self.tokenizer.tht_token_id]*N +
            [self.tokenizer.eot_token_id],
            device=device
        )  # shape (N+2,)
        

        appended_tokens = appended_tokens.unsqueeze(0).expand(B, -1)  # shape (B, N+2)

        new_input_ids = torch.cat([input_ids, appended_tokens], dim=1)  # shape (B, T + N + 2)

        #position_ids = self.create_position(input_ids, 3)

        #print(f"position_ids={position_ids}")

        attention_mask = self.build_bidirectional_tht_mask(
            input_ids=new_input_ids, 
        )

        #print("num_thought_tokens", num_thought_tokens)
        #attention_mask = new_input_ids.ne(self.tokenizer.pad_token_id)
        bot_token_mask = new_input_ids == self.tokenizer.bot_token_id
        eot_token_mask = new_input_ids == self.tokenizer.eot_token_id
        tht_token_mask = new_input_ids == self.tokenizer.tht_token_id
        time_token_mask = new_input_ids == self.tokenizer.time_token_id

        special_token_masks = bot_token_mask | tht_token_mask | eot_token_mask | time_token_mask

        new_input_ids[special_token_masks] = self.tokenizer.pad_token_id
        inputs_embeds = self.text_llama.get_input_embeddings()(new_input_ids).to(torch.bfloat16)

        if cfg_train:
            inputs_embeds = self.replace_text_before_tht_start(inputs_embeds, new_input_ids, self.tokenizer.pad_token_id)

        special_image_token_embeddings_weight = self.special_image_token_embeddings.weight

        inputs_embeds[bot_token_mask] = special_image_token_embeddings_weight[0].to(inputs_embeds.dtype)
        inputs_embeds[eot_token_mask] = special_image_token_embeddings_weight[1].to(inputs_embeds.dtype)

        inputs_embeds[tht_token_mask] = thought_token_embeds.reshape(B*N, -1)

        timestep_embeds = self.time_embed.get_t_embed(timestep, dtype=inputs_embeds.dtype).reshape(B, -1)

        inputs_embeds[time_token_mask] = timestep_embeds

        outputs = self.text_llama(
                        inputs_embeds=inputs_embeds,
                        attention_mask=attention_mask,
                        #position_ids=position_ids,
                        output_hidden_states=True,
                        return_dict=True,
                        use_cache=False,
                    )
        outputs_hidden_states = outputs.hidden_states[-1] 
        model_pred = outputs_hidden_states[tht_token_mask].reshape(B, N, -1)
        t_embedder_pred = self.t_embedder.get_t_embed(timestep, model_pred.dtype).reshape(B, -1)
        model_pred = self.output_projection(model_pred, t_embedder_pred)

        if infer_cfg > 1:
            null_inputs_embeds = inputs_embeds.clone()
            
            for b in range(B):
                tht_positions = (new_input_ids[b] == self.tokenizer.bot_token_id).nonzero(as_tuple=True)[0]
                
                # If there's at least one occurrence, take the first one
                if len(tht_positions) > 0:
                    first_video_idx = tht_positions[0].item()

                    # Replace from index 1 up to (but not including) first_video_idx
                    # We skip index 0 in order to exclude the BOS token
                    if first_video_idx > 1: 
                        null_inputs_embeds[b, 1:first_video_idx] = self.fake_input_embed

            null_outputs = self.text_llama(
                            inputs_embeds=null_inputs_embeds,
                            attention_mask=attention_mask,
                            #position_ids=position_ids,
                            output_hidden_states=True,
                            return_dict=True,
                            use_cache=False,
                        )
            outputs_hidden_states = null_outputs.hidden_states[-1] 
            null_model_pred = outputs_hidden_states[tht_token_mask].reshape(B, N, -1)
            null_model_pred = self.output_projection(null_model_pred)

            model_pred = null_model_pred + infer_cfg * (model_pred - null_model_pred)

        return model_pred    


    def forward(self,
        input,
        gt_solutions,
        reasoning_text,
        input_ids_q,
        output_ids,
        #labels_ans,
        #answer,
        ):

        """
        Forward pass with optional gradient checkpointing.
        """
        if self.gradient_checkpointing:
            # Use gradient checkpointing for memory savings
            return checkpoint(self._forward, 
                              gt_solutions,
                            reasoning_text,
                            input_ids_q,
                            output_ids,
                            #labels_ans,
                            #answer,
                            )
        else:
            return self._forward(
                    gt_solutions,
                    reasoning_text,
                    input_ids_q,
                    output_ids,
                    #labels_ans,
                    #answer,
                    )


    def _forward(
        self,
        gt_solutions,
        reasoning_text,
        input_ids_q,
        output_ids,
        #labels_ans,
        #answer,
        ):
        
        # if not self.prev_icae_params:  # empty dict
        #     for name, param in self.autoencoder.named_parameters():
        #         # store a clone of the initial param
        #         self.prev_icae_params[name] = param.detach().clone()

        # # 2) Compare the current param values to the old ones
        # #    If they differ (beyond floating-point tolerance), they've been updated
        # from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        # import torch.distributed as dist

        # if not dist.is_initialized() or dist.get_rank() == 0:
        #     with FSDP.summon_full_params(self.autoencoder, writeback=False):
        #         for name, param in self.autoencoder.named_parameters():
        #             current_val = param.detach()
        #             prev_val = self.prev_icae_params.get(name)
                    
        #             if prev_val is not None and current_val.shape == prev_val.shape:
        #                 if not torch.allclose(current_val, prev_val, atol=1e-8, rtol=1e-6):
        #                     print(f"[❌] VAE param changed: {name}")
        #                 else:
        #                     print(f"[✅] VAE param unchanged: {name}")
        #             else:
        #                 print(f"[SKIP] Shape mismatch or missing prev_val for: {name}")

        # # 3) Update self.prev_icae_params for the *next* iteration
        # for name, param in self.autoencoder.named_parameters():
        #     self.prev_icae_params[name] = param.detach().clone()

        #trainable_params = [n for n, p in self.autoencoder.named_parameters() if p.requires_grad]
        #print(f"Trainable AE parameters:\n{trainable_params}")

        #feeze_module(self.autoencoder)
        #trainable_params = [n for n, p in self.autoencoder.named_parameters() if p.requires_grad]
        #print(f"AFTER Trainable AE parameters:\n{trainable_params}")

        bs = input_ids_q.shape[0]
        device = input_ids_q.device

        #with torch.distributed.fsdp.FullyShardedDataParallel.summon_full_params(self.autoencoder):  # required to overcome to the error "'weight' must be 2-D"
        with torch.no_grad():
            thought_tokens = self.autoencoder._compress(output_ids) #, return_sample="sample")  
            # batch_mean = thought_tokens.mean()                 
            # batch_std = thought_tokens.std()
            # print(f"before rescale!! shift_factor={batch_mean}, batch_std={batch_std}, scale_factor={1.0 / (batch_std)}",)
            #thought_tokens = rescale(thought_tokens, self.model_config.ae.shift_factor, self.model_config.ae.scale_factor)

            # batch_mean = thought_tokens.mean()                 
            # batch_std = thought_tokens.std()
            # if random.uniform(0, 1) < 0.3:
            #     print(f"batch_mean={batch_mean}, batch_std={batch_std}")

        thought_tokens = thought_tokens.reshape(bs, -1, self.tht_token_dim).to(dtype=torch.bfloat16, device=device)


        gt_thought_tokens = thought_tokens.repeat(self.diffusion_batch_mul, 1, 1).clone().detach()

        noise = torch.randn(gt_thought_tokens.shape, dtype=gt_thought_tokens.dtype, device=gt_thought_tokens.device)
        timestep = torch.randint(0, 1000, noise.shape[:1]).to(noise.device)
        #timestep = self.noise_scheduler.sample_timesteps(noise.shape[:1], device=noise.device)
        batch_noisy_x = self.noise_scheduler.add_noise(original_samples=gt_thought_tokens, noise=noise, timesteps=timestep).to(device=device)
        timestep = getattr(self.noise_scheduler, "timestep", timestep)


        batch_projected_noisy_x = self.ae_to_latent(batch_noisy_x)

        batch_input_ids_q = input_ids_q.repeat(self.diffusion_batch_mul, 1)

        model_pred = self.prepare_input(input_ids=batch_input_ids_q, thought_token_embeds=batch_projected_noisy_x, timestep=timestep, cfg_train=True)

        if self.noise_scheduler.config.prediction_type == "epsilon":
            model_target = noise.float()

        elif self.noise_scheduler.config.prediction_type == "sample":
            model_target = gt_thought_tokens.float()
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            model_target = self.noise_scheduler.get_velocity(gt_thought_tokens.float(), noise, timestep)
        else:
            model_target = noise.sub(gt_thought_tokens).float()

        diff_loss = nn.functional.mse_loss(model_pred.float(), model_target.float())

        # LM Loss

        # q_embeds = self.text_llama.get_input_embeddings()(input_ids_q)

        # special_image_token_embeddings_weight = self.special_image_token_embeddings.weight

        # concat_embeds = torch.concat(tensors=[q_embeds, special_image_token_embeddings_weight[0].reshape(bs, 1, -1), thought_tokens, special_image_token_embeddings_weight[1].reshape(bs, 1, -1)], dim=1)

        # answer_embeds = self.text_llama.get_input_embeddings()(labels_ans[:, :-1])

        # teacher_forcing_embeds = torch.cat([concat_embeds, answer_embeds], dim=1)

        # attention_mask = torch.ones(teacher_forcing_embeds.shape[:2], dtype=torch.long, device=answer_embeds.device)

        # outputs = self.text_llama(
        #     attention_mask=attention_mask,
        #     inputs_embeds=teacher_forcing_embeds,
        #     use_cache=False
        # )
        # context_labels = torch.full((concat_embeds.shape[0], concat_embeds.shape[1]), -100, 
        #                             dtype=torch.long, device=concat_embeds.device)
        # answer_labels = labels_ans

        # labels = torch.cat([context_labels[:, 1:], answer_labels], dim=1)

        # logits = outputs.logits  
        # lm_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)

        #loss = 2 * diff_loss + lm_loss
        loss = diff_loss

        #print(loss.dtype, loss.requires_grad, loss.grad_fn)

        return {
            "loss": loss,
            #"lm_loss": lm_loss,
            #"diff_loss": diff_loss,
        }
    
