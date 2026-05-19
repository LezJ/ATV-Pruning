import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
import math
import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from tqdm import tqdm

from utils import print_time, model_setup_and_record_attributes, model_reset
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb as apply_rotary_pos_emb_llama
from transformers.models.llama.modeling_llama import repeat_kv as repeat_kv_llama
from transformers.models.qwen2_vl.modeling_qwen2_vl import \
    apply_multimodal_rotary_pos_emb as apply_multimodal_rotary_pos_emb_qwen
from transformers.models.qwen2.modeling_qwen2 import repeat_kv as repeat_kv_qwen


def normalize_rank(scores: torch.Tensor):
    """
    Rank-based normalization to [0,1].
    Args:
        scores: (N,) tensor
    """
    ranks = torch.argsort(torch.argsort(scores))  # 0..N-1
    normed = ranks.float() / (scores.numel() - 1 + 1e-8)
    return normed.to(scores.dtype)


def get_module_recursive(base, module_to_process):
    if module_to_process == "":
        return base

    splits = module_to_process.split(".")
    now = splits.pop(0)
    rest = ".".join(splits)
    base = getattr(base, now)

    return get_module_recursive(base, rest)


def compute_avg_cos_dist(x):
    x_norm = F.normalize(x, p=2, dim=-1)
    sim = x_norm @ x_norm.T
    n = x.size(0)
    sim.fill_diagonal_(0.0)
    avg_sim = sim.sum(dim=1) / (n - 1)
    avg_dist = 1 - avg_sim

    return avg_dist


def find_layers(module, layers=[nn.Linear], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


class WrappedWanda:
    """
    Wanda layer-wise activation statistics.
    """

    def __init__(self, layer, layer_id=0, layer_name="none"):
        self.layer = layer
        self.dev = self.layer.weight.device
        self.rows = layer.weight.data.shape[0]
        self.columns = layer.weight.data.shape[1]

        self.scaler_row = torch.zeros((self.columns), device=self.dev)
        self.nsamples = 0

        self.layer_id = layer_id
        self.layer_name = layer_name

    def add_batch(self, inp, out, image_mask=None, score=None, selected_idxs=None):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]

        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()

        self.scaler_row *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp

        inp = inp.type(torch.float32)
        self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2 / self.nsamples


class WrappedATV:
    """
    ATV-Pruning layer-wise activation statistics.
    """

    def __init__(self, layer, layer_id=0, layer_name="none"):
        self.layer = layer
        self.dev = self.layer.weight.device
        self.rows = layer.weight.data.shape[0]
        self.columns = layer.weight.data.shape[1]

        self.scaler_row = torch.zeros((self.columns), device=self.dev)
        self.nsamples = 0

        self.layer_id = layer_id
        self.layer_name = layer_name

    # selected_idxs: selected top salient visual token indexes
    def add_batch(self, inp, out, image_mask=None, score=None, selected_idxs=None):
        assert image_mask is not None
        assert selected_idxs is not None
        inp = inp[0]
        image_mask = image_mask[0]
        tmp = 1

        text_token = inp[~image_mask]
        image_token = inp[image_mask][selected_idxs]
        inp = torch.cat((text_token, image_token), dim=0)
        # print('$$$$$', text_token.shape, text_token.shape, inp.shape)

        if isinstance(self.layer, nn.Linear):
            inp = inp.t()

        self.scaler_row *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp

        inp = inp.type(torch.float32)
        self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2 / self.nsamples


# For TAMP only
class AdaptiveMultimodalInputActivation:
    """
    TAMP's AMIA layer-wise activation statistics.
    """

    def __init__(self, layer, layer_id=0, layer_name="none"):
        self.layer = layer
        self.dev = self.layer.weight.device
        self.rows = layer.weight.data.shape[0]
        self.columns = layer.weight.data.shape[1]
        self.scaler_row = torch.zeros((self.columns), device=self.dev)
        self.nsamples = 0
        self.layer_id = layer_id
        self.layer_name = layer_name

    def gaussian_rbf(self, X, Y, sigma=1.0):
        X_norm = (X ** 2).sum(dim=1).view(-1, 1)
        Y_norm = (Y ** 2).sum(dim=1).view(1, -1)
        pairwise_dists = X_norm + Y_norm - 2.0 * torch.mm(X, Y.T)

        K = torch.exp(-pairwise_dists / (2 * sigma ** 2))
        return K

    def cos_pairwise_density(self, embeddings, image_mask):
        v_embeddings = embeddings[image_mask]
        l_embeddings = embeddings[~image_mask]

        v_distances = torch.mm(v_embeddings, v_embeddings.T)
        v_upper_triangular = v_distances.triu(diagonal=1)  # Only take upper triangular distances (no duplicates)
        v_mean_dist = v_upper_triangular[v_upper_triangular > 0].mean().item()

        l_distances = torch.mm(l_embeddings, l_embeddings.T)
        l_upper_triangular = l_distances.triu(diagonal=1)  # Only take upper triangular distances (no duplicates)
        l_mean_dist = l_upper_triangular[l_upper_triangular > 0].mean().item()

        vl_dist = torch.mm(v_embeddings, l_embeddings.T)
        vl_mean_dist = vl_dist.mean().item()

        return (v_mean_dist + l_mean_dist + vl_mean_dist) / 3

    def add_batch(self, inp, out, image_mask=None, score=None, selected_idxs=None):
        assert image_mask.any() and (~image_mask).any(), "image_mask must contain both image and text tokens"

        if len(out.shape) == 3:
            out = out.reshape((-1, out.shape[-1]))
        out = out.type(torch.float32)

        out = F.normalize(out, dim=-1)
        # Compute density
        density = self.cos_pairwise_density(out, image_mask=image_mask[0])
        # print(out)

        # Initialize graph
        distances = 1 - torch.mm(out, out.T)

        num_neigh = 3
        knn_indices = torch.topk(distances, k=num_neigh + 1, largest=False).indices[:, 1:]
        # forward message pass
        neigh_dist = torch.exp(-torch.gather(distances, dim=1, index=knn_indices) * 1.0) * score[knn_indices]
        graph_score = score + neigh_dist.sum(dim=-1)

        K = self.gaussian_rbf(out, out)
        # Iterative selection
        selected_indices = set()

        while True:
            selected = torch.argmax(graph_score)
            neighbors = knn_indices[selected]
            graph_score[neighbors] = graph_score[neighbors] - torch.exp(
                -distances[selected, neighbors] * 0.2) * torch.maximum(graph_score[selected],
                                                                       torch.zeros_like(graph_score[selected]))
            selected_indices.add(selected.item())

            graph_score[torch.tensor(list(selected_indices))] = torch.min(graph_score) - 1

            K_XX = K.mean()
            temp_select = torch.tensor(list(selected_indices))
            K_XY = K[:, temp_select].mean()
            K_YY = K[temp_select, :][:, temp_select].mean()
            MMD2 = K_XX + K_YY - 2 * K_XY

            # print('$$$', MMD2, torch.sqrt(torch.tensor(1 - density)) * 0.1)
            if MMD2 < torch.sqrt(torch.tensor(1 - density)) * 0.1:
                break

        # score mask
        score_mask = torch.zeros_like(score, dtype=torch.bool)
        score_mask[torch.tensor(list(selected_indices))] = True

        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)

        inp = inp[score_mask[None, :]]
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        inp = inp.t()
        # Since we don't zero-pad the endding tokens
        tmp = inp.shape[1]
        self.scaler_row *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp

        inp = inp.type(torch.float32)
        self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2 / self.nsamples


class ActivationAwarePruner:
    pruner_name = "activation_aware_pruner"

    def __init__(
            self,
            model,
            data_loader,
            sparsity_ratio,
            pruning_method,
            num_samples=128,
            alpha=1,
            prune_n=0,
            prune_m=0,
            model_prefix='language_model',
            module_to_process='language_model.layers',
            **kwargs,
    ):
        self.model = model
        self.data_loader = data_loader
        self.sparsity_ratio = sparsity_ratio
        self.pruning_method = pruning_method
        self.num_samples = num_samples
        self.alpha = alpha
        self.prune_n = prune_n
        self.prune_m = prune_m
        self.model_prefix = model_prefix
        self.module_to_process = module_to_process

    def forward_to_cache(self, model, batch):
        return model(**batch)

    def prepare_calibration_input_encoder_qwen2vl(self, model, dataloader, model_prefix, n_samples, module_to_process):
        use_cache = getattr(model, model_prefix).config.use_cache
        getattr(model, model_prefix).config.use_cache = False
        print(model)

        layers = get_module_recursive(model, module_to_process)
        dtype = next(iter(model.parameters())).dtype

        inps = []
        caches = []
        keys_to_cache = ["attention_mask", "position_ids", "cache_position"]
        image_masks = []

        class Catcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, hidden_states, **kwargs):
                inps.append(hidden_states)
                cache = {k: kwargs[k] for k in keys_to_cache if k in kwargs}
                caches.append(cache)
                raise ValueError  # stop forward to avoid full compute

            def __getattr__(self, name):
                try:
                    return super().__getattr__(name)
                except AttributeError:
                    return getattr(self.module, name)

        layers[0] = Catcher(layers[0])

        total_samples = 0
        for batch in dataloader:
            batch["input_ids"] = batch["input_ids"].to(device="cuda", non_blocking=True)
            batch["attention_mask"] = batch["attention_mask"].to(device="cuda", non_blocking=True)
            batch["pixel_values"] = batch["pixel_values"].to(dtype=torch.float16, device="cuda", non_blocking=True)
            batch["image_grid_thw"] = batch["image_grid_thw"].to(device="cuda", non_blocking=True)

            if total_samples >= n_samples:
                break
            total_samples += batch["input_ids"].size(0)

            try:
                self.forward_to_cache(model, batch)
            except ValueError:
                input_ids = batch["input_ids"]

                IMAGE_TOKEN_ID = 151655
                IMAGE_TOKEN_IDS = [IMAGE_TOKEN_ID]

                mask = torch.zeros_like(input_ids, dtype=torch.bool)
                for tid in IMAGE_TOKEN_IDS:
                    mask |= (input_ids == tid)

                image_masks.append(mask)  # (seq_len,)
                pass

        layers[0] = layers[0].module
        outs = [None] * len(inps)
        getattr(model, model_prefix).config.use_cache = use_cache

        return inps, outs, caches, image_masks

    @print_time
    def _prune(self):

        use_cache = getattr(self.model, self.model_prefix).config.use_cache
        getattr(self.model, self.model_prefix).config.use_cache = False

        with torch.no_grad():
            inps, outs, caches, image_masks = self.prepare_calibration_input_encoder_qwen2vl(
                self.model, self.data_loader, self.model_prefix, self.num_samples, self.module_to_process
            )

        # print(
        #     '$$$ prepare_calibration_input_encoder', '\n',
        #     'inps[0].shape:', inps[0].shape, '\n',
        #     # 'outs[0].shape:', outs[0].shape, '\n',  is None list
        #     'caches[0][attention_mask].shape:', caches[0]['attention_mask'], '\n',
        #     'caches[0][position_ids]:', caches[0]['position_ids'], '\n',
        #     'caches[1][position_ids]:', caches[1]['position_ids'], '\n',
        #     'caches[0][cache_position]', caches[0]['cache_position'], '\n',
        #     'image_masks[0] is True:', torch.nonzero(image_masks[0]), '\n',
        #     'image_masks[1] is True:', torch.nonzero(image_masks[1]), '\n',
        # )

        n_samples = min(self.num_samples, len(inps))
        print('$$$ n_samples:', n_samples, len(inps))
        layers = get_module_recursive(self.model, self.module_to_process)

        for i in range(len(layers)):
            layer = layers[i]
            subset = find_layers(layer)
            # print('$$$ subset keys:', subset.keys())

            if i == 0:
                # scores for TAMP
                scores = [None] * len(inps)
                # selected top salient visual token indexes for ATV-Pruning
                selected_idxs = [None] * len(inps)

            for j in tqdm(range(n_samples)):
                with torch.no_grad():
                    cache_pos = caches[j].get("cache_position", None)
                    rotary_emb = getattr(layer.self_attn, "rotary_emb", None)
                    if cache_pos is not None and rotary_emb is not None:
                        caches[j]["position_embeddings"] = rotary_emb(inps[j], cache_pos.unsqueeze(0).unsqueeze(0))

            # Casual Attention map calculation
            if self.pruning_method == 'tamp':
                for j in range(n_samples):
                    with (torch.no_grad()):
                        hid_state = layer.input_layernorm(inps[j])
                        bsz, q_len, _ = hid_state.size()

                        q_state = layer.self_attn.q_proj(hid_state)
                        k_state = layer.self_attn.k_proj(hid_state)
                        v_state = layer.self_attn.v_proj(hid_state)

                        q_state = q_state.view(bsz, q_len, layer.self_attn.num_heads,
                                               layer.self_attn.head_dim).transpose(1, 2)
                        k_state = k_state.view(bsz, q_len, layer.self_attn.num_key_value_heads,
                                               layer.self_attn.head_dim).transpose(1, 2)
                        v_state = v_state.view(bsz, q_len, layer.self_attn.num_key_value_heads,
                                               layer.self_attn.head_dim).transpose(1, 2)

                        if "llama" in self.model.config.model_type:
                            cos, sin = layer.self_attn.rotary_emb(v_state, caches[j]['position_ids'])
                            q_state, k_state = apply_rotary_pos_emb_llama(q_state, k_state, cos, sin)
                            k_state = repeat_kv_llama(k_state, layer.self_attn.num_key_value_groups)
                            v_state = repeat_kv_llama(v_state, layer.self_attn.num_key_value_groups)

                        elif "qwen" in self.model.config.model_type:
                            cos, sin = caches[j]["position_embeddings"]

                            rope_params = (
                                layer.self_attn.config.rope_parameters
                                if hasattr(layer.self_attn.config, "rope_parameters")
                                else layer.self_attn.config.rope_scaling
                            )

                            mrope_section = rope_params.get("mrope_section", None)

                            q_state, k_state = apply_multimodal_rotary_pos_emb_qwen(
                                q_state, k_state, cos, sin, mrope_section
                            )

                            k_state = repeat_kv_qwen(k_state, layer.self_attn.num_key_value_groups)
                            v_state = repeat_kv_qwen(v_state, layer.self_attn.num_key_value_groups)

                        else:
                            raise ValueError(f"Attention not defined for {self.model.config.model_type}")

                        attn_weights = torch.matmul(q_state, k_state.transpose(2, 3)) / math.sqrt(
                            layer.self_attn.head_dim)

                        def update_causal_mask(attention_mask, input_tensor, cache_position):
                            dtype, device = input_tensor.dtype, input_tensor.device

                            sequence_length = input_tensor.shape[1]
                            target_length = attention_mask.shape[-1]

                            min_dtype = torch.finfo(dtype).min

                            causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype,
                                                     dtype=dtype, device=device)
                            causal_mask = torch.triu(causal_mask, diagonal=1)
                            causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
                            causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)

                            causal_mask = causal_mask.clone()
                            mask_length = attention_mask.shape[-1]
                            padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
                            padding_mask = padding_mask == 0
                            causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                                padding_mask, min_dtype
                            )

                            return causal_mask

                        # Causal mask
                        if self.model.config._attn_implementation in ['flash_attention_2', 'sdpa']:
                            temp = self.model.config._attn_implementation
                            self.model.config._attn_implementation = 'eager'
                            attention_mask = torch.ones((hid_state.shape[:2]), dtype=torch.bool,
                                                        device=hid_state.device)
                            causal_mask = update_causal_mask(attention_mask, hid_state, caches[j]['cache_position'])
                            self.model.config._attn_implementation = temp
                        else:
                            causal_mask = caches[j]['attention_mask']

                        if causal_mask is not None:
                            causal_mask = causal_mask[:, :, :, : k_state.shape[-2]]
                            attn_weights = attn_weights + causal_mask

                        # upcast attention to fp32 when applying softmax activation
                        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
                            q_state.dtype)
                        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
                        scores[j] = attn_weights.mean(dim=1).squeeze()[-1, :]

            if self.pruning_method == 'atv':
                cos_dist_ls = []
                for j in range(n_samples):
                    with torch.no_grad():
                        outs[j] = layer(inps[j], image_token_mask=image_masks[j], **caches[j])[0]
                    x_in = inps[j][image_masks[j]]
                    x_out = outs[j][image_masks[j]]
                    cos_sim = F.cosine_similarity(x_in, x_out, dim=-1)
                    cos_dist = 1 - cos_sim
                    cos_dist_ls.append(cos_dist)

                cos_dist_avg = torch.cat(cos_dist_ls).mean().item()
                # print(cos_dist_avg)

                for j in range(n_samples):
                    num_text_tokens = (~image_masks[j]).sum().item()
                    num_img_tokens = (image_masks[j]).sum().item()
                    k = round(min(1, self.alpha * cos_dist_avg) * num_text_tokens)
                    k = min(num_img_tokens, k)
                    selected_idxs[j] = torch.topk(cos_dist_ls[j], k=k).indices.sort().values

            wrapped_layers = {}
            for name in subset:
                if self.pruning_method == 'wanda':
                    wrapped_layers[name] = WrappedWanda(subset[name])
                elif self.pruning_method == 'tamp':
                    wrapped_layers[name] = AdaptiveMultimodalInputActivation(subset[name])
                elif self.pruning_method == 'atv':
                    wrapped_layers[name] = WrappedATV(subset[name])

            def add_batch(name):
                def tmp(_, inp, out):
                    wrapped_layers[name].add_batch(
                        inp=inp[0].data, out=out.data, image_mask=image_masks[j],
                        score=scores[j], selected_idxs=selected_idxs[j],
                    )

                return tmp

            handles = []
            for name in wrapped_layers:
                handles.append(subset[name].register_forward_hook(add_batch(name)))

            for j in tqdm(range(n_samples)):
                with torch.no_grad():
                    outs[j] = layer(inps[j], image_token_mask=image_masks[j], **caches[j])[0]

            for h in handles:
                h.remove()

            for name in subset:
                # structured n:m sparsity
                if self.prune_n != 0:
                    print(
                        f"pruning {self.model_prefix} layer {i} {name} at structured {self.prune_n}:{self.prune_m} sparsity")
                    for ii in range(W_metric.shape[1]):
                        if ii % self.prune_m == 0:
                            tmp = W_metric[:, ii:(ii + self.prune_m)].float()
                            W_mask.scatter_(1, ii + torch.topk(tmp, self.prune_n, dim=1, largest=False)[1], True)
                # unstructured pruning
                else:
                    W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(
                        wrapped_layers[name].scaler_row.reshape((1, -1)))
                    # setattr(subset[name].weight, "importance_score", W_metric.cpu().abs().mean().item())
                    W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
                    sort_res = torch.sort(W_metric, dim=-1, stable=True)
                    print(f"pruning {self.model_prefix} layer {i} {name} at unstructured {self.sparsity_ratio} sparsity")
                    indices = sort_res[1][:, :int(W_metric.shape[1] * self.sparsity_ratio)]
                    W_mask.scatter_(1, indices, True)

                subset[name].weight.data[W_mask] = 0

            for j in range(n_samples):
                with torch.no_grad():
                    outs[j] = layer(inps[j], image_token_mask=image_masks[j], **caches[j])[0]
            inps, outs = outs, inps

        getattr(self.model, self.model_prefix).config.use_cache = use_cache
        # del inps, outs, caches
        torch.cuda.empty_cache()
        gc.collect()
        return self.model

    @print_time
    def prune(self, importance_scores=None, keep_indices_or_masks=None):
        print('$$$ Pruner prune method starts.')

        dtype_record, requires_grad_record, device = model_setup_and_record_attributes(self.model)

        self.model = self._prune()

        model_reset(self.model, dtype_record, requires_grad_record, device)

        print('$$$ Pruner prune method ends.')

        return self.model
