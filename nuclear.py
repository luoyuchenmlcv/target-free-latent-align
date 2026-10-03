import functools
from torch import nn, vmap
import torch
from torch.func import vjp, jvp, grad
from torch.nn import functional as F
from tqdm import tqdm
import math
from scipy.optimize import root_scalar
import numpy as np

def rgetattr(obj, path):
    return functools.reduce(getattr, path.split("."), obj)
def rhasattr(obj, path):
    try:
        functools.reduce(hasattr, path.split("."), obj)
        return True
    except (AttributeError, TypeError):
        return False



class SlicedModel(nn.Module):
    """
    Run only a contiguous slice of transformer blocks.

    Important:
    - We disable the model's final norm during the sliced forward, because
      intermediate hidden_states from HF models are typically pre-final-norm.
    """

    def __init__(self, model, start_layer, end_layer, layers_name=None):
        super().__init__()
        self.model = model
        self.start_layer = int(start_layer)
        self.end_layer = int(end_layer)

        if self.end_layer < self.start_layer:
            raise ValueError(
                f"end_layer must be >= start_layer, got "
                f"start_layer={self.start_layer}, end_layer={self.end_layer}"
            )

        if layers_name is None:
            if hasattr(self.model, "layers"):
                self.layers_name = "model.layers"
            elif hasattr(self.model, "model"):
                self.layers_name = "model.model.layers"
            else:
                raise ValueError(f"don't know how to get layer list for {type(model)}")
        else:
            self.layers_name = layers_name

        self.layers = rgetattr(self.model, self.layers_name)
        self.layers_name_split = self.layers_name.split(".")
        self.backbone_path = ".".join(self.layers_name_split[:-1])
        self.backbone = rgetattr(self.model, self.backbone_path)

        # Try to infer final norm path from layers path:
        #   model.layers      -> model.norm
        #   model.model.layers -> model.model.norm
        norm_parent = ".".join(self.layers_name_split[:-1])
        self.norm_path = f"{norm_parent}.norm"

    def _has_norm(self):
        try:
            rgetattr(self.model, self.norm_path)
            return True
        except Exception:
            return False

    def reset(self):
        setattr(self.model.config, "num_hidden_layers", self.depth)
        setattr(
            rgetattr(self.model, ".".join(self.layers_name_split[:-1])),
            self.layers_name_split[-1],
            self.L,
        )

        if self.had_norm:
            setattr(
                rgetattr(self.model, ".".join(self.norm_path.split(".")[:-1])),
                self.norm_path.split(".")[-1],
                self.saved_norm,
            )

        for i in range(len(rgetattr(self.model, self.layers_name))):
            layer = rgetattr(self.model, self.layers_name)[i]
            if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "layer_idx"):
                layer.self_attn.layer_idx = i

    def forward(self, h):
        # Save original full model state
        self.L = self.layers
        self.depth = self.model.config.num_hidden_layers

        sliced_layers = self.L[self.start_layer : self.end_layer + 1]
        n_layers = len(sliced_layers)

        # Replace layer list by sliced sub-list
        setattr(
            rgetattr(self.model, ".".join(self.layers_name_split[:-1])),
            self.layers_name_split[-1],
            sliced_layers,
        )
        setattr(self.model.config, "num_hidden_layers", n_layers)

        # Disable final norm temporarily so returned hidden_states[n_layers]
        # matches intermediate hidden states from the original full model.
        self.had_norm = self._has_norm()
        self.saved_norm = None
        if self.had_norm:
            self.saved_norm = rgetattr(self.model, self.norm_path)
            setattr(
                rgetattr(self.model, ".".join(self.norm_path.split(".")[:-1])),
                self.norm_path.split(".")[-1],
                nn.Identity(),
            )

        for i in range(len(rgetattr(self.model, self.layers_name))):
            layer = rgetattr(self.model, self.layers_name)[i]
            if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "layer_idx"):
                layer.self_attn.layer_idx = i

        # hidden_states[0] = inputs_embeds
        # hidden_states[n_layers] = output after final sliced block
        try:
            # Call the decoder backbone directly. Calling the causal-LM wrapper
            # also projects every hidden state through the full vocabulary head,
            # which is unnecessary here and makes vmapped inner steps OOM.
            result = self.backbone(
                inputs_embeds=h,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            ).hidden_states[n_layers]
        finally:
            self.reset()
        return result

        
class DeltaActivations(nn.Module):
    def __init__(self, sliced_model, target_position_indices=slice(-3,None)):
        super().__init__()
        self.sliced_model = sliced_model
        self.device = sliced_model.model.device
        self.target_position_indices = target_position_indices
    def forward(self, theta, x, y):
        '''
        computes average delta in target layer activations as a 
        function of bias theta
        '''
        delta = self.sliced_model(x+theta) - y # batch_size x seq_len x d_model
        delta = delta[:, self.target_position_indices, :]
        return delta.mean(dim=1)



def compute_global_mean_shift_matrix(
    X: torch.Tensor,
    Y: torch.Tensor,
    delta_acts_single,
    theta: torch.Tensor,
    *,
    batch_size: int | None = None,
    factor_batch_size: int = 16,
    theta_chunk_size: int | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Compute the global mean-shift matrix C used by fit_nuclear.

    Args:
        X: Source activations of shape [N, T, d_source].
        Y: Clean target activations of shape [N, T, d_target].
        delta_acts_single: Callable(theta_single, x, y) -> [B, d_target].
        theta: Fixed steering matrix of shape [d_source, K].
        batch_size: Sample chunk size. Defaults to full dataset.
        factor_batch_size: Column chunk size passed into ``vmap`` for each theta block.
        theta_chunk_size: Optional explicit chunk size over the columns of ``theta``.
            When provided, the full C matrix is constructed by concatenating the per-block
            results, but the returned value is exactly the same as evaluating all columns
            in one call.
        device: Device for evaluation. Defaults to delta_acts_single.device when available,
            otherwise X.device.

    Returns:
        C in R^{d_target x K}, defined as (1 / N) * sum_i Delta_i.
    """
    if X.dim() != 3 or Y.dim() != 3:
        raise ValueError(f"Expected X,Y to be rank-3, got X={tuple(X.shape)} Y={tuple(Y.shape)}")
    if theta.dim() != 2:
        raise ValueError(f"Expected theta to be rank-2 [d_source,K], got {tuple(theta.shape)}")
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"Mismatched sample counts: X={tuple(X.shape)} Y={tuple(Y.shape)}")

    if batch_size is None or int(batch_size) <= 0:
        batch_size = int(X.shape[0])
    else:
        batch_size = int(batch_size)

    if device is None:
        device = getattr(delta_acts_single, 'device', X.device)

    theta = theta.to(device=device, dtype=X.dtype)
    if theta_chunk_size is None or int(theta_chunk_size) <= 0:
        theta_chunk_size = int(theta.shape[1])
    else:
        theta_chunk_size = int(theta_chunk_size)

    c_blocks = []
    for s in range(0, theta.shape[1], theta_chunk_size):
        e = min(theta.shape[1], s + theta_chunk_size)
        theta_block = theta[:, s:e]
        
        delta_acts = vmap(
            delta_acts_single,
            in_dims=(1, None, None),
            out_dims=2,
            chunk_size=min(factor_batch_size, theta_block.shape[1]),
        )

        c_sum_block = None
        for b in range(0, X.shape[0], batch_size):
            x = X[b:b + batch_size].to(device)
            y = Y[b:b + batch_size].to(device)
            delta_chunk = delta_acts(theta_block, x, y)
            chunk_sum = delta_chunk.sum(dim=0)
            c_sum_block = chunk_sum if c_sum_block is None else (c_sum_block + chunk_sum)

        c_blocks.append(c_sum_block / float(X.shape[0]))

    return torch.cat(c_blocks, dim=1)


class StreamingAverage:
    """
    Maintains a streaming average of tensors.
    Handles variable batch sizes and arbitrary tensor dimensions.
    """
    def __init__(self):
        self.count = 0
        self.mean = None
    
    def update(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Updates the streaming average with a new batch of data.
        
        Args:
            batch: Tensor of shape (batch_size, dim1, ..., dimk)
                  The first dimension is assumed to be the batch dimension
        
        Returns:
            Current mean after incorporating the new batch
        """
        batch_size = batch.size(0)
        
        if self.mean is None:
            # First batch - initialize mean with the correct shape
            self.mean = batch.mean(dim=0)
            self.count = batch_size
            return self.mean
        
        # Update count
        new_count = self.count + batch_size
        
        # Compute batch mean
        batch_mean = batch.mean(dim=0)
        
        # Update mean using formula:
        # new_mean = old_mean + (batch_mean - old_mean) * (batch_size / new_count)
        self.mean = self.mean + (batch_mean - self.mean) * (batch_size / new_count)
        self.count = new_count
        
        return self.mean
    
    def get_mean(self) -> torch.Tensor:
        """Returns the current mean."""
        if self.mean is None:
            raise ValueError("No data has been processed yet")
        return self.mean
    
    def reset(self):
        """Resets the streaming average."""
        self.count = 0
        self.mean = None




        

class Nuclear():
    def __init__(self, num_factors=512):
        self.num_factors = num_factors
    def _init_rand(self, delta_acts, X, Y):
        print("initializing V,U...")
        # initialize V randomly
        self.V = F.normalize(torch.randn(self.d_source, self.num_factors, dtype = X.dtype, device=self.device), dim=0)
        self.U = F.normalize(torch.randn(self.d_target, self.num_factors, dtype = X.dtype, device=self.device), dim=0)
            

    def _init_similarity(self, X, Y, target_sim=0.01):
        """
        Initialize V: [d, K] with mean cosine similarity ≈ target_sim
        """
        alpha = target_sim ** 0.5  # since sim ≈ alpha^2
        d, K = self.d_source, self.num_factors
        # shared direction
        u = torch.randn(d, device=X.device, dtype=X.dtype)
        u = F.normalize(u, dim=0)
    
        # random orthogonal-ish noise
        eps = torch.randn(d, K, device=X.device, dtype=X.dtype)
        eps = eps - (u.unsqueeze(1) * (u @ eps))  # remove u component
        eps = F.normalize(eps, dim=0)
    
        # combine
        V = alpha * u.unsqueeze(1) + (1 - alpha**2) ** 0.5 * eps
        V = F.normalize(V, dim=0)
    
        return V


    def rank(self, delta_acts_single, X, Y, target_vec=None, batch_size=1, factor_batch_size=16):
        delta_acts = vmap(delta_acts_single, in_dims=(1,None,None), out_dims=2,
                  chunk_size=factor_batch_size)
        num_samples = X.shape[0]
        Delta_avg = StreamingAverage()
        with torch.no_grad():
            for b in tqdm(range(0, num_samples, batch_size)):
                x = X[b:b+self.batch_size,:,:].to(self.device)
                y = Y[b:b+self.batch_size,:,:].to(self.device)
                Delta_batch = delta_acts(self.input_scale * self.V, x, y)              
                Delta_avg.update(Delta_batch)
            if target_vec is None:
                self.alphas = (Delta_avg.get_mean() * self.U).sum(dim=0)
                K = (self.U.t() @ self.U) * torch.expm1(self.V.t() @ self.V)
                self.alphas = torch.linalg.solve(K.float(), self.alphas.float())
                self.scores = self.alphas.pow(2)
                self.scores, self.indices = torch.sort(self.scores, descending=True)
            else:
                self.scores = Delta_avg.get_mean().t() @ target_vec.to(self.device)
                self.scores, self.indices = torch.sort(self.scores, descending=True)                
        return self.scores, self.indices
  


 
    def fit_nuclear(
        self,
        delta_acts_single,
        X,
        Y,
        batch_size=None,
        factor_batch_size=16,
        init="random",
        d_proj=32,
        input_scale=1.0,
        max_iters=100,
        lr=1e-2,
        weight_decay=0.0,
        verbose=True,
    ):
        """
        Pure nuclear-norm end-to-end fit using the GLOBAL mean-shift matrix.
    
        Objective:
            maximize ||C||_* 
        where
            C = (1 / N) * sum_i Delta_i  in R^{d_target x K}
            Delta_i[:, k] = Delta h_t^{(i)}(v_k)
    
        In addition to the original statistics, this version records:
        - intra_prompt_consistency: for each v_k, average cosine similarity among
          {Delta h_t(x_i, v_k)} across prompts, then averaged over k.
        - per-step V / scaled-V / C snapshots for later batch-steer evaluation.
        - step_stats: a JSON-friendly per-step summary.
        """
    
        assert init in ["random", "jacobian", "similarity"]
    
        self.num_samples, self.seq_len, self.d_source = X.shape
        _, _, self.d_target = Y.shape
        self.factor_batch_size = factor_batch_size
        self.input_scale = input_scale
        self.d_proj = d_proj
        self.device = delta_acts_single.device
    
        if batch_size is None or batch_size <= 0:
            batch_size = self.num_samples
        self.batch_size = int(batch_size)
    
        # delta_acts(theta, x, y) -> [B, d_target, K]
        delta_acts = vmap(
            delta_acts_single,
            in_dims=(1, None, None),
            out_dims=2,
            chunk_size=factor_batch_size,
        )
    
        # ---- initialize V ----
        if init == "random":
            V0 = F.normalize(
                torch.randn(
                    self.d_source,
                    self.num_factors,
                    device=self.device,
                    dtype=X.dtype,
                ),
                dim=0,
            )
        elif init == "similarity":
            V0 = self._init_similarity(X, Y)
        else:
            raise ValueError(f"Unsupported init={init!r}")
    
        self.V_raw = nn.Parameter(V0.clone())
        opt = torch.optim.AdamW([self.V_raw], lr=lr, weight_decay=weight_decay)
    
        self.objective_values = []
        self.nuclear_values = []
        self.mean_shift_avg_norm_values = []
        self.mean_shift_dir_sim_values = []
        self.v_self_sim_values = []
        self.intra_prompt_consistency_values = []
        self.v_history = []
        self.v_scaled_history = []
        self.c_history = []
        self.step_stats = []
    
        def normalize_v(V_raw: torch.Tensor) -> torch.Tensor:
            return F.normalize(V_raw, dim=0)
    
        def mean_offdiag_cos(M: torch.Tensor) -> torch.Tensor:
            K = M.shape[1]
            if K <= 1:
                return M.new_zeros(())
            Mn = F.normalize(M.float(), dim=0)
            S = Mn.T @ Mn
            mask = ~torch.eye(K, device=S.device, dtype=torch.bool)
            return S[mask].mean()
    
        def compute_intra_prompt_consistency(theta: torch.Tensor) -> torch.Tensor:
            vals = []
            for b in range(0, X.shape[0], self.batch_size):
                x = X[b:b + self.batch_size].to(self.device)
                y = Y[b:b + self.batch_size].to(self.device)
                vals.append(delta_acts(theta, x, y).float())
            if not vals:
                return X.new_zeros(())
            delta_all = torch.cat(vals, dim=0)  # [N, d_target, K]
            K = delta_all.shape[-1]
            per_k = []
            for k in range(K):
                z = delta_all[:, :, k]  # [N, d_target]
                if z.shape[0] <= 1:
                    continue
                zn = F.normalize(z, dim=1)
                s = zn @ zn.T
                mask = ~torch.eye(s.shape[0], device=s.device, dtype=torch.bool)
                if mask.any():
                    per_k.append(s[mask].mean())
            if not per_k:
                return delta_all.new_zeros(())
            return torch.stack(per_k).mean()
    
        # ------------------------------------------------------------
        # Save initialization snapshot only.
        # Post-update recording below remains unchanged.
        # This gives:
        #   V_step_0000 = initialization
        #   V_step_0001 = after epoch 0 update
        # ------------------------------------------------------------
        with torch.no_grad():
            V_init_log = normalize_v(self.V_raw).detach()
            theta_init_log = self.input_scale * V_init_log
            C_init_log = compute_global_mean_shift_matrix(
                X,
                Y,
                delta_acts_single,
                theta_init_log,
                batch_size=self.batch_size,
                factor_batch_size=factor_batch_size,
                theta_chunk_size=factor_batch_size,
                device=self.device,
            ).detach().float()

            init_nuclear = float(torch.linalg.matrix_norm(C_init_log.float(), ord="nuc").item())
            init_mean_shift_avg_norm = float(C_init_log.norm(dim=0).mean().item())
            init_mean_shift_dir_sim = float(mean_offdiag_cos(C_init_log).item())
            init_v_self_sim = float(mean_offdiag_cos(V_init_log).item())
            init_intra_prompt_consistency = float(compute_intra_prompt_consistency(theta_init_log).item())

            self.objective_values.append(init_nuclear)
            self.nuclear_values.append(init_nuclear)
            self.mean_shift_avg_norm_values.append(init_mean_shift_avg_norm)
            self.mean_shift_dir_sim_values.append(init_mean_shift_dir_sim)
            self.v_self_sim_values.append(init_v_self_sim)
            self.intra_prompt_consistency_values.append(init_intra_prompt_consistency)

            self.v_history.append(V_init_log.detach().cpu().clone())
            self.v_scaled_history.append(theta_init_log.detach().cpu().clone())
            self.c_history.append(C_init_log.detach().cpu().clone())
            self.step_stats.append({
                "step": 0,
                "epoch": -1,
                "state": "initialization",
                "objective": init_nuclear,
                "nuclear": init_nuclear,
                "mean_shift_avg_norm": init_mean_shift_avg_norm,
                "mean_shift_dir_sim": init_mean_shift_dir_sim,
                "v_self_sim": init_v_self_sim,
                "intra_prompt_consistency": init_intra_prompt_consistency,
            })

        if verbose:
            print("training fit_nuclear (global C)...")

        for epoch in range(max_iters):
            opt.zero_grad(set_to_none=True)

            # The nuclear norm couples all factor columns, but retaining the
            # transformer graph for every column at once is prohibitively
            # expensive for 7B models.  Compute d(-||C||_*)/dC first, then
            # recompute small column blocks and immediately backpropagate their
            # vector-Jacobian products.  This is the same first-order gradient
            # as a single loss.backward(), with bounded activation memory.
            with torch.no_grad():
                V_value = normalize_v(self.V_raw)
                theta_value = self.input_scale * V_value
                C_value = compute_global_mean_shift_matrix(
                    X,
                    Y,
                    delta_acts_single,
                    theta_value,
                    batch_size=self.batch_size,
                    factor_batch_size=factor_batch_size,
                    theta_chunk_size=factor_batch_size,
                    device=self.device,
                ).float()
                U_value, S_value, Vh_value = torch.linalg.svd(
                    C_value, full_matrices=False
                )
                nuclear = S_value.sum()
                loss_grad_C = -(U_value @ Vh_value)

            for start in range(0, self.num_factors, factor_batch_size):
                end = min(self.num_factors, start + factor_batch_size)
                V_block = normalize_v(self.V_raw[:, start:end])
                theta_block = self.input_scale * V_block
                C_block = compute_global_mean_shift_matrix(
                    X,
                    Y,
                    delta_acts_single,
                    theta_block,
                    batch_size=self.batch_size,
                    factor_batch_size=factor_batch_size,
                    theta_chunk_size=factor_batch_size,
                    device=self.device,
                )
                (C_block.float() * loss_grad_C[:, start:end]).sum().backward()
            opt.step()
    
            with torch.no_grad():
                V_log = normalize_v(self.V_raw).detach()
                theta_log = self.input_scale * V_log
                C_log = compute_global_mean_shift_matrix(
                    X,
                    Y,
                    delta_acts_single,
                    theta_log,
                    batch_size=self.batch_size,
                    factor_batch_size=factor_batch_size,
                    theta_chunk_size=factor_batch_size,
                    device=self.device,
                ).detach().float()
    
                mean_shift_avg_norm = C_log.norm(dim=0).mean()
                mean_shift_dir_sim = mean_offdiag_cos(C_log)
                v_self_sim = mean_offdiag_cos(V_log)
                intra_prompt_consistency = compute_intra_prompt_consistency(theta_log)
    
                epoch_obj = float(nuclear.item())
                epoch_nuc = float(nuclear.item())
                epoch_mean_shift_avg_norm = float(mean_shift_avg_norm.item())
                epoch_mean_shift_dir_sim = float(mean_shift_dir_sim.item())
                epoch_v_self_sim = float(v_self_sim.item())
                epoch_intra_prompt_consistency = float(intra_prompt_consistency.item())
    
                self.objective_values.append(epoch_obj)
                self.nuclear_values.append(epoch_nuc)
                self.mean_shift_avg_norm_values.append(epoch_mean_shift_avg_norm)
                self.mean_shift_dir_sim_values.append(epoch_mean_shift_dir_sim)
                self.v_self_sim_values.append(epoch_v_self_sim)
                self.intra_prompt_consistency_values.append(epoch_intra_prompt_consistency)
    
                self.v_history.append(V_log.detach().cpu().clone())
                self.v_scaled_history.append((theta_log).detach().cpu().clone())
                self.c_history.append(C_log.detach().cpu().clone())
                self.step_stats.append({
                    "step": int(epoch + 1),
                    "epoch": int(epoch),
                    "state": "post_update",
                    "objective": epoch_obj,
                    "nuclear": epoch_nuc,
                    "mean_shift_avg_norm": epoch_mean_shift_avg_norm,
                    "mean_shift_dir_sim": epoch_mean_shift_dir_sim,
                    "v_self_sim": epoch_v_self_sim,
                    "intra_prompt_consistency": epoch_intra_prompt_consistency,
                })
    
                if verbose:
                    print(
                        f"[fit_nuclear] epoch={epoch:03d} "
                        f"nuc={epoch_nuc:.6f} "
                        f"mean_shift_avg_norm={epoch_mean_shift_avg_norm:.6f} "
                        f"mean_shift_dir_sim={epoch_mean_shift_dir_sim:.6f} "
                        f"v_self_sim={epoch_v_self_sim:.6f} "
                        f"intra_prompt_consistency={epoch_intra_prompt_consistency:.6f}"
                    )
    
        with torch.no_grad():
            V_final = normalize_v(self.V_raw).detach()
            theta_final = self.input_scale * V_final
    
            C_final = compute_global_mean_shift_matrix(
                X,
                Y,
                delta_acts_single,
                theta_final,
                batch_size=self.batch_size,
                factor_batch_size=factor_batch_size,
                theta_chunk_size=factor_batch_size,
                device=self.device,
            ).float()
            U_final, S_final, _ = torch.linalg.svd(C_final, full_matrices=False)
    
            k_eff = min(self.num_factors, U_final.shape[1])
            self.U = U_final[:, :k_eff].to(dtype=X.dtype)
            self.V = V_final[:, :k_eff].to(dtype=X.dtype)
            self.final_mean_drift = C_final[:, :k_eff].to(dtype=X.dtype)
            self.final_singular_values = S_final[:k_eff].to(dtype=X.dtype)
    
            self.final_mean_shift_avg_norm = float(C_final.norm(dim=0).mean().item())
            self.final_mean_shift_dir_sim = float(mean_offdiag_cos(C_final).item())
            self.final_v_self_sim = float(mean_offdiag_cos(V_final).item())
            self.final_intra_prompt_consistency = float(compute_intra_prompt_consistency(theta_final).item())
    
            if verbose:
                print(
                    f"[fit_nuclear][final] "
                    f"mean_shift_avg_norm={self.final_mean_shift_avg_norm:.6f} "
                    f"mean_shift_dir_sim={self.final_mean_shift_dir_sim:.6f} "
                    f"v_self_sim={self.final_v_self_sim:.6f} "
                    f"intra_prompt_consistency={self.final_intra_prompt_consistency:.6f}"
                )
    
        return self.U, self.V
    
