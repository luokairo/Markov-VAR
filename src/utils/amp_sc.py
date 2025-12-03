import math
from typing import List, Optional, Tuple, Union

import torch


class NullCtx:
    def __enter__(self):
        pass
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class AmpOptimizer:
    def __init__(
        self,
        mixed_precision: int,
        optimizer: torch.optim.Optimizer, names: List[str], paras: List[torch.nn.Parameter],
        grad_clip: float, n_gradient_accumulation: int = 1,
    ):
        self.enable_amp = mixed_precision > 0
        self.using_fp16_rather_bf16 = mixed_precision == 1
        
        if self.enable_amp:
            self.amp_ctx = torch.autocast('cuda', enabled=True, dtype=torch.float16 if self.using_fp16_rather_bf16 else torch.bfloat16, cache_enabled=True)
            self.scaler = torch.amp.GradScaler('cuda', init_scale=2. ** 11, growth_interval=1000) if self.using_fp16_rather_bf16 else None
        else:
            self.amp_ctx = NullCtx()
            self.scaler = None
        
        self.optimizer, self.names, self.paras = optimizer, names, paras   # paras have been filtered so everyone requires grad
        self.grad_clip = grad_clip
        self.early_clipping = self.grad_clip > 0 and not hasattr(optimizer, 'global_grad_norm')
        self.late_clipping = self.grad_clip > 0 and hasattr(optimizer, 'global_grad_norm')
        
        self.r_accu = 1 / n_gradient_accumulation   # r_accu == 1.0 / n_gradient_accumulation
    
    def backward_clip_step(
        self, stepping: bool, loss: torch.Tensor,
    ) -> Tuple[Optional[Union[torch.Tensor, float]], Optional[float]]:

        import torch.distributed as dist

        loss = loss.mul(self.r_accu)
        orig_norm = scaler_sc = None

        # === (1) 检查 loss 是否异常 ===
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"[NaN LOSS DETECTED] loss={loss.item()}", flush=True)
            loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)

        # === (2) backward ===
        try:
            if self.scaler is not None:
                self.scaler.scale(loss).backward(retain_graph=False, create_graph=False)
            else:
                loss.backward(retain_graph=False, create_graph=False)
        except Exception as e:
            print("[Backward Error]", e, flush=True)
            raise e

        # === (3) 检查梯度中 NaN/Inf ===
        grad_nan_local = False
        max_grad = 0.0
        for n, p in zip(self.names, self.paras):
            if p.grad is None:
                continue
            # 检测 NaN / Inf
            if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                print(f"[NaN/INF GRADIENT] {n}", flush=True)
                grad_nan_local = True
            # 检查梯度是否过大（仅在异常时打印）
            with torch.no_grad():
                gnorm = p.grad.norm(2).item()
                max_grad = max(max_grad, gnorm)
                if gnorm > 1e4:  # 阈值可调
                    print(f"[LARGE GRADIENT] {n} grad_norm={gnorm:.3e}", flush=True)

        # === (3.1) 全局同步 NaN 状态 ===
        grad_nan_flag = torch.tensor(float(grad_nan_local), device=loss.device)
        if dist.is_initialized():
            dist.all_reduce(grad_nan_flag, op=dist.ReduceOp.SUM)

        if grad_nan_flag.item() > 0:
            # 所有 rank 一起跳过本 step
            if not grad_nan_local:  # 只让一个 rank 打印即可，减少日志重复
                print(f"[NaN sync-skip] global NaN detected, skipping step.", flush=True)
            self.optimizer.zero_grad(set_to_none=True)
            if self.scaler is not None:
                try:
                    current_scale = self.scaler.get_scale()
                    self.scaler.update(new_scale=max(current_scale / 2, 2. ** 8))
                except Exception:
                    pass
            return None, None

        # === (4) stepping 阶段 ===
        if stepping:
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)

            if self.early_clipping:
                try:
                    orig_norm = torch.nn.utils.clip_grad_norm_(self.paras, self.grad_clip)
                    if torch.isnan(orig_norm):
                        print(f"[CLIP ERROR] grad norm became NaN.", flush=True)
                except Exception as e:
                    print(f"[clip_grad_norm_ error] {e}", flush=True)
                    orig_norm = torch.tensor(float("nan"))

            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                scaler_sc = self.scaler.get_scale()
                if scaler_sc > 32768.:
                    print(f"[GradScaler Warning] scale={scaler_sc}, reset to 32768.", flush=True)
                    self.scaler.update(new_scale=32768.)
                else:
                    self.scaler.update()

                try:
                    scaler_sc = max(float(scaler_sc), 1e-16)
                    scaler_sc = float(math.log2(scaler_sc))
                except Exception as e:
                    print(f"[Scaler log2 Error] scaler_sc={scaler_sc}", flush=True)
                    raise e
            else:
                self.optimizer.step()

            if self.late_clipping:
                orig_norm = getattr(self.optimizer, 'global_grad_norm', float('nan'))

            self.optimizer.zero_grad(set_to_none=True)

        return orig_norm, scaler_sc


    
    def state_dict(self):
        return {
            'optimizer': self.optimizer.state_dict()
        } if self.scaler is None else {
            'scaler': self.scaler.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
    
    def load_state_dict(self, state, strict=True):
        if self.scaler is not None:
            try: self.scaler.load_state_dict(state['scaler'])
            except Exception as e: print(f'[fp16 load_state_dict err] {e}')
        self.optimizer.load_state_dict(state['optimizer'])
