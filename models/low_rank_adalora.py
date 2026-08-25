from peft import AdaLoraConfig, get_peft_model, TaskType
import torch
import torch.nn as nn


TARGET_MODULES = {
    'llama': ["q_proj", "v_proj"],
    'mistral': ["q_proj", "k_proj", "v_proj", "o_proj"],
    'opt': None,
    'gpt2': ["c_attn", "c_proj", "c_fc"],
    'llava': ["q_proj", "v_proj"]
}


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    print(
        f"trainable params: {trainable_params} || all params: {all_param} || "
        f"trainable%: {100 * trainable_params / all_param}"
    )


def peft_model(plm, plm_type, rank, total_step,
               task_type=TaskType.FEATURE_EXTRACTION,
               use_shapley=False, n_perm=3):
    """
    Wrap the plm with AdaLoRA instead of vanilla LoRA.

    Differences vs. the original LoRA version:
      - Uses AdaLoraConfig: starts at init_r (> target_r) and prunes the rank
        budget down to target_r according to a schedule.
      - Requires `total_step` = total number of OPTIMIZER update steps, so the
        RankAllocator can schedule the budget over the whole run.
      - The training loop MUST call `model.base_model.update_and_allocate(step)`
        after optimizer.step() and BEFORE optimizer.zero_grad() (see
        run_adalora.py). Without that call, AdaLoRA silently behaves like a
        fixed-rank LoRA.
    """
    for param in plm.parameters():
        param.requires_grad = False
        if param.ndim == 1:
            param.data = param.data.to(torch.float32)

    plm.gradient_checkpointing_enable()
    plm.enable_input_require_grads()

    # pruning schedule derived from the total number of optimizer steps
    tinit = max(1, int(total_step * 0.1))            # warmup: no pruning before this
    tfinal = max(tinit + 1, int(total_step * 0.15))   # stop pruning after this
    deltaT = 10                                      # reallocate every deltaT steps

    config = AdaLoraConfig(
        init_r=rank * 2,        # start with a larger rank ...
        target_r=rank,          # ... and prune down to this average rank
        tinit=tinit,
        tfinal=tfinal,
        deltaT=deltaT,
        lora_alpha=32,
        target_modules=TARGET_MODULES[plm_type],
        lora_dropout=0.05,
        bias="none",
        task_type=task_type,
        total_step=total_step,
    )

    model = get_peft_model(plm, config)
    print_trainable_parameters(model)
    # ── Shapley allocator 주입 (loss_fn 은 run()에서 나중에 꽂음) ──
    if use_shapley:
        from shapley_allocator import ShapleyRankAllocator
        am = model.base_model  # AdaLoraModel
        aname = am.trainable_adapter_name
        am.rankallocator = ShapleyRankAllocator(
            am.model, am.peft_config[aname], aname,
            loss_fn=None, n_perm=n_perm,
        )
        print(f"[shapley] ShapleyRankAllocator injected (n_perm={n_perm})")
    # ────────────────────────────────────────────────────────────
    return model
