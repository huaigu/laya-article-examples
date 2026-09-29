"""The validation-selection phase used for the episode 03 native Laya run."""

import gc
import json
import math
from pathlib import Path
import time

from common import save, sha, tensor_digest
from logic import best_epoch, groups
import modeling as m
from transformers import get_linear_schedule_with_warmup


def _status(root, phase, **values):
    value = {"phase": phase, "timestamp": time.time(), **values}
    save(Path(root) / "status.json", value)
    print(json.dumps(value, ensure_ascii=False), flush=True)


def _check_stop(root):
    if (Path(root) / "STOP").exists():
        raise InterruptedError("STOP file detected")


def train_selection(base, training, validation, protocol, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    if protocol["micro"] != 4:
        raise ValueError("modeling.loader is fixed to the registered micro-batch of 4")
    accumulation = protocol["batch"] // protocol["micro"]
    if protocol["batch"] % protocol["micro"]:
        raise ValueError("Effective batch must be divisible by micro-batch")

    m.seed_all(protocol["seed"])
    _status(output, "load_model")
    model, tokenizer, config, parameter_groups = m.load_model(base)
    probe = output / "weight-roundtrip.safetensors"
    m.save_weights(probe, model)
    weight_probe = m.verify_weights(probe, model)
    probe.unlink()
    frozen_action_head = tensor_digest(model.act_head.named_parameters())

    _status(output, "encode_training")
    items, budget = m.encode(training, tokenizer, protocol)
    validation_items, validation_budget = m.encode(validation, tokenizer, protocol)
    if budget["head_max_len"] != validation_budget["head_max_len"]:
        raise ValueError("Training and validation option budgets differ")
    budget["max_len"] = max(budget["max_len"], validation_budget["max_len"])
    budget["max_tokens"] = max(budget["max_tokens"], validation_budget["max_tokens"])
    save(
        output / "input-audit.json",
        {
            "train_examples": len(items),
            "validation_examples": len(validation_items),
            "test_read": False,
            "budget": budget,
            "weight_roundtrip": weight_probe,
        },
    )

    parameters = [parameter for group in parameter_groups for parameter in group["params"]]
    optimizer = m.torch.optim.AdamW(
        parameter_groups, lr=protocol["lr"], betas=(0.9, 0.999), eps=1e-8, foreach=False
    )
    updates_per_epoch = math.ceil(len(m.loader(items, tokenizer, True)) / accumulation)
    total_updates = updates_per_epoch * protocol["epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        max(1, round(total_updates * protocol["warmup"])),
        total_updates,
    )
    scaler = m.torch.amp.GradScaler("cuda", init_scale=1024.0)
    history = []

    for epoch in range(1, protocol["epochs"] + 1):
        _check_stop(output)
        model.train()
        m.torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        loss_sum = 0.0
        seen = successful = overflow_skips = consecutive_overflows = 0
        iterator = groups(m.loader(items, tokenizer, True), accumulation)
        for step, batch_group in enumerate(iterator, 1):
            _check_stop(output)
            count = sum(int(batch["label"].numel()) for batch in batch_group)
            for batch in batch_group:
                logits = m.forward(model, batch)
                loss = m.torch.nn.functional.cross_entropy(logits, batch["label"].to("cuda"))
                if not m.torch.isfinite(loss).item():
                    raise FloatingPointError("Nonfinite forward loss")
                size = int(batch["label"].numel())
                scaler.scale(loss * size / count).backward()
                loss_sum += float(loss.detach()) * size
                seen += size

            scaler.unscale_(optimizer)
            if not all(parameter.grad is not None for parameter in parameters):
                raise RuntimeError("Disconnected trainable parameter")
            finite = all(m.torch.isfinite(parameter.grad).all().item() for parameter in parameters)
            old_scale = scaler.get_scale()
            gradient_norm = None
            if finite:
                gradient_norm = m.torch.nn.utils.clip_grad_norm_(
                    parameters, 1.0, error_if_nonfinite=True, foreach=False
                )
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                successful += 1
                consecutive_overflows = 0
            else:
                scaler.step(optimizer)
                scaler.update()
                overflow_skips += 1
                consecutive_overflows += 1
                if scaler.get_scale() >= old_scale:
                    raise FloatingPointError("Gradient scaler did not detect overflow")
                if consecutive_overflows >= 3:
                    raise FloatingPointError("Three consecutive overflow skips")
            optimizer.zero_grad(set_to_none=True)
            if step == 1 or step % 25 == 0 or seen == len(items):
                _status(
                    output,
                    "training",
                    epoch=epoch,
                    max_epochs=protocol["epochs"],
                    seen=seen,
                    total=len(items),
                    train_loss=loss_sum / seen,
                    successful_updates=successful,
                    overflow_skips=overflow_skips,
                    lr=optimizer.param_groups[0]["lr"],
                    gradient_norm=float(gradient_norm) if gradient_norm is not None else None,
                    peak_allocated_mib=m.torch.cuda.max_memory_allocated() / 2**20,
                )
            del batch_group, batch, logits, loss

        if seen != len(items):
            raise RuntimeError("Training loader did not cover every item")
        _status(output, "validation", epoch=epoch)
        validation_metrics, predictions = m.score(
            model, validation_items, tokenizer, protocol["labels"]
        )
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / seen,
            "training_seconds": time.monotonic() - started,
            "optimizer_updates": successful,
            "overflow_skips": overflow_skips,
            "lr": optimizer.param_groups[0]["lr"],
            "validation": validation_metrics,
        }
        history.append(row)
        epoch_dir = output / f"epoch-{epoch:02d}"
        epoch_dir.mkdir()
        save(epoch_dir / "epoch.json", row)
        save(epoch_dir / "validation.json", predictions)
        save(output / "history.json", history)

        export = epoch_dir / "export"
        m.export_model(export, model, tokenizer, config, budget)
        snapshot = {
            "epoch": epoch,
            "inference_only": True,
            "optimizer_state_saved": False,
            "model_sha256": sha(export / "model.safetensors"),
            "successful_updates": successful,
            "test_read": False,
        }
        save(epoch_dir / "snapshot.json", snapshot)
        if tensor_digest(model.act_head.named_parameters()) != frozen_action_head:
            raise RuntimeError("Frozen action head changed")
        picked = best_epoch(history)
        _status(
            output,
            "epoch_complete",
            epoch=epoch,
            validation_accuracy=validation_metrics["accuracy"],
            validation_macro_f1=validation_metrics["macro_f1"],
            best_epoch=picked["epoch"],
        )
        if epoch - picked["epoch"] >= protocol["patience"]:
            break

    selection = {
        "best_epoch": best_epoch(history)["epoch"],
        "validation": best_epoch(history)["validation"],
        "criterion": "accuracy then macro_f1; earlier wins exact ties",
        "test_seen_in_this_run": False,
        "history": history,
    }
    save(output / "selection.json", selection)
    del optimizer, scheduler, scaler, parameters, parameter_groups
    gc.collect()
    m.torch.cuda.empty_cache()
    if sha(Path(base) / "model.safetensors") != protocol["base_sha256"]:
        raise RuntimeError("Base checkpoint changed during training")
    return selection
