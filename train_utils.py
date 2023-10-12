import os
from tqdm import tqdm
import torch
from transformers import get_cosine_schedule_with_warmup
from save_load import save_model


def create_loaders(model, train_dataset, dev_dataset, test_dataset, train_batch_size, val_batch_size):
    train_loader = model.create_dataloader(train_dataset, batch_size=train_batch_size, shuffle=True)
    val_loader = model.create_dataloader(dev_dataset, batch_size=val_batch_size, shuffle=False)
    test_loader = None if test_dataset is None else model.create_dataloader(test_dataset, batch_size=val_batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def create_scheduler(optimizer, min_num_steps, train_dataset, train_batch_size, num_epochs, warmup_ratio):
    # to int
    min_num_steps = int(min_num_steps)
    num_epochs = int(num_epochs)
    num_steps = max(min_num_steps, int(len(train_dataset) // train_batch_size * num_epochs))
    num_warmup_steps = int(num_steps * warmup_ratio)
    sch = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_steps)
    return sch, num_steps


def log_metrics(log_dir, description, table_dev, table_test, test_dataset):
    with open(os.path.join(log_dir, 'log_metrics.txt'), 'a') as f:
        f.write(f'{description}\n')
        f.write("dev\n")
        f.write(f'{table_dev}')
        if test_dataset is not None:
            f.write("test\n")
            f.write(f'{table_test}\n\n')


def train_model(model, optimizer, train_dataset, dev_dataset, test_dataset, min_num_steps,
                num_epochs, eval_every, log_dir, warmup_ratio, train_batch_size, val_batch_size, device):
    train_loader, val_loader, test_loader = create_loaders(model, train_dataset, dev_dataset, test_dataset,
                                                           train_batch_size, val_batch_size)

    scheduler, num_steps = create_scheduler(optimizer, min_num_steps, train_dataset,
                                            train_batch_size, num_epochs, warmup_ratio)

    # mkdir log_dir
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    pbar = tqdm(range(num_steps))
    losses = []
    metrics_values_dev = []
    metrics_values_test = []
    best_f1, current_f1, best_f1_test, current_f1_test = 0, 0, 0, 0
    best_path = None

    iter_train_loader = iter(train_loader)

    for step in pbar:
        try:
            batch = next(iter_train_loader)
        except StopIteration:
            iter_train_loader = iter(train_loader)
            batch = next(iter_train_loader)

        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        loss = model.compute_loss(batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.05)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        losses.append(loss.item())
        description = f"step: {step} | epoch: {step // len(train_loader)} | loss: {loss.item()}"

        if (step+1) % eval_every == 0:
            model.eval()
            table_dev, f1_dev = model.evaluate(val_loader)
            metrics_values_dev.append(f1_dev)

            if f1_dev > best_f1:
                best_path = save_best_model(model, log_dir, best_path, step, f1_dev)
                best_f1 = f1_dev
                current_f1 = f1_dev

            if test_loader:
                table_test, f1_test = model.evaluate(test_loader)
                metrics_values_test.append(f1_test)
                current_f1_test = f1_test
                if f1_test > best_f1_test:
                    best_f1_test = f1_test
            else:
                table_test = ""

            log_metrics(log_dir, description, table_dev, table_test, test_dataset)

            model.train()

        pbar.set_description(description + f" | best f1 dev: {best_f1:.4f} | current f1 dev: {current_f1:.4f} | "
                                           f"best f1 test: {best_f1_test:.4f} | current f1 test: {current_f1_test:.4f}")


def save_best_model(model, log_dir, best_path, step, f1_dev):
    current_path = os.path.join(log_dir, f'best_model_dev_{step}_{f1_dev}.pt')
    save_model(model, current_path)
    if best_path is not None:
        os.remove(best_path)
    return current_path
