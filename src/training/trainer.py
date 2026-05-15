import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.occupancy_dataset import OccupancyDataset
from src.utils.seed import seed_all

from pathlib import Path
from typing import Callable

def train_single_case(
    model: nn.Module,
    coords: np.ndarray,
    occ: np.ndarray,
    device: torch.device,
    seed: int,
    case_name: str = "",
    batch_size: int = 2048,
    lr: float = 1e-3,
    num_epochs: int = 600,
    checkpoint_epochs: list[int] | None = None,
    checkpoint_dir: Path | None = None,
    checkpoint_callback: Callable[[int, nn.Module, list[float]], None] | None = None,
):
    """
    Train one INR model on one liver occupancy point set.
    """
    torch_gen, _ = seed_all(seed)

    dataset = OccupancyDataset(coords, occ)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=torch_gen,
    )

    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loss_history = []
    
    checkpoint_epochs = {
        int(epoch)
        for epoch in (checkpoint_epochs or [])
        if 0 < int(epoch) <= num_epochs
}
    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    bar_desc = f"Training {case_name}" if case_name else "Training"
    epoch_bar = tqdm(
        range(1, num_epochs + 1),
        desc=bar_desc,
        unit="epoch",
        leave=False,
    )

    for epoch in epoch_bar:
        model.train()
        running_loss = 0.0

        for batch_coords, batch_occ in dataloader:
            batch_coords = batch_coords.to(device, non_blocking=True)
            batch_occ = batch_occ.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            logits = model(batch_coords)
            loss = criterion(logits, batch_occ)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_coords.size(0)

        epoch_loss = running_loss / len(dataset)
        loss_history.append(epoch_loss)

        epoch_bar.set_postfix(loss=f"{epoch_loss:.6f}")
        
        if epoch in checkpoint_epochs:
            if checkpoint_dir is not None:
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "loss_history": loss_history,
                    },
                    checkpoint_dir / f"epoch_{epoch:04d}.pt",
                )
            if checkpoint_callback is not None:
                checkpoint_callback(epoch, model, loss_history)
        
    if checkpoint_dir is not None:
        torch.save(
            {
                "epoch": num_epochs,
                "model_state_dict": model.state_dict(),
                "loss_history": loss_history,
            },
            checkpoint_dir / "final.pt",
        )

    return model, loss_history