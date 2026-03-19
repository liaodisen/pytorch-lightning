import argparse
import logging
from pathlib import Path
from typing import Callable

from torch.utils.data import DataLoader, Dataset

from lightning.pytorch import Callback, LightningDataModule, Trainer
from lightning.pytorch.demos.boring_classes import BoringModel, RandomDataset, RandomIterableDataset
from lightning.pytorch.loggers import CSVLogger


class EmptyDataset(Dataset):
    def __len__(self) -> int:
        return 0

    def __getitem__(self, index: int) -> int:
        raise IndexError(index)


class StopAfterFirstEpoch(Callback):
    def on_train_epoch_end(self, trainer: Trainer, pl_module: BoringModel) -> None:
        if trainer.current_epoch == 0:
            trainer.should_stop = True


class CountingDataModule(LightningDataModule):
    def __init__(self) -> None:
        super().__init__()
        self.train_dataloader_calls = 0

    def train_dataloader(self) -> DataLoader:
        self.train_dataloader_calls += 1
        return DataLoader(RandomDataset(32, 16), batch_size=8)


def _base_trainer_kwargs() -> dict:
    return {
        "accelerator": "cpu",
        "devices": 1,
        "enable_checkpointing": False,
        "enable_progress_bar": False,
        "enable_model_summary": False,
    }


def _configure_logging(case_name: str) -> None:
    debug_enabled = case_name in {"case7", "case10"}
    logging.basicConfig(level=logging.DEBUG if debug_enabled else logging.INFO, force=True)
    logging.getLogger("lightning.pytorch.loops.fit_loop").setLevel(logging.DEBUG if debug_enabled else logging.INFO)
    logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.DEBUG if debug_enabled else logging.INFO)


def case1() -> str:
    model = BoringModel()
    train_loader = DataLoader(EmptyDataset(), batch_size=8)
    trainer = Trainer(max_epochs=3, logger=False, **_base_trainer_kwargs())
    trainer.fit(model, train_dataloaders=train_loader)
    return "completed"


def case2() -> str:
    model = BoringModel()
    train_loader = DataLoader(RandomDataset(32, 64), batch_size=8)
    trainer = Trainer(max_epochs=0, logger=False, **_base_trainer_kwargs())
    trainer.fit(model, train_dataloaders=train_loader)
    return "completed"


def case3() -> str:
    model = BoringModel()
    train_loader = DataLoader(RandomDataset(32, 64), batch_size=8)
    trainer = Trainer(max_epochs=2, logger=False, **_base_trainer_kwargs())
    trainer.fit(model, train_dataloaders=train_loader)
    return "completed"


def case4() -> str:
    model = BoringModel()
    train_loader = DataLoader(RandomDataset(32, 64), batch_size=8)
    trainer = Trainer(
        max_steps=2,
        max_epochs=10,
        logger=False,
        **_base_trainer_kwargs(),
    )
    trainer.fit(model, train_dataloaders=train_loader)
    return "completed"


def case5() -> str:
    model = BoringModel()
    model.eval()
    train_loader = DataLoader(RandomDataset(32, 64), batch_size=8)
    trainer = Trainer(max_epochs=1, logger=False, **_base_trainer_kwargs())
    trainer.fit(model, train_dataloaders=train_loader)
    return "completed"


def case6() -> str:
    model = BoringModel()
    train_loader = DataLoader(RandomIterableDataset(32, 8), batch_size=2)
    val_loader = DataLoader(RandomIterableDataset(32, 8), batch_size=2)
    trainer = Trainer(
        max_epochs=1,
        val_check_interval=1.0,
        logger=False,
        **_base_trainer_kwargs(),
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    return "completed"


def case7() -> str:
    model = BoringModel()
    datamodule = CountingDataModule()
    trainer = Trainer(
        max_epochs=3,
        logger=False,
        reload_dataloaders_every_n_epochs=1,
        **_base_trainer_kwargs(),
    )
    trainer.fit(model, datamodule=datamodule)
    return "completed"


def case8() -> str:
    model = BoringModel()
    train_loader = DataLoader(RandomDataset(32, 16), batch_size=8)
    logger = CSVLogger(save_dir=str(Path("examples") / "toy_logs"), name="missing_fit_loop_logs")
    trainer = Trainer(
        max_epochs=1,
        logger=logger,
        log_every_n_steps=50,
        **_base_trainer_kwargs(),
    )
    trainer.fit(model, train_dataloaders=train_loader)
    return "completed"


def case9() -> str:
    model = BoringModel()
    train_loader = DataLoader(RandomDataset(32, 16), batch_size=8)
    val_loader = DataLoader(RandomDataset(32, 16), batch_size=8)
    trainer = Trainer(
        max_epochs=1,
        val_check_interval=1,
        logger=False,
        **_base_trainer_kwargs(),
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    return "completed"


def case10() -> str:
    model = BoringModel()
    trainer = Trainer(
        max_epochs=5,
        logger=False,
        callbacks=[StopAfterFirstEpoch()],
        **_base_trainer_kwargs(),
    )
    train_loader = DataLoader(RandomDataset(32, 64), batch_size=8)
    trainer.fit(model, train_dataloaders=train_loader)
    return "completed"


CASES: dict[str, Callable[[], str]] = {
    "case1": case1,
    "case2": case2,
    "case3": case3,
    "case4": case4,
    "case5": case5,
    "case6": case6,
    "case7": case7,
    "case8": case8,
    "case9": case9,
    "case10": case10,
}


def _run_case(case_name: str, run_case: Callable[[], str]) -> None:
    _configure_logging(case_name)
    print(f"\n=== {case_name} ===")
    try:
        result = run_case()
        print(f"{case_name}: {result}")
    except Exception as ex:
        print(f"{case_name}: {type(ex).__name__}: {ex}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        choices=["all", *CASES],
        default="all",
    )
    args = parser.parse_args()

    if args.case == "all":
        for case_name, run_case in CASES.items():
            _run_case(case_name, run_case)
        return

    _run_case(args.case, CASES[args.case])


if __name__ == "__main__":
    main()
