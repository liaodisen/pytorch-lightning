# Toy Run Instructor Reference

This file maps each case in [toy_run.py](/Users/disen/pytorch-lightening-example/examples/toy_run.py) to the missing or previously removed logging in [fit_loop.py](/Users/disen/pytorch-lightening-example/src/lightning/pytorch/loops/fit_loop.py). Cases 3, 6, and 9 are intentional control cases: the code is valid and there is no underlying issue for students to diagnose.

## case1

- Hidden log: ``Trainer.fit` stopped: No training batches.``
- Problem: A data pipeline bug produced an empty dataset, so training exits before a single optimization step.
- Relevant code path: `_FitLoop.done` when `self.max_batches == 0`

## case2

- Hidden log: ``Trainer.fit` stopped: `max_epochs=0` reached.``
- Problem: A config or CLI parsing mistake disabled training completely, but without the log it looks like `fit` returned for no reason.
- Relevant code path: `_FitLoop.done` when `stop_epochs` is reached

## case3

- Hidden log: none expected
- Problem: None. This is a normal successful run.
- Relevant code path: control case

## case4

- Hidden log: ``Trainer.fit` stopped: `max_steps=...` reached.``
- Problem: Training stops after only a couple of optimization steps because `max_steps` is tighter than `max_epochs`, which can look like a random early exit.
- Relevant code path: `_FitLoop.done` when `stop_steps` is reached

## case5

- Hidden log: `Found N module(s) in eval mode at the start of training. This may lead to unexpected behavior during training. If this is intentional, you can ignore this warning.`
- Problem: Dropout and BatchNorm can stay in inference mode, so the model may train with the wrong behavior and there is no warning explaining why.
- Relevant code path: `_warn_if_modules_in_eval_mode()` called from `_FitLoop.on_run_start`

## case6

- Hidden log: none expected
- Problem: None. This is a normal successful run with a valid iterable-dataset configuration.
- Relevant code path: control case

## case7

- Hidden log: `_FitLoop: resetting train dataloader`
- Problem: The training dataloader is silently recreated every epoch, which can hide expensive I/O, shuffled-state resets, or dataloader-side bugs.
- Relevant code path: debug logging in `_FitLoop.setup_data`

## case8

- Hidden log: `The number of training batches (...) is smaller than the logging interval Trainer(log_every_n_steps=...). Set a lower value for log_every_n_steps if you want to see logs for the training epoch.`
- Problem: Training metrics appear to be missing, but the actual issue is a logger configuration that can never emit within one epoch.
- Relevant code path: warning block in `_FitLoop.setup_data`

## case9

- Hidden log: none expected
- Problem: None. This is a normal successful run with a valid validation interval.
- Relevant code path: control case

## case10

- Hidden log: ``Trainer.fit` stopped: `trainer.should_stop` was set.``
- Problem: A callback, hook, or custom control flow stopped training early, making the short run look unexplained.
- Relevant code path: `_FitLoop.done` when `trainer.should_stop and self._can_stop_early`
