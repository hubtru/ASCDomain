import lightning.pytorch as pl
# from pytorch_lightning.loggers import WandbLogger
import torch
import torchaudio
import time
from torch.utils.data import DataLoader
import argparse
import torch.nn.functional as F
import transformers
import wandb
import json
import optuna
import logging
import sys
from optuna.integration import PyTorchLightningPruningCallback


from dataset.dcase24 import get_training_set, get_test_set, get_eval_set
from helpers.init import worker_init_fn
from models.convmixer import get_model
from helpers.utils import mixstyle
from helpers import nessi


class PLModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config  # results from argparse, contains all configurations for our experiment

        # module for resampling waveforms on the fly
        resample = torchaudio.transforms.Resample(
            orig_freq=self.config["orig_sample_rate"],
            new_freq=self.config["sample_rate"]
        )

        # module to preprocess waveforms into log mel spectrograms
        mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=config["sample_rate"],
            n_fft=config["n_fft"],
            win_length=config["window_length"],
            hop_length=config["hop_length"],
            n_mels=config["n_mels"],
            f_min=config["f_min"],
            f_max=config["f_max"]
        )

        freqm = torchaudio.transforms.FrequencyMasking(config["freqm"], iid_masks=True)
        timem = torchaudio.transforms.TimeMasking(config["timem"], iid_masks=True)

        self.mel = torch.nn.Sequential(
            resample,
            mel
        )

        self.mel_augment = torch.nn.Sequential(
            freqm,
            timem
        )

        # the convmixer model
        self.model = get_model(
            in_channels=config["in_channels"],
            filter=config["filters"],
            depth=config["depth"],
            kernel_size=config["kernel_size"],
            patch_size=config["patch_size"],
            n_classes=config["n_classes"]
        )

        self.device_ids = ['a', 'b', 'c', 's1', 's2', 's3', 's4', 's5', 's6']
        self.label_ids = ['airport', 'bus', 'metro', 'metro_station', 'park', 'public_square', 'shopping_mall',
                          'street_pedestrian', 'street_traffic', 'tram']
        # categorization of devices into 'real', 'seen' and 'unseen'
        self.device_groups = {'a': "real", 'b': "real", 'c': "real",
                              's1': "seen", 's2': "seen", 's3': "seen",
                              's4': "unseen", 's5': "unseen", 's6': "unseen"}

        # pl 2 containers:
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.test_acc = 0

    def mel_forward(self, x):
        """
        :param x: batch of raw audio signals (waveforms)
        :return: log mel spectrogram
        """
        x = self.mel(x)
        if self.training:
            x = self.mel_augment(x)
        x = (x + 1e-5).log()
        return x

    def forward(self, x):
        """
        :param x: batch of raw audio signals (waveforms)
        :return: final model predictions
        """
        x = self.mel_forward(x)
        x = self.model(x)
        return x

    def configure_optimizers(self):
        """
        This is the way pytorch lightening requires optimizers and learning rate schedulers to be defined.
        The specified items are used automatically in the optimization loop (no need to call optimizer.step() yourself).
        :return: optimizer and learning rate scheduler
        """

        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config["lr"], weight_decay=self.config["weight_decay"])
        scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config["warmup_steps"],
            num_training_steps=self.trainer.estimated_stepping_batches,
        )

        lr_scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1
        }
        return [optimizer], [lr_scheduler_config]

    def training_step(self, train_batch, batch_idx):
        """
        :param train_batch: contains one batch from train dataloader
        :param batch_idx
        :return: loss to update model parameters
        """
        x, files, labels, devices, cities = train_batch
        x = self.mel_forward(x)  # we convert the raw audio signals into log mel spectrograms

        if self.config["mixstyle_p"] > 0:
            # frequency mixstyle
            x = mixstyle(x, self.config["mixstyle_p"], self.config["mixstyle_alpha"])
        y_hat = self.model(x)
        samples_loss = F.cross_entropy(y_hat, labels, reduction="none")
        loss = samples_loss.mean()

        self.log("lr", self.trainer.optimizers[0].param_groups[0]['lr'])
        self.log("epoch", self.current_epoch)
        self.log("train/loss", loss.detach().cpu())
        return loss

    def on_train_epoch_end(self):
        pass

    def validation_step(self, val_batch, batch_idx):
        x, files, labels, devices, cities = val_batch

        y_hat = self.forward(x)
        samples_loss = F.cross_entropy(y_hat, labels, reduction="none")

        # for computing accuracy
        _, preds = torch.max(y_hat, dim=1)
        n_correct_per_sample = (preds == labels)
        n_correct = n_correct_per_sample.sum()

        dev_names = [d.rsplit("-", 1)[1][:-4] for d in files]
        results = {'loss': samples_loss.mean(), "n_correct": n_correct,
                   "n_pred": torch.as_tensor(len(labels), device=self.device)}

        # log metric per device and scene
        for d in self.device_ids:
            results["devloss." + d] = torch.as_tensor(0., device=self.device)
            results["devcnt." + d] = torch.as_tensor(0., device=self.device)
            results["devn_correct." + d] = torch.as_tensor(0., device=self.device)
        for i, d in enumerate(dev_names):
            results["devloss." + d] = results["devloss." + d] + samples_loss[i]
            results["devn_correct." + d] = results["devn_correct." + d] + n_correct_per_sample[i]
            results["devcnt." + d] = results["devcnt." + d] + 1

        for l in self.label_ids:
            results["lblloss." + l] = torch.as_tensor(0., device=self.device)
            results["lblcnt." + l] = torch.as_tensor(0., device=self.device)
            results["lbln_correct." + l] = torch.as_tensor(0., device=self.device)
        for i, l in enumerate(labels):
            results["lblloss." + self.label_ids[l]] = results["lblloss." + self.label_ids[l]] + samples_loss[i]
            results["lbln_correct." + self.label_ids[l]] = \
                results["lbln_correct." + self.label_ids[l]] + n_correct_per_sample[i]
            results["lblcnt." + self.label_ids[l]] = results["lblcnt." + self.label_ids[l]] + 1
        results = {k: v.cpu() for k, v in results.items()}
        self.validation_step_outputs.append(results)

    def on_validation_epoch_end(self):
        # convert a list of dicts to a flattened dict
        outputs = {k: [] for k in self.validation_step_outputs[0]}
        for step_output in self.validation_step_outputs:
            for k in step_output:
                outputs[k].append(step_output[k])
        for k in outputs:
            outputs[k] = torch.stack(outputs[k])

        avg_loss = outputs['loss'].mean()
        acc = sum(outputs['n_correct']) * 1.0 / sum(outputs['n_pred'])

        logs = {'acc': acc, 'loss': avg_loss}

        # log metric per device and scene
        for d in self.device_ids:
            dev_loss = outputs["devloss." + d].sum()
            dev_cnt = outputs["devcnt." + d].sum()
            dev_corrct = outputs["devn_correct." + d].sum()
            logs["loss." + d] = dev_loss / dev_cnt
            logs["acc." + d] = dev_corrct / dev_cnt
            logs["cnt." + d] = dev_cnt
            # device groups
            logs["acc." + self.device_groups[d]] = logs.get("acc." + self.device_groups[d], 0.) + dev_corrct
            logs["count." + self.device_groups[d]] = logs.get("count." + self.device_groups[d], 0.) + dev_cnt
            logs["lloss." + self.device_groups[d]] = logs.get("lloss." + self.device_groups[d], 0.) + dev_loss

        for d in set(self.device_groups.values()):
            logs["acc." + d] = logs["acc." + d] / logs["count." + d]
            logs["lloss." + d] = logs["lloss." + d] / logs["count." + d]

        for l in self.label_ids:
            lbl_loss = outputs["lblloss." + l].sum()
            lbl_cnt = outputs["lblcnt." + l].sum()
            lbl_corrct = outputs["lbln_correct." + l].sum()
            logs["loss." + l] = lbl_loss / lbl_cnt
            logs["acc." + l] = lbl_corrct / lbl_cnt
            logs["cnt." + l] = lbl_cnt

        logs["macro_avg_acc"] = torch.mean(torch.stack([logs["acc." + l] for l in self.label_ids]))
        # prefix with 'val' for logging
        self.log_dict({"val/" + k: logs[k] for k in logs})
        self.validation_step_outputs.clear()

    def test_step(self, test_batch, batch_idx):
        x, files, labels, devices, cities = test_batch

        # maximum memory allowance for parameters: 128 KB
        # baseline has 61148 parameters -> we can afford 16-bit precision
        # since 61148 * 16 bit ~ 122 kB

        # assure fp16
        self.model.half()
        x = self.mel_forward(x)
        x = x.half()
        y_hat = self.model(x)
        samples_loss = F.cross_entropy(y_hat, labels, reduction="none")

        # for computing accuracy
        _, preds = torch.max(y_hat, dim=1)
        n_correct_per_sample = (preds == labels)
        n_correct = n_correct_per_sample.sum()

        dev_names = [d.rsplit("-", 1)[1][:-4] for d in files]
        results = {'loss': samples_loss.mean(), "n_correct": n_correct,
                   "n_pred": torch.as_tensor(len(labels), device=self.device)}

        # log metric per device and scene
        for d in self.device_ids:
            results["devloss." + d] = torch.as_tensor(0., device=self.device)
            results["devcnt." + d] = torch.as_tensor(0., device=self.device)
            results["devn_correct." + d] = torch.as_tensor(0., device=self.device)
        for i, d in enumerate(dev_names):
            results["devloss." + d] = results["devloss." + d] + samples_loss[i]
            results["devn_correct." + d] = results["devn_correct." + d] + n_correct_per_sample[i]
            results["devcnt." + d] = results["devcnt." + d] + 1

        for l in self.label_ids:
            results["lblloss." + l] = torch.as_tensor(0., device=self.device)
            results["lblcnt." + l] = torch.as_tensor(0., device=self.device)
            results["lbln_correct." + l] = torch.as_tensor(0., device=self.device)
        for i, l in enumerate(labels):
            results["lblloss." + self.label_ids[l]] = results["lblloss." + self.label_ids[l]] + samples_loss[i]
            results["lbln_correct." + self.label_ids[l]] = \
                results["lbln_correct." + self.label_ids[l]] + n_correct_per_sample[i]
            results["lblcnt." + self.label_ids[l]] = results["lblcnt." + self.label_ids[l]] + 1
        self.test_step_outputs.append(results)

    def on_test_epoch_end(self):
        # convert a list of dicts to a flattened dict
        outputs = {k: [] for k in self.test_step_outputs[0]}
        for step_output in self.test_step_outputs:
            for k in step_output:
                outputs[k].append(step_output[k])
        for k in outputs:
            outputs[k] = torch.stack(outputs[k])

        avg_loss = outputs['loss'].mean()
        acc = sum(outputs['n_correct']) * 1.0 / sum(outputs['n_pred'])

        logs = {'acc': acc, 'loss': avg_loss}

        # log metric per device and scene
        for d in self.device_ids:
            dev_loss = outputs["devloss." + d].sum()
            dev_cnt = outputs["devcnt." + d].sum()
            dev_corrct = outputs["devn_correct." + d].sum()
            logs["loss." + d] = dev_loss / dev_cnt
            logs["acc." + d] = dev_corrct / dev_cnt
            logs["cnt." + d] = dev_cnt
            # device groups
            logs["acc." + self.device_groups[d]] = logs.get("acc." + self.device_groups[d], 0.) + dev_corrct
            logs["count." + self.device_groups[d]] = logs.get("count." + self.device_groups[d], 0.) + dev_cnt
            logs["lloss." + self.device_groups[d]] = logs.get("lloss." + self.device_groups[d], 0.) + dev_loss

        for d in set(self.device_groups.values()):
            logs["acc." + d] = logs["acc." + d] / logs["count." + d]
            logs["lloss." + d] = logs["lloss." + d] / logs["count." + d]

        for l in self.label_ids:
            lbl_loss = outputs["lblloss." + l].sum()
            lbl_cnt = outputs["lblcnt." + l].sum()
            lbl_corrct = outputs["lbln_correct." + l].sum()
            logs["loss." + l] = lbl_loss / lbl_cnt
            logs["acc." + l] = lbl_corrct / lbl_cnt
            logs["cnt." + l] = lbl_cnt

        logs["macro_avg_acc"] = torch.mean(torch.stack([logs["acc." + l] for l in self.label_ids]))
        # prefix with 'test' for logging
        self.log_dict({"test/" + k: logs[k] for k in logs})
        self.test_step_outputs.clear()

    def predict_step(self, eval_batch, batch_idx, dataloader_idx=0):
        x, files = eval_batch

        # assure fp16
        self.model.half()

        x = self.mel_forward(x)
        x = x.half()
        y_hat = self.model(x)

        return files, y_hat


def train(config, trial):
    # train dataloader
    assert config["subset"] in {100, 50, 25, 10, 5}, "Specify an integer value in: {100, 50, 25, 10, 5} to use one of " \
                                                  "the given subsets."
    roll_samples = config["orig_sample_rate"] * config["roll_sec"]
    train_dl = DataLoader(dataset=get_training_set(config["subset"], roll=roll_samples),
                          worker_init_fn=worker_init_fn,
                          num_workers=config["num_workers"],
                          batch_size=config["batch_size"],
                          shuffle=True)

    test_dl = DataLoader(dataset=get_test_set(),
                         worker_init_fn=worker_init_fn,
                         num_workers=config["num_workers"],
                         batch_size=config["batch_size"])

    # create pytorch lightening module
    pl_module = PLModule(config)

    # get model complexity from nessi and log results to wandb
    sample = next(iter(test_dl))[0][0].unsqueeze(0)
    shape = pl_module.mel_forward(sample).size()
    macs, params = nessi.get_torch_size(pl_module.model, input_size=shape)

    # return 0 if maximum macs or params are exceeded
    if macs > nessi.MAX_MACS or params > nessi.MAX_PARAMS_MEMORY:
        return 0.0

    # create the pytorch lightening trainer by specifying the number of epochs to train, the logger,
    # on which kind of device(s) to train and possible callbacks
    trainer = pl.Trainer(max_epochs=config["n_epochs"],
                         logger=True,
                         accelerator='gpu',
                         devices=1,
                         precision=config["precision"],
                         callbacks=[
                            pl.callbacks.ModelCheckpoint(save_last=True),
                            PyTorchLightningPruningCallback(trial, monitor="test/acc")
                        ])
    # start training and validation for the specified number of epochs
    trainer.fit(pl_module, train_dl, test_dl)

    # final test step
    # here: use the validation split
    trainer.test(ckpt_path='last', dataloaders=test_dl)
    return trainer.callback_metrics["test/acc"].item()


def objective(trial):
    config = {
        # convmixer hyper parameter
        'filters': trial.suggest_int('filters', 32, 128),
        'depth': trial.suggest_int('depth', 4, 24) ,
        'kernel_size': trial.suggest_int('kernel_size', 3, 6),
        'patch_size': trial.suggest_int('patch_size', 4, 16),
        'n_classes': 10,
        'in_channels':1,

        # general
        'project_name': "DCASE24_Task1",
        'experiment_name': "ConvMixer10",
        'num_workers': 8, # number of workers for dataloaders
        'precision': "32",

        # dataset
        'orig_sample_rate': 44100,
        'subset': 5,

        # training
        'n_epochs': 10,
        'batch_size': 256,
        'mixstyle_p': 0.4, # frequency mixstyle
        'mixstyle_alpha': 0.3,
        'weight_decay': 0.0001,
        'roll_sec': 0.1,  # roll waveform over time

        # peak learning rate (in cosinge schedule)
        'lr': 0.005,
        'warmup_steps':2000,

        # preprocessing
        'sample_rate':  32000,
        'window_length': 3072,  # in samples (corresponds to 96 ms)
        'hop_length': 500,  # in samples (corresponds to ~16 ms)
        'n_fft': 4096,  # length (points) of fft, e.g. 4096 point FFT
        'n_mels': 256,  # number of mel bins
        'freqm': 48,  # mask up to 'freqm' spectrogram bins
        'timem': 0,  # mask up to 'timem' spectrogram frames
        'f_min': 0,  # mel bins are created for freqs. between 'f_min' and 'f_max'
        'f_max': None,
    }

    test_acc = train(config, trial)
    return test_acc

if __name__ == '__main__':
    filters = [32, 64, 128]
    depth = [4,6,8,9,10]
    kernel_size = [3,4,5,6]
    patch_size = [4,5,6,7,8,10,12,14,16]
    n_trials = len(filters) * len(kernel_size) * len(patch_size) * len(depth)
    print('n_trials', n_trials)
    sys.stdout.flush()

    search_space = {
        'filters': filters, 
        'depth': depth,
        'kernel_size': kernel_size,
        'patch_size': patch_size
    }

    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name = "convmixer_hpo"  # Unique identifier of the study.
    storage = "sqlite:///{}.db".format(study_name)
    study = optuna.create_study(
            direction='maximize', 
            sampler=optuna.samplers.GridSampler(search_space),
            study_name=study_name,
            storage=storage
        )
    
    start = time.time()
    study.optimize(objective, n_trials=n_trials)
    end = time.time()
    print('Time:', (end - start) / 60_000, 'minutes')

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
