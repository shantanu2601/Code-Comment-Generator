import pytorch_lightning as pl 
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

from src.datamodule import NL2NLDM, PL2NLDM
from src.model import NL2NL, PLNLSpace, PL2NL

from config import (
    GLOBAL_SEED,
    PROJECT_NAME
)

def train_nl2nl(args):
    seed_everything(GLOBAL_SEED)

    dm = NL2NLDM(
        tokenizer_model=args.tokenizer,
        pl=args.pl,
        path_base_models=args.path_base_models,
        path_cache_dataset=args.path_cache_datasets,
        max_seq_len=args.max_seq_len,
        padding=args.padding,
        batch_size=args.batch_size,
        num_workers=args.workers,
    )

    model = NL2NL(
        encoder_model=args.nl_en_model,
        decoder_model=args.nl_de_model,
        learning_rate=args.lr,
        weight_decay=args.wd,
    )

    logger = TensorBoardLogger(
        save_dir=args.path_logs,
        run_name=args.run_name,
    )

    if args.logger == "wandb":
        logger = WandbLogger(
            save_dir=args.path_logs,
            name=args.run_name,
            id=args.run_name,
            project=PROJECT_NAME
        )

        logger.log_hyperparams({"jobid": args.jobid})        # Logging jobid of HPC


    trainer = pl.Trainer(
        logger=logger,
        accelerator="gpu", 
        devices=args.gpus,   
        max_epochs=args.epochs,
        log_every_n_steps=2,
        deterministic=True           # Hopefully get same results on different GPUs
    )

    trainer.fit(model, datamodule=dm)
    

    model.save(
        encoder_path=args.path_save_nl_encoder,
        decoder_path=args.path_save_nl_decoder,
        lm_path=args.path_save_nl_lm
    )


def train_plnlspace(args):
    seed_everything(GLOBAL_SEED)

    dm = PL2NLDM(
        tokenizer_model=args.tokenizer,
        pl=args.pl,
        path_base_models=args.path_base_models,
        path_cache_dataset=args.path_cache_datasets,
        max_seq_len=args.max_seq_len,
        padding=args.padding,
        batch_size=args.batch_size,
        num_workers=args.workers
    )

    model = PLNLSpace(
        nl_encoder_base_model=args.nl_en_model,
        pl_encoder_base_model=args.pl_en_model,
        g_learning_rate=args.g_lr,
        d_learning_rate=args.d_lr,
        g_weight_decay=args.g_wd,
        d_weight_decay=args.d_wd,
        max_seq_len=args.max_seq_len,
        path_base_models=args.path_base_models
    )

    logger = TensorBoardLogger(
        save_dir=args.path_logs,
        run_name=args.run_name,
    )

    if args.logger == "wandb":
        logger = WandbLogger(
            save_dir=args.path_logs,
            name=args.run_name,
            id=args.run_name,
            project=PROJECT_NAME
        )

        logger.log_hyperparams({"jobid": args.jobid})        # Logging jobid of HPC
    
    trainer = pl.Trainer(
        logger=logger,
        accelerator="gpu", 
        devices=args.gpus,   
        max_epochs=args.epochs,
        log_every_n_steps=2,
        deterministic=True           # Hopefully get same results on different GPUs
    )

    trainer.fit(model, dm)
    
    model.save(
        generator_path=args.path_save_pl_encoder,
    )