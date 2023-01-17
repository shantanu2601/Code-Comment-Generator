from base64 import encode
import pytorch_lightning as pl 

from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

from src.datamodule import NL2NLDM, PL2NLDM
from src.model import NL2NL, PL2NL, NLPLSpace
from src.callbacks import PLEncoderCheckPoint

from config import (
    GLOBAL_SEED,
    PROJECT_NAME
)


# TRAIN NL2NL 
def train_nl2nl(args):
    
    # Setting Global Seed
    seed_everything(GLOBAL_SEED)

    # Initialize data module
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

    # Initialize model
    model = NL2NL(
        learning_rate=args.lr, 
        weight_decay=args.wd, 
    )

    # Initialize appropriate logging platform
    logger = TensorBoardLogger(
        save_dir=args.path_logs,
        run_name=args.run_name,
    )

    if args.logger == "wandb":
        logger = WandbLogger(
            name=args.run_name, 
            id=args.run_name, 
            save_dir=args.path_logs,
            project=PROJECT_NAME
        )

        # Log HPC JobID to identify the experiment and logs stored on HPC    
        logger.log_hyperparams({"jobid": args.jobid})
    

    # Initialize trainer 
    trainer = pl.Trainer(
        logger=logger, 
        accelerator='gpu', 
        devices=args.gpus,
        max_epochs=args.epochs, 
        log_every_n_steps=2,
        deterministic=True,
    )

    # Run!
    trainer.fit(model, datamodule=dm)

    # Save trained encoder and decoder 
    model.save(
        encoder_path=args.path_save_nl_encoder, 
        decoder_path=args.path_save_nl_decoder
    )
    

# Train NL-PL Space 
def train_nlplspace(args):

    # Set global seed 
    seed_everything(GLOBAL_SEED)

    # Initialize data module    
    dm = PL2NLDM(
        tokenizer_model=args.tokenizer,
        pl=args.pl,
        path_base_models=args.path_base_models,
        path_cache_dataset=args.path_cache_datasets,
        max_seq_len=args.max_seq_len,
        padding=args.padding,
        batch_size=args.batch_size,
        num_workers=args.workers,
    )

    # Initialize model 
    model = NLPLSpace(
        path_sot_model=args.nl_en_model,
        encoder_name_or_path=args.pl_en_model,
        g_lr=args.g_lr, 
        d_lr=args.d_lr, 
        g_wd=args.g_wd,
        d_wd=args.d_wd,
        path_base_models=args.path_base_models
    )

    if args.logger == "wandb":
        logger = WandbLogger(
            name=args.run_name, 
            id=args.run_name, 
            save_dir=args.path_logs,
            project=PROJECT_NAME
        )

        # Log HPC JobID to identify the experiment and logs stored on HPC    
        logger.log_hyperparams({"jobid": args.jobid})
    
    # Checkpoint callback 
    cb = PLEncoderCheckPoint(
        path_save_model=args.path_save_pl_encoder, 
    )

    # Initialize trainer 
    trainer = pl.Trainer(
        logger=logger, 
        accelerator='gpu', 
        devices=args.gpus,
        max_epochs=args.epochs, 
        log_every_n_steps=2,
        deterministic=True,
        callbacks=[cb]
    )

    # Run!
    trainer.fit(model, datamodule=dm)

    # Save Generator => PL Encoder 
    # model.save(args.path_save_pl_encoder)


def test_pl2nl(args):

    # Set global seed 
    seed_everything(GLOBAL_SEED)

    # Initialize data module    
    dm = PL2NLDM(
        tokenizer_model=args.tokenizer,
        pl=args.pl,
        path_base_models=args.path_base_models,
        path_cache_dataset=args.path_cache_datasets,
        max_seq_len=args.max_seq_len,
        padding=args.padding,
        batch_size=args.batch_size,
        num_workers=args.workers,
    )

    # Initialize model 
    model = PL2NL(
        encoder_name_or_path=args.pl_en_model, 
        decoder_name_or_path=args.nl_de_model, 
        tokenizer=args.tokenizer,
        path_base_models=args.path_base_models
    )

    if args.logger == "wandb":
        logger = WandbLogger(
            name=args.run_name, 
            id=args.run_name, 
            save_dir=args.path_logs,
            project=PROJECT_NAME
        )

        # Log HPC JobID to identify the experiment and logs stored on HPC    
        logger.log_hyperparams({"jobid": args.jobid})
    

    # Initialize trainer 
    trainer = pl.Trainer(
        logger=logger, 
        accelerator='gpu', 
        devices=args.gpus,
        max_epochs=args.epochs, 
        log_every_n_steps=2,
        deterministic=True,
    )

    # Run!
    trainer.test(model, datamodule=dm)
