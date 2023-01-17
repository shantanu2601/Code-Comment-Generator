import pytorch_lightning as pl

from config import PATH_SAVE_PL_ENCODER 

class PLEncoderCheckPoint(pl.Callback):
    def __init__(
        self, 
        path_save_model: str = PATH_SAVE_PL_ENCODER
    ) -> None:
        super().__init__()
        self.path = path_save_model
    
    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        "Checkpoint PL Encoder every epoch"
        
        path = self.path + f"/pl-encoder-epoch-{pl_module.current_epoch}"
        pl_module.save(path)
