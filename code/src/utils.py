from src.datamodule import PL2NLDM

def test_pl2nldm(args):
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
    
    dm.setup()

    print(next(iter(dm.train_dataloader())))