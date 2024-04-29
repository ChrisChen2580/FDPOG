from .trainer import RegularTrainer, DpsgdTrainer, DPSURTrainer, DpnsgdTrainer, DpsgdFTrainer, DpissgdTrainer,\
    DpsgdGlobalAdaptiveTrainer, FDPTrainer


def create_trainer(
        train_loader,
        valid_loader,
        test_loader,
        model,
        optimizer,
        privacy_engine,
        evaluator,
        writer,
        device,
        config
):
    kwargs = {
        'method': config['method'],
        'max_epochs': config['max_epochs'],
        'num_groups': config['num_groups'],
        'selected_groups': config['selected_groups'],
        'lr': config['lr'],
        'seed': config['seed']
    }

    if config["method"] == "regular":
        trainer = RegularTrainer(
            model,
            optimizer,
            train_loader,
            valid_loader,
            test_loader,
            writer,
            evaluator,
            device,
            **kwargs
        )
    elif config["method"] == "dpsgd":
        trainer = DpsgdTrainer(
            model,
            optimizer,
            privacy_engine,
            train_loader,
            valid_loader,
            test_loader,
            writer,
            evaluator,
            device,
            delta=config["delta"],
            **kwargs
        )
    elif config["method"] == "dpsur":
        trainer = DPSURTrainer(
            model,
            optimizer,
            privacy_engine,
            train_loader,
            valid_loader,
            test_loader,
            writer,
            evaluator,
            device,
            sigma_v=config["sigma_v"],
            C_v=config["C_v"],
            bs_valid=config["bs_valid"],
            beta=config["beta"],
            delta=config["delta"],
            **kwargs
        )
    elif config["method"] == "dpnsgd":
        trainer = DpnsgdTrainer(
            model,
            optimizer,
            privacy_engine,
            train_loader,
            valid_loader,
            test_loader,
            writer,
            evaluator,
            device,
            delta=config["delta"],
            **kwargs
        )
    elif config["method"] == "dpsgd-f":
        trainer = DpsgdFTrainer(
            model,
            optimizer,
            privacy_engine,
            train_loader,
            valid_loader,
            test_loader,
            writer,
            evaluator,
            device,
            delta=config["delta"],
            base_max_grad_norm=config["base_max_grad_norm"],  # C0
            counts_noise_multiplier=config["counts_noise_multiplier"],  # noise scale applied on mk and ok
            **kwargs
        )
    elif config["method"] == "dp-is-sgd":
        trainer = DpissgdTrainer(
            model,
            optimizer,
            privacy_engine,
            train_loader,
            valid_loader,
            test_loader,
            writer,
            evaluator,
            device,
            delta=config["delta"],
            **kwargs
        )
    elif config["method"] == "dpsgd-global-adapt":
        trainer = DpsgdGlobalAdaptiveTrainer(
            model,
            optimizer,
            privacy_engine,
            train_loader,
            valid_loader,
            test_loader,
            writer,
            evaluator,
            device,
            delta=config["delta"],
            strict_max_grad_norm=config["strict_max_grad_norm"],
            bits_noise_multiplier=config["bits_noise_multiplier"],
            lr_Z=config["lr_Z"],
            threshold=config["threshold"],
            **kwargs
        )
    elif config["method"] == "fdp":
        trainer = FDPTrainer(
            model,
            optimizer,
            privacy_engine,
            train_loader,
            valid_loader,
            test_loader,
            writer,
            evaluator,
            device,
            delta=config["delta"],
            sample_rate=config["sample_rate"],
            **kwargs
        )
    else:
        raise ValueError("Training method not implemented")

    return trainer
