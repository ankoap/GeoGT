import os
import warnings
from argparse import Namespace
from datasets import load_dataset, DatasetDict
from dataclasses import dataclass

from transformers import EvalPrediction
from transformers import TrainingArguments, Trainer
from transformers import HfArgumentParser
from transformers.trainer_utils import get_last_checkpoint

from models import GeoGTConfig, GeoGTForConformerPrediction, GeoGTCollator

warnings.simplefilter("ignore", UserWarning)

@dataclass
class ScriptArguments:
    tokenized_dataset_path: str


def compute_metrics(eval_pred: EvalPrediction):
    preds, _ = eval_pred
    mae, mse, rmsd, *_ = preds
    mae, mse, rmsd = mae.mean().item(), mse.mean().item(), rmsd.mean().item()
    return {"mae": mae, "mse": mse, "rmsd": rmsd}


def main():
    parse = HfArgumentParser(dataclass_types=[ScriptArguments, TrainingArguments])
    script_args, training_args = parse.parse_args_into_dataclasses()
    print(script_args)
    print(training_args)
    dataset = DatasetDict.load_from_disk(script_args.tokenized_dataset_path)
    train_set, eval_set, test_set = (
        dataset["train"],
        dataset["validation"],
        dataset["test"],
    )
    collate_func = GeoGTCollator()

    config = GeoGTConfig(
        n_encode_layers=6,
        encoder_use_A_in_attn=True,
        encoder_use_D_in_attn=False,
        encoder_use_e_d=False,
        encoder_use_e_a=False,
        encoder_use_D_cache=False,
        n_decode_layers=6,
        decoder_use_A_in_attn=True,
        decoder_use_D_in_attn=True,
        decoder_use_e_d=True,
        decoder_use_e_a=True,
        decoder_use_D_cache=True,
        embed_style="atom_tokenized_ids",
        atom_vocab_size=513,
        d_embed=256,
        pre_ln=False,
        d_q=256,
        d_k=256,
        d_v=256,
        d_model=256,
        n_head=8, 
        qkv_bias=True,
        attn_drop=0.00,
        norm_drop=0.00,
        ffn_drop=0.00,
        d_ffn=1024,
    )
    model = GeoGTForConformerPrediction(config)
    print(model.config)

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collate_func,
        train_dataset=train_set,
        eval_dataset=eval_set,
        compute_metrics=compute_metrics,
    )

    if training_args.do_train:
        resume_from_checkpoint = training_args.resume_from_checkpoint
        resume_f = eval(resume_from_checkpoint)
        train_result = trainer.train()
        trainer.log_metrics("train", train_result.metrics)

    if training_args.do_eval:
        # TODO: if training_args.do_train is False, model used to test is not the best!
        test_metrics = trainer.evaluate(eval_dataset=test_set)
        trainer.log_metrics("test", test_metrics)


if __name__ == "__main__":
    main()
