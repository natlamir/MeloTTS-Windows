import json
from collections import defaultdict
from random import shuffle
from typing import Optional
import logging

from tqdm import tqdm
import click
from text.cleaner import clean_text_bert
import os
import torch
from text.symbols import symbols, num_languages, num_tones

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@click.command()
@click.option(
    "--metadata",
    default="data/example/metadata.list",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
)
@click.option("--cleaned-path", default=None)
@click.option("--train-path", default=None)
@click.option("--val-path", default=None)
@click.option(
    "--config_path",
    default="configs/config.json",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
)
@click.option("--val-per-spk", default=4)
@click.option("--max-val-total", default=8)
@click.option("--clean/--no-clean", default=True)
def main(
    metadata: str,
    cleaned_path: Optional[str],
    train_path: str,
    val_path: str,
    config_path: str,
    val_per_spk: int,
    max_val_total: int,
    clean: bool,
):
    logger.info(f"Starting preprocessing with metadata file: {metadata}")
    if train_path is None:
        train_path = os.path.join(os.path.dirname(metadata), 'train.list')
    if val_path is None:
        val_path = os.path.join(os.path.dirname(metadata), 'val.list')
    out_config_path = os.path.join(os.path.dirname(metadata), 'config.json')

    if cleaned_path is None:
        cleaned_path = metadata + ".cleaned"

    logger.info(f"Cleaned path: {cleaned_path}")
    logger.info(f"Train path: {train_path}")
    logger.info(f"Val path: {val_path}")

    if clean:
        logger.info("Starting cleaning process")
        with open(cleaned_path, "w", encoding="utf-8") as out_file, open(metadata, encoding="utf-8") as in_file:
            new_symbols = []
            for line in tqdm(in_file):
                try:
                    utt, spk, language, text = line.strip().split("|")
                    logger.debug(f"Processing line: {line.strip()}")
                    norm_text, phones, tones, word2ph, bert = clean_text_bert(text, language, device='cuda:0')
                    for ph in phones:
                        if ph not in symbols and ph not in new_symbols:
                            new_symbols.append(ph)
                            logger.info(f'New symbol added: {ph}')
                    
                    assert len(phones) == len(tones), f"Mismatch in phones and tones length for {utt}"
                    assert len(phones) == sum(word2ph), f"Mismatch in phones and word2ph for {utt}"
                    out_file.write(
                        "{}|{}|{}|{}|{}|{}|{}\n".format(
                            utt, spk, language, norm_text,
                            " ".join(phones),
                            " ".join([str(i) for i in tones]),
                            " ".join([str(i) for i in word2ph]),
                        )
                    )
                    bert_path = utt.replace(".wav", ".bert.pt")
                    os.makedirs(os.path.dirname(bert_path), exist_ok=True)
                    torch.save(bert.cpu(), bert_path)
                    logger.debug(f"Successfully processed and saved: {utt}")
                except Exception as error:
                    logger.error(f"Error processing line: {line.strip()}")
                    logger.error(f"Error details: {error}")

        logger.info("Cleaning process completed")

        if os.path.getsize(cleaned_path) == 0:
            logger.error("Error: No valid entries were processed. The cleaned file is empty.")
            return

        metadata = cleaned_path

    logger.info("Starting to create train and val lists")
    spk_utt_map = defaultdict(list)
    spk_id_map = {}
    current_sid = 0

    with open(metadata, encoding="utf-8") as f:
        for line in f:
            utt, spk, language, text, phones, tones, word2ph = line.strip().split("|")
            spk_utt_map[spk].append(line)

            if spk not in spk_id_map:
                spk_id_map[spk] = current_sid
                current_sid += 1

    logger.info(f"Total speakers: {len(spk_id_map)}")

    train_list = []
    val_list = []

    for spk, utts in spk_utt_map.items():
        shuffle(utts)
        val_list += utts[:val_per_spk]
        train_list += utts[val_per_spk:]

    logger.info(f"Initial train list size: {len(train_list)}")
    logger.info(f"Initial val list size: {len(val_list)}")

    if len(val_list) > max_val_total:
        train_list += val_list[max_val_total:]
        val_list = val_list[:max_val_total]

    logger.info(f"Final train list size: {len(train_list)}")
    logger.info(f"Final val list size: {len(val_list)}")

    with open(train_path, "w", encoding="utf-8") as f:
        for line in train_list:
            f.write(line)

    with open(val_path, "w", encoding="utf-8") as f:
        for line in val_list:
            f.write(line)

    logger.info("Train and val lists created")

    logger.info("Updating config file")
    config = json.load(open(config_path, encoding="utf-8"))
    config["data"]["spk2id"] = spk_id_map
    config["data"]["training_files"] = train_path
    config["data"]["validation_files"] = val_path
    config["data"]["n_speakers"] = len(spk_id_map)
    config["num_languages"] = num_languages
    config["num_tones"] = num_tones
    config["symbols"] = symbols

    with open(out_config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    logger.info("Preprocessing completed successfully")

if __name__ == "__main__":
    main()
