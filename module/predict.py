#! modified fairseq_cli.generate
import logging
import os

import numpy as np
import torch
from fairseq import checkpoint_utils, options, tasks, utils
from fairseq.logging import progress_bar


class Predictor:
    def __init__(self, model):
        self.model = model
    
    def predict(self, sample):
        infos = sample['infos']
        inputs = sample['inputs']
        labels = sample['labels']
        with torch.no_grad():
            out = self.model(**inputs)
            probs = torch.softmax(out['logits'], dim=-1).cpu().tolist()
        
        fpros = infos['fpros']
        spros = infos['spros']
        lines = []
        for idx in range(len(probs)):
            line = "{}\t{}\t{}\t{}\t{}\n".format(
                fpros[idx], spros[idx], str(int(labels[idx])), 
                str(round(probs[idx][0], 4)), str(round(probs[idx][1], 4)))
            lines.append(line)
        return lines

def main(args):
    assert args.path is not None, "--path required for generation!"

    os.makedirs(args.results_path, exist_ok=True)
    output_path = os.path.join(args.results_path, "{}.txt".format(args.gen_subset))
    with open(output_path, "w", buffering=1, encoding="utf-8") as h:
        return _main(args, h)


def _main(args, output_file):
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper())
    logger = logging.getLogger("predict")

    utils.import_user_module(args)
    logger.info(args)

    # Fix seed for stochastic decoding
    if args.seed is not None and not args.no_seed_provided:
        np.random.seed(args.seed)
        utils.set_torch_seed(args.seed)

    use_cuda = torch.cuda.is_available() and not args.cpu

    # Load dataset splits
    task = tasks.setup_task(args)
    task.load_dataset(args.gen_subset)

   
    # Load ensemble
    logger.info("loading model(s) from {}".format(args.path))
    models, _model_args = checkpoint_utils.load_model_ensemble(
        utils.split_paths(args.path),
        task=task,
        num_shards=args.checkpoint_shard_count,
    )

    model = models[0]
    if args.fp16:
        model.half()
    if use_cuda and not args.pipeline_model_parallel:
        model.cuda()
    model.prepare_for_inference_(args)

    itr = task.get_batch_iterator(
        dataset=task.dataset(args.gen_subset),
        max_tokens=args.max_tokens,
        max_sentences=args.batch_size,
        max_positions=utils.resolve_max_positions(
            task.max_positions(), *[model.max_positions() for model in models]
        ),
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=args.required_batch_size_multiple,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
        num_workers=args.num_workers,
        data_buffer_size=args.data_buffer_size,
    ).next_epoch_itr(shuffle=False)
    progress = progress_bar.progress_bar(
        itr,
        log_format=args.log_format,
        log_interval=args.log_interval,
        default_log_format=("tqdm" if not args.no_progress_bar else "none"),
    )
    predictor = Predictor(model)
    out_lines = []
    for sample in progress:
        sample = utils.move_to_cuda(sample) if use_cuda else sample
        out_lines.extend(predictor.predict(sample))
    output_file.writelines(out_lines)


def cli_main():
    parser = options.get_generation_parser()
    options.add_model_args(parser)
    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == "__main__":
    cli_main()
