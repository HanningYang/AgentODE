
import os
import logging
logging.getLogger("httpx").setLevel(logging.WARNING)

from argparse import ArgumentParser
import numpy as np
import torch
import pandas as pd

from llmode.core import pipeline
from llmode import config
from llmode.core import sampler
from llmode.core import evaluator


parser = ArgumentParser()
parser.add_argument('--port', type=int, default=None)
parser.add_argument('--use_api', type=bool, default=False)
parser.add_argument('--api_model', type=str, default=None)
parser.add_argument('--api_provider', type=str, default="openai",
                    help="API provider to use when API backends are used (e.g., 'openai' or 'deepseek').")
parser.add_argument('--structure_backend', type=str, default=None,
                    help="Backend for ODE structure inference: 'api' | 'openwebui' | 'local'.")
parser.add_argument('--param_backend', type=str, default=None,
                    help="Backend for parameter inference/optimisation: 'api' or 'local' (defaults to structure_backend).")
parser.add_argument('--structure_model', type=str, default=None,
                    help="Model identifier for ODE structure inference (overrides api_model for structure).")
parser.add_argument('--param_model', type=str, default=None,
                    help="Model identifier for parameter inference / optimisation (defaults to structure_model/api_model).")
parser.add_argument('--spec_path', type=str)
parser.add_argument('--log_path', type=str, default="./logs/oscillator1")
parser.add_argument('--problem_name', type=str, default="oscillator1")
parser.add_argument('--run_id', type=int, default=1)
parser.add_argument(
    '--trajectory_bin_width',
    type=float,
    default=None,
    help="Override default Config.trajectory_bin_width (time-bin width for trajectory comparison figures).",
)
args = parser.parse_args()




if __name__ == '__main__':
    # Load config and parameters
    class_config = config.ClassConfig(llm_class=sampler.LocalLLM, sandbox_class=evaluator.LocalSandbox)
    config_kwargs = dict(use_api=args.use_api, api_provider=args.api_provider)
    if args.api_model is not None:
        config_kwargs["api_model"] = args.api_model
    if args.structure_backend is not None:
        config_kwargs["structure_backend"] = args.structure_backend
    if args.param_backend is not None:
        config_kwargs["param_backend"] = args.param_backend
    if args.structure_model is not None:
        config_kwargs["structure_model"] = args.structure_model
    if args.param_model is not None:
        config_kwargs["param_model"] = args.param_model
    if args.trajectory_bin_width is not None:
        config_kwargs["trajectory_bin_width"] = args.trajectory_bin_width
    config = config.Config(**config_kwargs)
    global_max_sample_num = 10000

    # Load prompt specification
    with open(
        os.path.join(args.spec_path),
        encoding="utf-8",
    ) as f:
        specification = f.read()
    
    # Load dataset (if available).
    # For centralized evaluation problems (e.g. synthetic likelihood with AKI),
    # the evaluator does not use these inputs; in that case we safely fall back
    # to an empty list when no train.csv is present.
    problem_name = args.problem_name
    train_path = os.path.join('data', problem_name, 'train.csv')
    if os.path.exists(train_path):
        df = pd.read_csv(train_path)
        data = np.array(df)
        X = data[:, :-1]
        y = data[:, -1].reshape(-1)
        if 'torch' in args.spec_path:
            X = torch.Tensor(X)
            y = torch.Tensor(y)
        data_dict = {'inputs': X, 'outputs': y}
        dataset = {'data': data_dict}
    else:
        # No supervised train.csv for this problem (e.g. AKI); inputs are unused
        # in the centralized synthetic-likelihood evaluation path.
        dataset = []
    
    
    pipeline.main(
        specification=specification,
        inputs=dataset,
        config=config,
        max_sample_nums=global_max_sample_num,
        class_config=class_config,
        problem_name=problem_name,
        log_dir=args.log_path,
    )
