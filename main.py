
import os
from argparse import ArgumentParser
import numpy as np
import torch
import pandas as pd

from llmode import pipeline
from llmode import config
from llmode import sampler
from llmode import evaluator


parser = ArgumentParser()
parser.add_argument('--port', type=int, default=None)
parser.add_argument('--use_api', type=bool, default=False)
parser.add_argument('--api_model', type=str, default="gpt-3.5-turbo")
parser.add_argument('--spec_path', type=str)
parser.add_argument('--log_path', type=str, default="./logs/oscillator1")
parser.add_argument('--problem_name', type=str, default="oscillator1")
parser.add_argument('--run_id', type=int, default=1)
args = parser.parse_args()




if __name__ == '__main__':
    # Load config and parameters
    class_config = config.ClassConfig(llm_class=sampler.LocalLLM, sandbox_class=evaluator.LocalSandbox)
    config = config.Config(use_api = args.use_api, 
                           api_model = args.api_model,)
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
