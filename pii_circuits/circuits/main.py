import argparse
import transformer_lens.utils as utils

from . import data
from . import models
from . import analysis_logitlens
from . import analysis_actpatching
from . import plot_utils

import warnings
warnings.filterwarnings("ignore")

def main(args):

    model = models.load_model(args.model)
    prompts, answers, answer_tokens = data.build_prompts_and_answers(model)

    tokens = model.to_tokens(prompts, prepend_bos=True)
    original_logits, cache = model.run_with_cache(tokens)

    example_prompt = "After John and Mary went to the store, John gave a bottle of milk to"
    example_answer = " Mary"
    utils.test_prompt(example_prompt, example_answer, model, prepend_bos=True)

    analysis_logitlens.run_logit_diff_attribution_and_visualization(model, cache, answer_tokens, prompts, tokens, plot_utils, analysis_logitlens, top_k=3)
    analysis_actpatching.run_activation_patching_analysis(model, cache, tokens, prompts, answer_tokens, original_logits, plot_utils)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",type=str,default="gpt2-small",choices=["gpt2-small","gpt2-medium","gpt2-large","gpt2-xl"],help="Model type")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    main(args)