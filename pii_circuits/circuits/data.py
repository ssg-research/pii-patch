import torch


def build_prompts_and_answers(model):
    prompt_format = [
        "When John and Mary went to the shops,{} gave the bag to",
        "When Tom and James went to the park,{} gave the ball to",
        "When Dan and Sid went to the shops,{} gave an apple to",
        "After Martin and Amy went to the park,{} gave a drink to",
    ]
    names = [
        (" Mary", " John"),
        (" Tom", " James"),
        (" Dan", " Sid"),
        (" Martin", " Amy"),
    ]
    prompts = []
    answers = []
    answer_tokens = []
    for i in range(len(prompt_format)):
        for j in range(2):
            answers.append((names[i][j], names[i][1 - j]))
            answer_tokens.append(
                (
                    model.to_single_token(answers[-1][0]),
                    model.to_single_token(answers[-1][1]),
                )
            )
            # Insert the *incorrect* answer to the prompt, making the correct answer the indirect object.
            prompts.append(prompt_format[i].format(answers[-1][1]))
    answer_tokens = torch.tensor(answer_tokens).to(model.cfg.device)
    # Print prompt/token info for debugging
    # for prompt in prompts:
    #     str_tokens = model.to_str_tokens(prompt)
    #     print("Prompt length:", len(str_tokens))
    #     print("Prompt as tokens:", str_tokens)
    # print(prompts)
    # print(answers)
    return prompts, answers, answer_tokens