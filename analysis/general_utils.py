import constants


def display_name(task):
    return constants.display_name_dict[task]


def find_pii_token_spans(tokenizer, prompt: str, pii_text: str):
    """
    Helper function to automatically find token spans containing PII

    Args:
        tokenizer: The tokenizer used to tokenize the prompt
        prompt: The full prompt text
        pii_text: The PII substring to locate

    Returns:
        List[Tuple[int, int]]: List of (start, end) token spans containing the PII
    """
    tokens = tokenizer.tokenize(prompt)

    char_start = prompt.find(pii_text)
    if char_start == -1:
        return []

    char_end = char_start + len(pii_text) - 1

    token_spans = []
    current_pos = 0

    for i, token in enumerate(tokens):
        token_text = tokenizer.convert_tokens_to_string([token])
        token_end = current_pos + len(token_text) - 1

        if not (token_end < char_start or current_pos > char_end):
            if not token_spans or token_spans[-1][1] + 1 < i:
                token_spans.append((i, i))
            else:
                # Extend the current span
                token_spans[-1] = (token_spans[-1][0], i)

        current_pos += len(token_text)

    return token_spans
