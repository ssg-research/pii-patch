from typing import Callable
from transformer_lens import HookedTransformer
from gencircuits.eap.utils import tokenize_plus


def replace_entity_with_fake(
    document: str,
    sentence: str,
    start: int,
    end: int,
    hooked_model: HookedTransformer,
    faker_fnc: Callable,
) -> str:
    """
    Replaces only the surname (last word) of the entity at [start:end] with a fake one.

    Args:
        document (str): Full text containing the entity.
        sentence (str): Full sentence containing the entity.
        start (int): Character start index of the entity.
        end (int): Character end index of the entity.

    Returns:
        str: Modified sentence with fake surname.
    """
    if start < 0 or end > len(document) or start >= end:
        raise ValueError("Invalid start or end indices.")

    entity_text = document[start:end]
    parts = entity_text.strip().split()
    original_entity = parts[-1] if parts else ""
    if "-" in entity_text:
        # If the entity contains a hyphenated surname, we take the last part after the hyphen
        original_entity = parts[-1].split("-")[-1] if parts[-1] else ""

    if not parts:
        raise ValueError("No valid name found in the specified span.")

    original_surname_tokens, _, _, _ = tokenize_plus(hooked_model, original_entity)
    original_surname_tokens = original_surname_tokens[0].tolist()
    target_token_length = len(original_surname_tokens)

    # Try to find a fake surname with the same token length to maintain tokenization length downstream
    new_entity = ""
    max_attempts = 100
    for i in range(max_attempts):
        candidate_entity = faker_fnc()
        candidate_tokens, _, _, _ = tokenize_plus(hooked_model, candidate_entity)
        candidate_tokens = candidate_tokens[0].tolist()

        if len(candidate_tokens) == target_token_length:
            new_entity = candidate_entity
            break

    # corrupt the sentence
    modified_sentence = sentence.replace(original_entity, new_entity).strip()

    # we take the idx of the word only
    original_surname_idx, _, _, _ = tokenize_plus(hooked_model, f" {original_entity}")
    new_surname_idx, _, _, _ = tokenize_plus(hooked_model, f" {new_entity}")

    return (
        modified_sentence,
        original_surname_idx[0].tolist().pop(),
        new_surname_idx[0].tolist().pop(),
    )
