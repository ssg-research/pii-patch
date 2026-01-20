import os
import random
from typing import Optional

import pandas as pd
from datasets import load_dataset
from faker import Faker
from transformer_lens import HookedTransformer

from gencircuits.data.pii_leakage.utils import (
    gen_demos,
    select_random_prefix,
)
from gencircuits.eap.utils import tokenize_plus
from constants import model_display_name_dict, original_model_name

fake = Faker()
WINDOW_SIZE = 3


def extract_individual_names(entity_text: str) -> list:
    """
    Extract individual names from an entity text.
    Handles:
    - Single names: "John" -> ["John"]
    - Multiple names: "John Smith" -> ["John", "Smith"]
    - Hyphenated names: "Mary-Jane" -> ["Mary", "Jane"]
    - Mixed: "Jean-Claude Smith" -> ["Jean", "Claude", "Smith"]
    - Locations: "New York" -> ["New", "York"] (for LOC type)

    Returns only alphabetic single words.
    """
    names = []

    # Split by spaces first
    words = entity_text.strip().split()

    for word in words:
        # Handle hyphenated names/places
        if "-" in word:
            hyphen_parts = word.split("-")
            for part in hyphen_parts:
                part = part.strip()
                if part and part.isalpha() and len(part) > 1:  # At least 2 characters
                    names.append(part)
        else:
            # Single word
            if word.isalpha() and len(word) > 1:  # At least 2 characters
                names.append(word)

    return names


def generate_pii_task_data(
    dataset: pd.DataFrame,
    pii_type: str,
    hooked_model: HookedTransformer,
    window_size: int = WINDOW_SIZE,
) -> pd.DataFrame:
    """
    Generate a DataFrame for a specific PII type from the dataset.
    Simplified approach:
    1. Find all PII entities of the given type
    2. Create clean examples with prefixes
    3. Generate corrupted examples with single-token replacements
    4. Ensure token length consistency

    Args:
        dataset (pd.DataFrame): The input dataset containing PII data.
        pii_type (str): The type of PII to filter by.

    Returns:
        pd.DataFrame: A DataFrame containing the specified PII type.
    """
    # Step 1: Extract all PII entities of the given type
    pii_entities = []

    for row in dataset.itertuples():
        # Find the first annotator with entity_mentions
        tab_annotations = None
        for _, ann in row.annotations.items():
            if ann is not None and "entity_mentions" in ann:
                tab_annotations = ann["entity_mentions"]
                break

        if tab_annotations is None:
            continue

        for mention in tab_annotations:
            if mention["entity_type"] == pii_type:
                document = row.text
                entity_text = document[
                    mention["start_offset"] : mention["end_offset"]
                ].strip()

                # Extract individual names from the entity
                extracted_names = extract_individual_names(entity_text)
                pii_entities.extend(extracted_names)

    unique_entities = list(set(pii_entities))

    if not unique_entities:
        print(f"No valid single-word {pii_type} entities found")
        return pd.DataFrame(
            columns=[
                "clean",
                "corrupted",
                "corrupted_hard",
                "correct_idx",
                "incorrect_idx",
            ]
        )

    print(f"Found {len(unique_entities)} unique {pii_type} entities")
    print("Sample entities:", unique_entities[:5])

    # Step 2: Create clean examples with prefixes
    clean_sentences = []
    prefixes = []  # Store prefixes to reuse for corrupted sentences

    for entity in unique_entities:
        prefix = select_random_prefix(pii_type)
        prefixes.append(prefix)  # Save the prefix
        clean_sentence = f" {prefix} {entity}"
        clean_sentences.append(clean_sentence)

    # Step 3: Generate corrupted examples with exact token length matching
    faker_func = get_faker_function(pii_type)
    valid_pairs = []

    for i, entity in enumerate(unique_entities):
        prefix = prefixes[i]  # Use the same prefix as the clean sentence
        clean_sentence = clean_sentences[i]

        # Find a replacement that results in exact token length match
        replacement_entity, corrupted_sentence, clean_tokens_no_pad, corrupted_tokens_no_pad = find_equal_length_replacement(
            hooked_model, clean_sentence, entity, faker_func, prefix
        )
        
        if replacement_entity is not None and corrupted_sentence is not None:
            # Verify token lengths match exactly (no padding)
            assert len(clean_tokens_no_pad) == len(corrupted_tokens_no_pad), \
                f"Token length mismatch: clean={len(clean_tokens_no_pad)}, corrupted={len(corrupted_tokens_no_pad)}"
            
            # Get the token indices for the actual entities
            clean_entity_tokens, _, _, _ = tokenize_plus(hooked_model, [f" {entity}"])
            corrupted_entity_tokens, _, _, _ = tokenize_plus(hooked_model, [f" {replacement_entity}"])
            
            # Get the actual entity token (last non-padding token)
            clean_entity_token_list = [t for t in clean_entity_tokens[0].tolist() if t != 50256]
            corrupted_entity_token_list = [t for t in corrupted_entity_tokens[0].tolist() if t != 50256]
            
            clean_entity_token = clean_entity_token_list[-1]  # Last non-padding token
            corrupted_entity_token = corrupted_entity_token_list[-1]  # Last non-padding token
            
            valid_pairs.append({
                "clean": clean_sentence,
                "corrupted": corrupted_sentence,
                "corrupted_hard": "",
                "correct_idx": clean_entity_token,
                "incorrect_idx": corrupted_entity_token,
            })

    print(f"Generated {len(valid_pairs)} valid {pii_type} examples with exact token length matches")

    final_frame = pd.DataFrame(
        valid_pairs,
        columns=[
            "clean",
            "corrupted",
            "corrupted_hard",
            "correct_idx",
            "incorrect_idx",
        ],
    )
    return final_frame


def get_faker_function(pii_type: str):
    """Get the appropriate faker function for the PII type."""
    if pii_type == "PERSON":
        return fake.first_name  # Use first names for simplicity
    elif pii_type == "LOC":
        return fake.city  # Use city names
    elif pii_type == "DEM":
        return gen_demos
    else:
        return fake.word


def gen_demos():
    """Generate demographic terms."""
    demo_terms = [
        # Gender
        "male", "female", "nonbinary",
        # Marital status
        "married", "single", "divorced", "widowed",
        # Employment
        "student", "retired", "employed", "unemployed", "self-employed",
        # Nationalities/Ethnicities
        "american", "british", "english", "welsh", "scottish", "irish", "turkish", 
        "german", "french", "italian", "spanish", "polish", "russian", "ukrainian",
        "chinese", "japanese", "korean", "indian", "pakistani", "bangladeshi",
        "mexican", "brazilian", "canadian", "australian", "south-african",
        "dutch", "belgian", "swedish", "norwegian", "danish", "finnish", "austrian",
        "swiss", "portuguese", "greek", "czech", "hungarian", "romanian", "bulgarian",
        "croatian", "serbian", "slovenian", "slovak", "estonian", "latvian", "lithuanian",
        "thai", "vietnamese", "filipino", "indonesian", "malaysian", "singaporean",
        "egyptian", "moroccan", "algerian", "tunisian", "nigerian", "kenyan", "ethiopian",
        "ghanaian", "sudanese", "zambian", "zimbabwean", "tanzanian", "ugandan",
        "armenian", "georgian", "azerbaijani", "iranian", "iraqi", "saudi", "jordanian",
        "lebanese", "syrian", "israeli", "palestinian", "kuwaiti", "qatari", "emirati",
        "afghan", "nepalese", "sri-lankan", "burmese", "cambodian", "laotian",
        "argentinian", "chilean", "peruvian", "colombian", "venezuelan", "ecuadorian",
        "uruguayan", "paraguayan", "bolivian", "cuban", "puerto-rican", "dominican",
        # Races/Ethnic groups
        "white", "black", "asian", "hispanic", "latino", "native-american", 
        "middle-eastern", "mixed-race", "caucasian", "african-american",
        "pacific-islander", "aboriginal", "inuit", "indigenous", "romani",
        "kurdish", "berber", "bedouin", "maori", "polynesian", "melanesian",
        # Religions
        "christian", "muslim", "jewish", "hindu", "buddhist", "atheist", "agnostic",
        "sikh", "orthodox", "catholic", "protestant", "evangelical", "baptist",
        "methodist", "presbyterian", "lutheran", "mormon", "jehovah-witness",
        "shia", "sunni", "sufi", "jain", "taoist", "confucian", "shinto",
        "zoroastrian", "bahai", "druze", "pagan", "wiccan", "spiritual",
        # Age groups
        "young", "elderly", "middle-aged", "teenage", "adult", "senior", "minor",
        "toddler", "child", "adolescent", "twenty-something", "thirty-something"
    ]
    return random.choice(demo_terms)


def find_equal_length_replacement(
    hooked_model: HookedTransformer,
    clean_sentence: str,
    original_entity: str,
    faker_func,
    prefix: str,
    max_attempts: int = 100,
):
    """
    Find a replacement entity that results in the same token count as the clean sentence.
    No padding should be used - we iterate until we find exact token length matches.
    """
    # Get the exact token count of the clean sentence (without padding)
    clean_tokens, _, _, _ = tokenize_plus(hooked_model, [clean_sentence])
    clean_token_list = clean_tokens[0].tolist()
    
    # Remove padding tokens (50256) to get actual length
    actual_clean_tokens = [t for t in clean_token_list if t != 50256]
    target_token_count = len(actual_clean_tokens)
    
    for attempt in range(max_attempts):
        candidate = faker_func()
        
        # Ensure it's a single word
        if len(candidate.split()) == 1 and candidate.isalpha():
            # Create the corrupted sentence with this candidate
            corrupted_sentence = f" {prefix} {candidate}"
            
            # Check if the token count matches exactly (without padding)
            corrupted_tokens, _, _, _ = tokenize_plus(hooked_model, [corrupted_sentence])
            corrupted_token_list = corrupted_tokens[0].tolist()
            actual_corrupted_tokens = [t for t in corrupted_token_list if t != 50256]
            corrupted_token_count = len(actual_corrupted_tokens)
            
            if corrupted_token_count == target_token_count:
                return candidate, corrupted_sentence, actual_clean_tokens, actual_corrupted_tokens
    
    # If we can't find an exact match, return None to skip this example
    return None, None, None, None


def load_tab():
    """
    Load the ECHR tab dataset from the Hugging Face hub.
    Returns:
        pd.DataFrame: The loaded dataset as a pandas DataFrame.
    """
    dataset = load_dataset(
        "mattmdjaga/text-anonymization-benchmark-train", split="train"
    )
    print("Dataset loaded successfully.")
    print(f"Number of records: {len(dataset)}")
    return dataset.to_pandas()


def save_file(filename: str, df: Optional[pd.DataFrame] = None):
    # Save the DataFrame to a CSV file
    df.to_csv(filename, index=True)
    print(f"File saved: {filename}")


if __name__ == "__main__":
    # Load the dataset once
    print("Loading dataset...")
    tab_dataset = load_tab()

    pii_types = [
        "PERSON",
        "LOC",
        "DEM",
    ]

    # for model in ['gpt2-small', 'pythia-1b-deduped']:
    for model in ['qwen3']:
        print(f"Processing model: {model}")
        # original_name = original_model_name(model)

        # Load the specific model for this iteration
        # print(f"Loading model: {original_name}")
        hooked_model = HookedTransformer.from_pretrained(
            'qwen3-0.6b',
            center_writing_weights=False,
            center_unembed=False,
            fold_ln=False,
            device="cuda",
        )

        for pii_type in pii_types:
            print(f"Generating PII task data for {pii_type}...")
            pii_task_data = generate_pii_task_data(tab_dataset, pii_type, hooked_model)
            pii_type_label = pii_type.lower()
            os.makedirs(f"gencircuits/data/pii_leakage_{pii_type_label}", exist_ok=True)
            print(f"Saving PII task data for {pii_type} and model {model}...")
            save_file(
                f"gencircuits/data/pii_leakage_{pii_type_label}/{model}.csv",
                pii_task_data,
            )

        # Clean up GPU memory after processing each model
        del hooked_model
