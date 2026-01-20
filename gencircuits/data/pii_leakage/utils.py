import random
import re
from faker import Faker
from transformer_lens import HookedTransformer
from constants import PREFIXES
from gencircuits.data.pii_leakage.optimized_pii_replacer import replace_entity_with_fake_optimized, EntityReplacerCache

fake = Faker()

def select_random_prefix(pii_class: str) -> str:
    """
    Select a random prefix for the given PII class.
    
    Args:
        pii_class (str): The type of PII (e.g., "PERSON", "LOC", etc.).
    
    Returns:
        str: A randomly selected prefix for the PII class.
    """
    prefixes = PREFIXES.get(pii_class, [])
    return random.choice(prefixes) if prefixes else ""


def create_clean_sentence(document: str, start: int, end: int, prefix: str) -> str:
    """
    Create a clean sentence by extracting the text from the document
    between the start and end indices.

    Args:
        document (str): The full text.
        start (int): Character start index of the entity.
        end (int): Character end index of the entity.

    Returns:
        str: The extracted clean sentence.
    """
    return f"{prefix} {document[start:end].strip()}"


def replace_pii(
    pii_type: str,
    document: str,
    clean_sentence: str,
    start: int,
    end: int,
    hooked_model: HookedTransformer,
    cache: EntityReplacerCache,
    prefix: str = "",
) -> tuple:
    if pii_type == "PERSON":
        corrupted_sentence, baseline_idx, corrputed_idx = replace_entity_with_fake_optimized(
            document=document,
            prefix=prefix,
            start=start,
            end=end,
            hooked_model=hooked_model,
            cache=cache,
            faker_fnc=fake.last_name,
        )
    elif pii_type == "LOC":
        corrupted_sentence, baseline_idx, corrputed_idx = replace_location_with_fake(
            document=document,
            sentence=clean_sentence,
            start=start,
            end=end,
            hooked_model=hooked_model,
            cache=cache,
        )
    elif pii_type == "DEM":
        corrupted_sentence, baseline_idx, corrputed_idx = replace_dem_with_fake(
            document=document,
            sentence=clean_sentence,
            start=start,
            end=end,
            hooked_model=hooked_model,
            cache=cache,
        )
    return corrupted_sentence, baseline_idx, corrputed_idx


def extract_word_window(
    document: str,
    start: int,
    end: int,
    window_before: int,
    window_after: int,
) -> str:
    """
    Returns a span of text containing `window_before` words before and `window_after` words after the entity,
    based on character start and end indices.

    Args:
        document (str): The full text.
        start (int): Character start index of the entity.
        end (int): Character end index of the entity.
        window_before (int): Number of words before the entity to include.
        window_after (int): Number of words after the entity to include.

    Returns:
        str: The extracted window of text.
    """
    if start < 0 or end > len(document) or start >= end:
        raise ValueError("Invalid start or end indices.")

    # Tokenize the text into words with character positions
    word_matches = list(re.finditer(r"\b\w+\b", document))

    # Find the index of the word that includes the start position
    entity_word_index = None
    for i, match in enumerate(word_matches):
        if (
            match.start() <= start < match.end()
            or match.start() < end <= match.end()
            or (start <= match.start() and end >= match.end())
        ):
            entity_word_index = i
            break

    if entity_word_index is None:
        raise ValueError("Entity does not align with any word.")

    # Calculate window range
    start_index = max(0, entity_word_index - window_before)
    end_index = min(len(word_matches), entity_word_index + window_after + 1)

    # Extract start and end char positions from word boundaries
    window_start_char = word_matches[start_index].start()
    window_end_char = word_matches[end_index + 1].end()

    result = document[window_start_char:window_end_char].strip()
    result = result.replace("\n", " ").replace("\r", " ")
    return result


def replace_surname_with_fake(
    document: str, sentence: str, start: int, end: int, hooked_model: HookedTransformer, cache: EntityReplacerCache
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
    return replace_entity_with_fake_optimized(
        document, sentence, start, end, hooked_model, fake.last_name, cache
    )


def replace_location_with_fake(
    document: str, sentence: str, start: int, end: int, hooked_model: HookedTransformer, cache: EntityReplacerCache
) -> str:
    """
    Replaces the entire location entity at [start:end] with a fake one.

    Args:
        document (str): Full text containing the entity.
        sentence (str): Full sentence containing the entity.
        start (int): Character start index of the entity.
        end (int): Character end index of the entity.

    Returns:
        tuple: (modified_sentence, original_location_tokens, new_location_tokens)
    """

    modified_sentence, original_location_tokens, new_location_tokens = (
        replace_entity_with_fake_optimized(
            document, sentence, start, end, hooked_model, fake.country, cache
        )
    )

    return modified_sentence, original_location_tokens, new_location_tokens


def replace_organisation_with_fake(
    document: str, sentence: str, start: int, end: int, hooked_model: HookedTransformer, cache: EntityReplacerCache
) -> str:
    """
    Replaces the entire organization entity at [start:end] with a fake one.

    Args:
        document (str): Full text containing the entity.
        sentence (str): Full sentence containing the entity.
        start (int): Character start index of the entity.
        end (int): Character end index of the entity.

    Returns:
        tuple: (modified_sentence, original_organisation_tokens, new_organisation_tokens)
    """
    modified_sentence, original_location_tokens, new_location_tokens = (
        replace_entity_with_fake_optimized(
            document, sentence, start, end, hooked_model, fake.company, cache
        )
    )

    return modified_sentence, original_location_tokens, new_location_tokens


def gen_code():
    # generate a random 4 or 5 digit code
    part_one = str(fake.random_int(min=1000, max=99999))
    part_two = str(fake.random_int(min=10, max=99))
    return f"{part_one}/{part_two}"


def replace_code_with_fake(
    document: str, sentence: str, start: int, end: int, hooked_model: HookedTransformer, cache: EntityReplacerCache
) -> str:
    """
    Replaces the entire organization entity at [start:end] with a fake one.

    Args:
        document (str): Full text containing the entity.
        sentence (str): Full sentence containing the entity.
        start (int): Character start index of the entity.
        end (int): Character end index of the entity.

    Returns:
        tuple: (modified_sentence, original_organisation_tokens, new_organisation_tokens)
    """

    modified_sentence, original_location_tokens, new_location_tokens = (
        replace_entity_with_fake_optimized(document, sentence, start, end, hooked_model, gen_code, cache)
    )

    return modified_sentence, original_location_tokens, new_location_tokens


def gen_demos() -> list:
    demos = """        
        white, black, African American, Afro-Caribbean, Asian, South Asian, East Asian, Southeast Asian, 
        Middle Eastern, Arab, Persian, Kurdish, Turkish, British, Irish, Scottish, Welsh, German, Italian, 
        French, Spanish, Portuguese, Greek, Russian, Ukrainian, Polish, Jewish, Ashkenazi, Sephardi, Latino, 
        Hispanic/Latino, Mexican, Puerto Rican, Cuban, Colombian, Brazilian, Argentine, Chilean, mixed, biracial, 
        multiracial, Indigenous, Native, American, Alaska, Native, Nations, Inuit, Māori, Aboriginal, Australian, 
        Pacific Islander, Hawaiian, Samoan, Tongan, Fijian, Romani, Basque, Catalan, Sardinian, Korean, Japanese, 
        Chinese, Filipino, Vietnamese, Thai, Cambodian, Laotian, Malaysian, Indonesian, Indian, Pakistani, 
        Bangladeshi, Sri Lankan, Nepali, Bhutanese, Ethiopian, Somali, Nigerian, Kenyan, Ghanaian, 
        Jamaican, Haitian, Christian, Catholic, Protestant, Orthodox Christian, Muslim, Hindu, Buddhist, 
        Jewish (religion), Sikh, Taoist, Shinto, Jain, atheist, agnostic, secular/none, other
    """
    unique_demos = set(x.strip() for x in demos.split(",") if x.strip())
    return random.choice(list(unique_demos))


def replace_dem_with_fake(
    document: str, sentence: str, start: int, end: int, hooked_model: HookedTransformer, cache: EntityReplacerCache
) -> str:
    """
    Replaces the entire organization entity at [start:end] with a fake one.

    Args:
        document (str): Full text containing the entity.
        sentence (str): Full sentence containing the entity.
        start (int): Character start index of the entity.
        end (int): Character end index of the entity.

    Returns:
        tuple: (modified_sentence, original_organisation_tokens, new_organisation_tokens)
    """

    modified_sentence, original_location_tokens, new_location_tokens = (
        replace_entity_with_fake_optimized(
            document, sentence, start, end, hooked_model, gen_demos, cache
        )
    )

    return modified_sentence, original_location_tokens, new_location_tokens
