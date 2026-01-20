from typing import Callable, Dict, List, Tuple
from transformer_lens import HookedTransformer
from gencircuits.eap.utils import tokenize_plus
import random


class EntityReplacerCache:
    """Cache for pre-generated entities by token length to avoid repeated tokenization."""
    
    def __init__(self, hooked_model: HookedTransformer):
        self.hooked_model = hooked_model
        self.cache: Dict[int, List[Tuple[str, int]]] = {}  # token_length -> [(entity, token_id), ...]
        self.max_cache_size = 1000  # Maximum entities to cache per token length
        
    def get_cached_entity(self, target_token_length: int, faker_fnc: Callable) -> Tuple[str, int]:
        """Get a cached entity or generate one if not available."""
        if target_token_length not in self.cache:
            self.cache[target_token_length] = []
            
        cache_list = self.cache[target_token_length]
        
        # If we have cached entities for this length, return a random one
        if cache_list:
            return random.choice(cache_list)
            
        # Generate entities in batch to fill the cache
        self._populate_cache(target_token_length, faker_fnc)
        
        if cache_list:
            return random.choice(cache_list)
        else:
            # Fallback - return any entity even if token length doesn't match
            entity = faker_fnc()
            tokens, _, _, _ = tokenize_plus(self.hooked_model, [entity])
            token_id = tokens[0].tolist()[-1]  # Last token
            return entity, token_id
    
    def _populate_cache(self, target_token_length: int, faker_fnc: Callable, batch_size: int = 50):
        """Populate cache by generating entities in batches."""
        max_attempts = 200
        batch_entities = []
        
        for _ in range(max_attempts // batch_size):
            # Generate a batch of candidate entities
            candidates = [faker_fnc() for _ in range(batch_size)]
            
            # Tokenize all candidates at once
            tokens_batch, _, _, _ = tokenize_plus(self.hooked_model, candidates)
            
            # Filter entities that match the target token length
            for i, tokens in enumerate(tokens_batch):
                token_list = tokens.tolist()
                if len(token_list) == target_token_length:
                    entity = candidates[i]
                    token_id = token_list[-1]  # Last token
                    batch_entities.append((entity, token_id))
                    
                    if len(batch_entities) >= 20:  # Enough for this cache
                        break
            
            if len(batch_entities) >= 20:
                break
        
        # Add to cache (limit cache size)
        cache_list = self.cache[target_token_length]
        cache_list.extend(batch_entities)
        if len(cache_list) > self.max_cache_size:
            cache_list[:] = cache_list[-self.max_cache_size:]


def replace_entity_with_fake_optimized(
    document: str,
    start: int,
    end: int,
    hooked_model: HookedTransformer,
    faker_fnc: Callable,
    cache: EntityReplacerCache,
    prefix: str = "",
) -> Tuple[str, int, int]:
    """
    Optimized version of replace_entity_with_fake using caching.
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

    # Get original entity token info - batch with a space prefix
    original_with_space = f" {original_entity}"
    original_tokens_batch, _, _, _ = tokenize_plus(hooked_model, [original_with_space])
    
    # what are the token of the current surname
    original_surname_tokens = original_tokens_batch[0].tolist()
    target_token_length = len(original_surname_tokens)
    original_surname_idx = original_surname_tokens[1][-1]  # Last token with space prefix

    # Get a cached replacement entity
    new_entity, new_surname_idx = cache.get_cached_entity(target_token_length, faker_fnc)

    # Corrupt the sentence
    modified_sentence = f"{prefix} {new_entity}"

    return modified_sentence, original_surname_idx, new_surname_idx
