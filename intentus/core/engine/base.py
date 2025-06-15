# Reference: https://github.com/zou-group/textgrad/blob/main/textgrad/engine/base.py

import hashlib
import diskcache as dc
from abc import ABC, abstractmethod


class EngineLM(ABC):
    system_prompt: str = "You are a helpful, creative, and smart assistant."
    model_string: str

    @abstractmethod
    def generate(self, prompt, system_prompt=None, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        pass


class CachedEngine:
    def __init__(self, cache_path=None):
        super().__init__()
        self.cache_path = cache_path
        if cache_path is not None:
            self.cache = dc.Cache(cache_path)
        else:
            self.cache = None

    def _hash_prompt(self, prompt: str):
        return hashlib.sha256(f"{prompt}".encode()).hexdigest()

    def _check_cache(self, prompt: str):
        if self.cache is not None and prompt in self.cache:
            return self.cache[prompt]
        else:
            return None

    def _save_cache(self, prompt: str, response: str):
        if self.cache is not None:
            self.cache[prompt] = response

    def __getstate__(self):
        # Remove the cache from the state before pickling
        state = self.__dict__.copy()
        del state["cache"]
        return state

    def __setstate__(self, state):
        # Restore the cache after unpickling
        self.__dict__.update(state)
        if self.cache_path is not None:
            self.cache = dc.Cache(self.cache_path)
        else:
            self.cache = None
