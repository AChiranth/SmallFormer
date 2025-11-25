from collections import defaultdict
from typing import List, Tuple, Dict, Optional

from collections import defaultdict
from typing import List, Tuple, Dict, Optional


class BPETokenizer:
    def __init__(
        self,
        vocab_size: Optional[int] = None,
        num_merges: Optional[int] = None,
        end_of_word: str = "</w>",
        unk_token: str = "<unk>",
        add_bos: bool = False,
        add_eos: bool = False,
        bos_token: str = "<bos>",
        eos_token: str = "<eos>",
    ):
        """
        Either specify vocab_size or num_merges.

        add_bos / add_eos:
            If True, the tokenizer will add <bos> and/or <eos> to the encoded sequence;
            they will also be added to the vocabulary.
        """
        # Required settings
        if vocab_size is None and num_merges is None:
            raise ValueError("You must provide either vocab_size or num_merges.")

        # Parameters
        self.vocab_size = vocab_size
        self.num_merges = num_merges
        self.end_of_word = end_of_word
        self.unk_token = unk_token

        # BOS/EOS settings
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.bos_token = bos_token
        self.eos_token = eos_token

        # Learned data
        self.merges: List[Tuple[str, str]] = []
        self.merge_ranks: Dict[Tuple[str, str], int] = {}
        self.alphabet: set[str] = set()
        self.token2id: Dict[str, int] = {}
        self.id2token: Dict[int, str] = {}

        self._fitted = False

    # ----- Training -----

    def _build_initial_vocab(self, sentences: List[str]) -> Dict[Tuple[str, ...], int]:
        vocab: Dict[Tuple[str, ...], int] = defaultdict(int)

        for sent in sentences:
            for word in sent.split():
                symbols = list(word) + [self.end_of_word]
                vocab[tuple(symbols)] += 1
                self.alphabet.update(symbols)

        return vocab

    @staticmethod
    def _get_pair_stats(vocab):
        stats = defaultdict(int)
        for word, freq in vocab.items():
            for i in range(len(word) - 1):
                stats[(word[i], word[i + 1])] += freq
        return stats

    @staticmethod
    def _merge_pair_in_vocab(pair, vocab):
        merged = pair[0] + pair[1]
        new_vocab = {}

        for word, freq in vocab.items():
            new_word = []
            i = 0
            while i < len(word):
                if (
                    i < len(word) - 1
                    and word[i] == pair[0]
                    and word[i + 1] == pair[1]
                ):
                    new_word.append(merged)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_vocab[tuple(new_word)] = freq

        return new_vocab

    def _build_final_token_vocab(self):
        tokens = set(self.alphabet)
        tokens.add(self.unk_token)

        # Add merge symbols
        for a, b in self.merges:
            tokens.add(a + b)

        # Add bos/eos if required
        if self.add_bos:
            tokens.add(self.bos_token)
        if self.add_eos:
            tokens.add(self.eos_token)

        tokens_list = sorted(tokens)
        self.token2id = {t: i for i, t in enumerate(tokens_list)}
        self.id2token = {i: t for t, i in self.token2id.items()}

    def fit(self, sentences: List[str]):
        vocab = self._build_initial_vocab(sentences)

        initial_symbols = len(self.alphabet)
        if self.vocab_size is not None and self.num_merges is None:
            self.num_merges = max(0, self.vocab_size - initial_symbols)

        # Learn merges
        for _ in range(self.num_merges):
            stats = self._get_pair_stats(vocab)
            if not stats:
                break
            best = max(stats, key=stats.get)
            self.merges.append(best)
            vocab = self._merge_pair_in_vocab(best, vocab)

        self.merge_ranks = {pair: i for i, pair in enumerate(self.merges)}
        self._build_final_token_vocab()

        self._fitted = True

    # ----- Encoding -----

    def _encode_word(self, word: str) -> List[str]:
        symbols = list(word) + [self.end_of_word]

        while True:
            # get all pairs
            pairs = [(symbols[i], symbols[i + 1]) for i in range(len(symbols) - 1)]

            candidate = None
            best_rank = None

            for p in pairs:
                if p in self.merge_ranks:
                    rank = self.merge_ranks[p]
                    if best_rank is None or rank < best_rank:
                        best_rank = rank
                        candidate = p

            if candidate is None:
                break

            merged = candidate[0] + candidate[1]
            new_symbols = []
            i = 0
            while i < len(symbols):
                if (
                    i < len(symbols) - 1
                    and symbols[i] == candidate[0]
                    and symbols[i + 1] == candidate[1]
                ):
                    new_symbols.append(merged)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols

        return symbols

    def encode(self, text: str, return_ids: bool = False):
        if not self._fitted:
            raise RuntimeError("Tokenizer must be fitted first.")

        tokens = []

        if self.add_bos:
            tokens.append(self.bos_token)

        for word in text.strip().split():
            tokens.extend(self._encode_word(word))

        if self.add_eos:
            tokens.append(self.eos_token)

        if return_ids:
            return [self.token2id.get(t, self.token2id[self.unk_token]) for t in tokens]
        return tokens

    # ----- Decoding -----

    def decode(self, tokens: List[str] | List[int]) -> str:
        if not self._fitted:
            raise RuntimeError("Tokenizer must be fitted first.")

        # Convert ids → tokens if needed
        if tokens and isinstance(tokens[0], int):
            tokens = [self.id2token[t] for t in tokens]

        # Strip BOS/EOS if present
        filtered = []
        for t in tokens:
            if t == self.bos_token or t == self.eos_token:
                continue
            filtered.append(t)

        words = []
        current = ""

        for tok in filtered:
            if tok.endswith(self.end_of_word):
                piece = tok[: -len(self.end_of_word)]
                current += piece
                words.append(current)
                current = ""
            else:
                current += tok

        if current:
            words.append(current)

        return " ".join(words)
