from collections import defaultdict
from typing import List, Dict, Tuple, Optional


class BPETokenizer:

    def __init__(
        self,
        vocab_size: Optional[int] = None,
        num_merges: Optional[int] = None,
        end_of_word="</w>",
        unk_token="<unk>",
        add_bos=False,
        add_eos=False,
        bos_token="<bos>",
        eos_token="<eos>"
    ):
        # User must specify exactly 1 of vocab_size or num_merges
        if (vocab_size is None) == (num_merges is None):
            raise ValueError("Specify exactly one: vocab_size OR num_merges.")

        self.vocab_size = vocab_size
        self.num_merges = num_merges

        self.end_of_word = end_of_word
        self.unk_token = unk_token

        self.add_bos = add_bos
        self.add_eos = add_eos
        self.bos_token = bos_token
        self.eos_token = eos_token

        # Learned data structures
        self.merges: List[Tuple[str, str]] = []
        self.merge_ranks: Dict[Tuple[str, str], int] = {}

        # Token ID maps
        self.token2id = {}
        self.id2token = {}

        self._fitted = False

    # ----------------------------------------------------------
    #                  BPE TRAINING
    # ----------------------------------------------------------

    def _word_to_symbols(self, word: str) -> List[str]:
        """
        Convert a raw word → initial list of characters + end token.

        Reason:
        - Keeping symbols as *list[str]* (not tuples) reduces memory use.
        - This avoids your original recursion-heavy tuple reconstruction.
        """
        return list(word) + [self.end_of_word]

    def _build_initial_vocab(self, sentences: List[str]):
        """
        Build initial vocabulary = list of word-symbol sequences with counts.

        Using dict[str, int], where words are stored as space-separated symbols.

        Reason:
        - Strings are cheaper than tuples.
        - GPT-2 stores words as 't h e </w>' strings internally.
        """
        vocab = defaultdict(int)

        for sent in sentences:
            for word in sent.split():
                symbols = self._word_to_symbols(word)
                key = " ".join(symbols)
                vocab[key] += 1

        return vocab

    def _get_pair_stats(self, vocab):
        """
        Count frequency of all symbol pairs in the vocab.

        Reason:
        - Much faster than your tuple-based nested loops.
        """
        stats = defaultdict(int)

        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                stats[pair] += freq

        return stats

    def _merge_pair(self, pair: Tuple[str, str], vocab):
        """
        Replace all occurrences of pair (A,B) → merged token "AB".

        Reason:
        - Uses string replace with sentinel protection for correctness.
        - Much faster than manual list reconstruction.
        """
        merged_token = pair[0] + pair[1]
        new_vocab = {}

        pattern = " ".join(pair)
        replacement = merged_token

        for word, freq in vocab.items():
            # replace whole pair as a unit
            new_word = word.replace(pattern, replacement)
            new_vocab[new_word] = freq

        return new_vocab

    def fit(self, sentences: List[str]):
        """
        Train BPE merges — now with progress printing.
        """
        vocab = self._build_initial_vocab(sentences)

        # Initial symbols count determines num_merges if vocab_size given
        initial_tokens = set()
        for w in vocab:
            initial_tokens.update(w.split())
        initial_symbol_count = len(initial_tokens)

        if self.vocab_size is not None:
            target_merges = max(self.vocab_size - initial_symbol_count, 0)
        else:
            target_merges = self.num_merges

        print(f"[BPE] Starting training: {target_merges} merges to learn.")

        # Learn merges
        for i in range(target_merges):

            # print progress every 100 merges
            if i % 100 == 0 and i > 0:
                pct = (i / target_merges) * 100
                print(f"[BPE] Merge {i}/{target_merges}  ({pct:.1f}%)")

            stats = self._get_pair_stats(vocab)
            if not stats:
                print("[BPE] No more pairs to merge — stopping early.")
                break

            best = max(stats, key=stats.get)
            self.merges.append(best)
            vocab = self._merge_pair(best, vocab)

        print("[BPE] Finished all merges.")

        # Assign ranks
        self.merge_ranks = {pair: i for i, pair in enumerate(self.merges)}

        # Build actual vocab tokens
        final_tokens = initial_tokens.copy()
        for a, b in self.merges:
            final_tokens.add(a + b)

        final_tokens.add(self.unk_token)
        if self.add_bos:
            final_tokens.add(self.bos_token)
        if self.add_eos:
            final_tokens.add(self.eos_token)

        sorted_tokens = sorted(final_tokens)
        self.token2id = {tok: i for i, tok in enumerate(sorted_tokens)}
        self.id2token = {i: tok for tok, i in self.token2id.items()}

        self._fitted = True
        print(f"[BPE] Training complete. Final vocab size = {len(self.token2id)}")


    # ----------------------------------------------------------
    #                  ENCODING
    # ----------------------------------------------------------

    def _encode_word(self, word: str):
        symbols = self._word_to_symbols(word)

        # Greedy merge loop
        while True:
            pairs = [(symbols[i], symbols[i+1]) for i in range(len(symbols)-1)]

            best_pair = None
            best_rank = None

            for p in pairs:
                if p in self.merge_ranks:
                    rank = self.merge_ranks[p]
                    if best_rank is None or rank < best_rank:
                        best_rank = rank
                        best_pair = p

            if best_pair is None:
                break

            merged = best_pair[0] + best_pair[1]
            new_symbols = []

            i = 0
            while i < len(symbols):
                if i < len(symbols)-1 and symbols[i]==best_pair[0] and symbols[i+1]==best_pair[1]:
                    new_symbols.append(merged)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1

            symbols = new_symbols

        return symbols

    def encode(self, text: str, return_ids=False):
        if not self._fitted:
            raise RuntimeError("Tokenizer must be fitted before encoding.")

        tokens = []
        if self.add_bos:
            tokens.append(self.bos_token)

        for word in text.split():
            tokens.extend(self._encode_word(word))

        if self.add_eos:
            tokens.append(self.eos_token)

        if return_ids:
            return [self.token2id.get(t, self.token2id[self.unk_token]) for t in tokens]

        return tokens

    # ----------------------------------------------------------
    #                  DECODING
    # ----------------------------------------------------------

    def decode(self, tokens):
        if not self._fitted:
            raise RuntimeError("Tokenizer must be fitted first.")

        if tokens and isinstance(tokens[0], int):
            tokens = [self.id2token[t] for t in tokens]

        words = []
        current = ""

        for tok in tokens:
            if tok in (self.bos_token, self.eos_token):
                continue

            if tok.endswith(self.end_of_word):
                current += tok[:-len(self.end_of_word)]
                words.append(current)
                current = ""
            else:
                current += tok

        if current:
            words.append(current)

        return " ".join(words)
