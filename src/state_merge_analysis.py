import torch
import numpy as np
from collections import defaultdict
import pandas as pd
from pathlib import Path
import pickle
from datetime import datetime
import os
import time
import glob
import hashlib
import gc
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence


class StateMergerAnalyzer:
    """
    Analyzes pairwise state mergers to reproduce Figure 7 from the paper.

    The key insight: prefixes that "agree" on all future continuations
    should have their hidden states merge during training.
    """

    def __init__(self, model, device, L_train=10, epsilon=0.1):
        self.model = model
        self.device = device
        self.L_train = L_train
        self.epsilon = epsilon
        self.cache_dir = Path("./cache/state_merger")
        self.merge_data = None
        self.heatmap_data = None
        self.merge_fractions = None
        self.threshold_data = None

    def generate_all_prefixes(self, max_length):
        """Generate all binary prefixes up to max_length"""
        prefixes = []
        for length in range(1, max_length + 1):
            for i in range(2**length):
                # Convert integer to binary prefix
                prefix = [(i >> j) & 1 for j in range(length-1, -1, -1)]
                prefixes.append(prefix)
        return prefixes

    def get_all_continuations(self, prefix, continuation_length):
        """Get all possible continuations of a given prefix up to continuation_length"""
        continuations = []
        for length in range(1, continuation_length + 1):
            for i in range(2**length):
                continuation = [(i >> j) & 1 for j in range(length-1, -1, -1)]
                full_sequence = prefix + continuation
                parity = sum(full_sequence) % 2
                continuations.append((continuation, parity))
        return continuations

    def prefixes_agree(self, prefix1, prefix2, max_continuation_length):
        """
        Check if two prefixes agree on all continuations up to max_continuation_length.

        Two prefixes "agree" if for every possible continuation that can be applied
        to both prefixes (within L_train constraint), the resulting sequences have
        the same parity.
        """

        max_len1 = self.L_train - len(prefix1)
        max_len2 = self.L_train - len(prefix2)
        max_feasible_cont_len = min(
            max_continuation_length, max_len1, max_len2)

        if max_feasible_cont_len <= 0:
            return True

        for cont_length in range(1, max_feasible_cont_len + 1):
            for i in range(2**cont_length):
                continuation = [(i >> j) & 1 for j in range(
                    cont_length-1, -1, -1)]

                if (len(prefix1) + cont_length <= self.L_train and
                        len(prefix2) + cont_length <= self.L_train):

                    seq1 = prefix1 + continuation
                    seq2 = prefix2 + continuation
                    parity1 = sum(seq1) % 2
                    parity2 = sum(seq2) % 2

                    if parity1 != parity2:
                        return False

        return True

    def convert_agreeing_pairs_format(self, agreeing_pairs_list):
        """Convert list of prefix pairs to dict organized by length pairs"""
        organized_pairs = {}

        for prefix1, prefix2 in agreeing_pairs_list:
            m1, m2 = len(prefix1), len(prefix2)
            key = (m1, m2)

            if key not in organized_pairs:
                organized_pairs[key] = []

            pair_dict = {
                'prefix1': prefix1,
                'prefix2': prefix2,
                'agrees': True
            }
            organized_pairs[key].append(pair_dict)

        return organized_pairs

    def get_hidden_state(self, sequence):
        """Get the final hidden state for a given sequence"""
        self.model.eval()

        with torch.no_grad():
            try:
                L = len(sequence)
                tensor = torch.zeros(
                    L, 2, device=self.device, dtype=torch.float32)

                for i, bit in enumerate(sequence):
                    tensor[i, int(bit)] = 1.0

                packed = pack_sequence([tensor], enforce_sorted=False)

                output, hidden = self.model.rnn(packed)
                unpacked, lengths = pad_packed_sequence(
                    output, batch_first=True)

                # Get the final hidden state
                final_hidden = unpacked[0, -1, :]

                result = final_hidden.cpu().numpy().copy()
                del tensor, packed, output, hidden, unpacked, final_hidden
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                return result

            except Exception as e:
                # Clean up on error
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise e

    def states_merged(self, state1, state2):
        """Check if two hidden states have merged (distance < epsilon) - SAFE VERSION"""
        try:
            if hasattr(state1, 'cpu'):
                state1 = state1.cpu().numpy()
            if hasattr(state2, 'cpu'):
                state2 = state2.cpu().numpy()

            distance = np.linalg.norm(state1 - state2)
            result = distance < self.epsilon

            del distance

            return result

        except Exception as e:
            print(f"Error in states_merged: {e}")
            return False

    def find_agreeing_prefix_pairs(self, prefixes, max_continuation_length):
        """
        Find all pairs of prefixes that agree on all continuations.

        This is the computationally expensive part, but necessary for the analysis.
        """
        print(
            f"Finding agreeing prefix pairs among {len(prefixes)} prefixes...")
        agreeing_pairs = []

        # Only consider pairs where both prefixes are <= max_continuation_length
        valid_prefixes = [p for p in prefixes if len(
            p) <= max_continuation_length]

        total_pairs = len(valid_prefixes) * (len(valid_prefixes) - 1) // 2
        checked = 0

        for i, prefix1 in enumerate(valid_prefixes):
            for j, prefix2 in enumerate(valid_prefixes[i+1:], i+1):
                checked += 1
                if checked % 10000 == 0:
                    print(
                        f"  Checked {checked}/{total_pairs} pairs ({100*checked/total_pairs:.1f}%)")

                if self.prefixes_agree(prefix1, prefix2, max_continuation_length):
                    agreeing_pairs.append((prefix1, prefix2))

        print(f"Found {len(agreeing_pairs)} agreeing prefix pairs")
        return agreeing_pairs

    def _get_cache_filename(self, max_continuation_length):
        """Generate cache filename based on model parameters"""

        model_str = str(sorted(self.model.state_dict().items())
                        [:5])  # First 5 items for speed
        params_str = f"L{self.L_train}_eps{self.epsilon}_cont{max_continuation_length}"

        # Create hash
        hash_input = f"{model_str}_{params_str}".encode()
        model_hash = hashlib.md5(hash_input).hexdigest()[:8]

        return self.cache_dir / f"merger_analysis_{model_hash}.pkl"

    # Core analysis (everything above is a helper function, make sure you understand this below) -----------------

    def analyze_state_mergers(self, max_continuation_length=3):
        """Analyze state mergers with caching"""
        cache_file = self._get_cache_filename(max_continuation_length)

        if cache_file.exists():
            try:
                print(f"📁 Loading cached results from: {cache_file}")
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)

                merge_data = cached_data['merge_data']
                merge_fractions = cached_data['merge_fractions']
                cache_params = cached_data['parameters']

                print(f"✅ Cache loaded successfully!")
                print(f"   - Cached parameters: {cache_params}")
                print(f"   - Found {len(merge_fractions)} unique prefix pairs")

                return merge_data, merge_fractions

            except Exception as e:
                print(f"⚠️  Cache loading failed: {e}")
                print("   Proceeding with fresh analysis...")

        print(f"🔄 No cache found. Running fresh analysis...")
        print(f"   Cache will be saved to: {cache_file}")

        agreeing_prefix_pairs = self._gen_agreeing_prefix(
            max_continuation_length)

        merge_data, merge_fractions = self._run_fresh_analysis_chunked_xl(
            agreeing_prefix_pairs, max_continuation_length)

        if merge_data is not None:
            try:
                cache_data = {
                    'merge_data': merge_data,
                    'merge_fractions': merge_fractions,
                    'parameters': {
                        'L_train': self.L_train,
                        'epsilon': self.epsilon,
                        'max_continuation_length': max_continuation_length,
                        'model_params': sum(p.numel() for p in self.model.parameters())
                    },
                    'timestamp': str(datetime.now())
                }

                with open(cache_file, 'wb') as f:
                    pickle.dump(cache_data, f)

                print(f"💾 Analysis results cached to: {cache_file}")

            except Exception as e:
                print(f"⚠️  Failed to save cache: {e}")

        return merge_data, merge_fractions

    def _gen_agreeing_prefix(self, max_continuation_length):
        print(
            f"\nStep 1: Generating all prefixes up to length {max_continuation_length}...")
        all_prefixes = self.generate_all_prefixes(max_continuation_length)
        print(f"Generated {len(all_prefixes)} prefixes")

        print("\nStep 2: Finding pairs that agree on continuations...")
        agreeing_pairs = self.find_agreeing_prefix_pairs(
            all_prefixes, max_continuation_length
        )

        if len(agreeing_pairs) == 0:
            print("WARNING: No agreeing pairs found! This might indicate an issue.")
            return None, None, None

        agreeing_pairs = self.convert_agreeing_pairs_format(agreeing_pairs)

        return agreeing_pairs

    def _pairs_generator(self, agreeing_pairs):
        """Generator that yields (key, pair) tuples without storing them all in memory"""
        for key in agreeing_pairs.keys():
            for pair in agreeing_pairs[key]:
                yield key, pair

    # Called xl because I spent hours trying to get it to work with larger input data without freezing computer
    def _run_fresh_analysis_chunked_xl(self, agreeing_pairs, max_continuation_length, chunk_size=100, save_every=5):
        """
        Process in chunks using generator and save intermediate results every few chunks.
        """

        if max_continuation_length is not None:
            analysis_base_dir = "../models/analysis"
            if os.path.exists(analysis_base_dir):
                pattern = f"{analysis_base_dir}/results_{max_continuation_length}_*/final_results.pkl"
                existing_files = glob.glob(pattern)

                if existing_files:
                    most_recent = max(existing_files, key=os.path.getmtime)
                    print(
                        f"\n🔍 Found existing results for max_continuation_length={max_continuation_length}")
                    print(f"📁 Loading cached results from: {most_recent}")

                    try:
                        with open(most_recent, 'rb') as f:
                            cached_merge_data = pickle.load(f)

                        print("Computing merge fractions from cached data...")
                        merge_fractions = {}
                        for (m1, m2), data_list in cached_merge_data.items():
                            total_pairs = len(data_list)
                            merged_pairs = sum(
                                1 for d in data_list if d['merged'])
                            merge_fraction = merged_pairs / total_pairs if total_pairs > 0 else 0.0

                            merge_fractions[(m1, m2)] = {
                                'fraction': merge_fraction,
                                'total_pairs': total_pairs,
                                'merged_pairs': merged_pairs
                            }

                        print(
                            f"✅ Loaded cached results with {len(cached_merge_data)} groups and {sum(len(pairs) for pairs in cached_merge_data.values())} total pairs")
                        return cached_merge_data, merge_fractions

                    except Exception as e:
                        print(f"⚠️  Error loading cached results: {e}")
                        print("Proceeding with fresh analysis...")

        results_dir = f"../models/analysis/results_{max_continuation_length}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(results_dir, exist_ok=True)

        n_p = len(agreeing_pairs)
        print(f"\nProcessing {n_p} pairs in chunks of {chunk_size}")
        print(f"Results will be saved to: {results_dir}")

        merge_data = defaultdict(list)
        chunk_count = 0
        current_chunk = []
        processed_pairs = 0

        # Process using generator - no memory overhead for flattening
        for original_key, pair in self._pairs_generator(agreeing_pairs):
            current_chunk.append((original_key, pair))
            processed_pairs += 1

            if len(current_chunk) >= chunk_size:
                chunk_count += 1
                print(
                    f"Processing chunk {chunk_count} (pairs {processed_pairs-len(current_chunk)+1}-{processed_pairs} of {n_p})")

                for chunk_key, chunk_pair in current_chunk:
                    prefix1 = chunk_pair['prefix1']
                    prefix2 = chunk_pair['prefix2']

                    state1 = self.get_hidden_state(prefix1)
                    state2 = self.get_hidden_state(prefix2)

                    merged = self.states_merged(state1, state2)

                    m1, m2 = len(prefix1), len(prefix2)
                    merge_data[(m1, m2)].append({
                        'original_key': chunk_key,
                        'prefix1': prefix1,
                        'prefix2': prefix2,
                        'merged': merged,
                        'distance': float(np.linalg.norm(state1 - state2))
                    })

                    del state1, state2

                current_chunk = []

                if chunk_count % save_every == 0:
                    save_file = os.path.join(
                        results_dir, f'intermediate_{chunk_count}.pkl')
                    with open(save_file, 'wb') as f:
                        pickle.dump(dict(merge_data), f)
                    print(f"  Saved intermediate results to {save_file}")

                gc.collect()
                time.sleep(0.5)  # Small cooling delay between chunks

        # Process any remaining pairs in the final partial chunk
        if current_chunk:
            chunk_count += 1
            print(
                f"Processing final chunk {chunk_count} (pairs {processed_pairs-len(current_chunk)+1}-{processed_pairs} of {n_p})")

            for chunk_key, chunk_pair in current_chunk:
                prefix1 = chunk_pair['prefix1']
                prefix2 = chunk_pair['prefix2']

                state1 = self.get_hidden_state(prefix1)
                state2 = self.get_hidden_state(prefix2)

                merged = self.states_merged(state1, state2)

                m1, m2 = len(prefix1), len(prefix2)
                merge_data[(m1, m2)].append({
                    'original_key': chunk_key,
                    'prefix1': prefix1,
                    'prefix2': prefix2,
                    'merged': merged,
                    'distance': float(np.linalg.norm(state1 - state2))
                })

                del state1, state2

        final_save_file = os.path.join(results_dir, f'final_results.pkl')
        with open(final_save_file, 'wb') as f:
            pickle.dump(dict(merge_data), f)
        print(f"  Saved final results to {final_save_file}")

        print("\nComputing merge fractions...")
        merge_fractions = {}
        for (m1, m2), data_list in merge_data.items():
            total_pairs = len(data_list)
            merged_pairs = sum(1 for d in data_list if d['merged'])
            merge_fraction = merged_pairs / total_pairs if total_pairs > 0 else 0.0

            merge_fractions[(m1, m2)] = {
                'fraction': merge_fraction,
                'total_pairs': total_pairs,
                'merged_pairs': merged_pairs
            }

            print(
                f"  ({m1}, {m2}): {merged_pairs}/{total_pairs} = {merge_fraction:.3f}")

        print(f"\nAnalysis complete! Results saved in: {results_dir}")
        print(f"Processed {processed_pairs} pairs in {chunk_count} chunks")

        return merge_data, merge_fractions
