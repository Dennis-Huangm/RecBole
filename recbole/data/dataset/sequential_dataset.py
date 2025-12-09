# @Time   : 2020/9/16
# @Author : Yushuo Chen
# @Email  : chenyushuo@ruc.edu.cn

# UPDATE:
# @Time   : 2022/7/8, 2020/9/16, 2021/7/1, 2021/7/11
# @Author : Zhen Tian, Yushuo Chen, Xingyu Pan, Yupeng Hou
# @Email  : chenyuwuxinn@gmail.com, chenyushuo@ruc.edu.cn, xy_pan@foxmail.com, houyupeng@ruc.edu.cn

"""
recbole.data.sequential_dataset
###############################
"""

import numpy as np
import torch
import pickle
import os
import hashlib
from tqdm import tqdm
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    jit = lambda func: func  # Dummy decorator
    prange = range

from recbole.data.dataset import Dataset
from recbole.data.interaction import Interaction
from recbole.utils.enum_type import FeatureType, FeatureSource


if NUMBA_AVAILABLE:
    @jit(nopython=True, parallel=True)
    def _fast_copy_sequences_1d(source, target, starts, ends, max_len):
        """Fast numba-optimized function to copy variable-length sequences.
        
        Args:
            source: source array (1D)
            target: target array (2D) 
            starts: start indices for each sequence
            ends: end indices for each sequence
            max_len: maximum sequence length
        """
        n = len(starts)
        for i in prange(n):
            start = starts[i]
            end = ends[i]
            length = min(end - start, max_len)
            for j in range(length):
                target[i, j] = source[start + j]
else:
    def _fast_copy_sequences_1d(source, target, starts, ends, max_len):
        """Fallback function when numba is not available."""
        n = len(starts)
        for i in range(n):
            start = starts[i]
            end = ends[i]
            length = min(end - start, max_len)
            target[i, :length] = source[start:end][:length]


class SequentialDataset(Dataset):
    """:class:`SequentialDataset` is based on :class:`~recbole.data.dataset.dataset.Dataset`,
    and provides augmentation interface to adapt to Sequential Recommendation,
    which can accelerate the data loader.

    Attributes:
        max_item_list_len (int): Max length of historical item list.
        item_list_length_field (str): Field name for item lists' length.
    """

    def __init__(self, config):
        self.max_item_list_len = config["MAX_ITEM_LIST_LENGTH"]
        self.item_list_length_field = config["ITEM_LIST_LENGTH_FIELD"]
        super().__init__(config)
        
        # Initialize cache directory
        self.cache_dir = os.path.join(config["checkpoint_dir"], "augmentation_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        if config["benchmark_filename"] is not None:
            self._benchmark_presets()

    def _change_feat_format(self):
        """Change feat format from :class:`pandas.DataFrame` to :class:`Interaction`,
        then perform data augmentation.
        """
        super()._change_feat_format()

        if self.config["benchmark_filename"] is not None:
            return
        
        # Save original data fingerprint BEFORE augmentation for cache key generation
        self._save_original_data_fingerprint()
        
        # Try to load cached augmented data
        cache_loaded = self._load_augmented_cache()
        if cache_loaded:
            self.logger.info("Successfully loaded augmented data from cache.")
            return
        
        self.logger.debug("Augmentation for sequential recommendation.")
        self.data_augmentation()
        
        # Save augmented data to cache
        self._save_augmented_cache()

    def _aug_presets(self):
        list_suffix = self.config["LIST_SUFFIX"]
        for field in self.inter_feat:
            if field != self.uid_field:
                list_field = field + list_suffix
                setattr(self, f"{field}_list_field", list_field)
                ftype = self.field2type[field]

                if ftype in [FeatureType.TOKEN, FeatureType.TOKEN_SEQ]:
                    list_ftype = FeatureType.TOKEN_SEQ
                else:
                    list_ftype = FeatureType.FLOAT_SEQ

                if ftype in [FeatureType.TOKEN_SEQ, FeatureType.FLOAT_SEQ]:
                    list_len = (self.max_item_list_len, self.field2seqlen[field])
                else:
                    list_len = self.max_item_list_len

                self.set_field_property(
                    list_field, list_ftype, FeatureSource.INTERACTION, list_len
                )

        self.set_field_property(
            self.item_list_length_field, FeatureType.TOKEN, FeatureSource.INTERACTION, 1
        )

    def data_augmentation(self):
        """Augmentation processing for sequential dataset.

        Modified to use non-overlapping fixed-length sequences.
        E.g., ``u1`` has purchase sequence ``<i1, i2, ..., i50>`` and MAX_ITEM_LIST_LENGTH=20,
        then we generate non-overlapping samples:

        ``u1, <i1, i2, ..., i20> | i21``
        ``u1, <i22, i23, ..., i41> | i42``
        
        Each sequence has exactly MAX_ITEM_LIST_LENGTH items, and sequences don't overlap.
        Items that don't form a complete sequence are discarded.
        """
        self.logger.debug("data_augmentation")

        self._aug_presets()

        self._check_field("uid_field", "time_field")
        max_item_list_len = self.config["MAX_ITEM_LIST_LENGTH"]
        self.sort(by=[self.uid_field, self.time_field], ascending=True)
        
        # Optimized: vectorized computation of sequence indices
        uid_array = self.inter_feat[self.uid_field].numpy()
        
        # Find where user changes (user boundaries)
        uid_changes = np.concatenate([[True], uid_array[1:] != uid_array[:-1]])
        user_starts = np.where(uid_changes)[0]
        user_ends = np.concatenate([user_starts[1:], [len(uid_array)]])
        
        if NUMBA_AVAILABLE:
            self.logger.info("Using Numba JIT optimization for faster data augmentation")
        else:
            self.logger.info("Numba not available, using standard optimization (consider installing numba for better performance)")
        
        self.logger.info(f"Processing {len(user_starts)} users for NON-OVERLAPPING fixed-length augmentation")
        
        uid_list, seq_start_indices, seq_end_indices, target_index, item_list_length = [], [], [], [], []
        
        # Process each user's sequence with progress bar
        for start, end in tqdm(zip(user_starts, user_ends), 
                               total=len(user_starts), 
                               desc="Processing users",
                               disable=len(user_starts) < 100):  # Disable for small datasets
            user_len = end - start
            # Need at least max_item_list_len + 1 items (sequence + target)
            if user_len <= max_item_list_len:
                continue
                
            uid = uid_array[start]
            # Generate non-overlapping fixed-length sequences
            # Each sample uses max_item_list_len items as history + 1 item as target
            for i in range(max_item_list_len, user_len, max_item_list_len + 1):
                seq_start = i - max_item_list_len
                seq_end = i
                seq_len = seq_end - seq_start
                
                # Ensure sequence length is exactly max_item_list_len
                if seq_len != max_item_list_len:
                    continue
                
                # Ensure target index is within bounds
                if i >= user_len:
                    continue
                
                uid_list.append(uid)
                seq_start_indices.append(start + seq_start)
                seq_end_indices.append(start + seq_end)
                target_index.append(start + i)
                item_list_length.append(max_item_list_len)  # Always fixed length
        
        uid_list = np.array(uid_list)
        seq_start_indices = np.array(seq_start_indices, dtype=np.int64)
        seq_end_indices = np.array(seq_end_indices, dtype=np.int64)
        target_index = np.array(target_index, dtype=np.int64)
        item_list_length = np.array(item_list_length, dtype=np.int64)

        new_length = len(target_index)
        self.logger.info(f"Generated {new_length} augmented samples")
        
        new_data = self.inter_feat[target_index]
        new_dict = {
            self.item_list_length_field: torch.tensor(item_list_length),
        }

        # Optimized: batch process fields with progress bar
        fields_to_process = [f for f in self.inter_feat if f != self.uid_field]
        for field in tqdm(fields_to_process, desc="Processing fields", disable=len(fields_to_process) < 5):
            list_field = getattr(self, f"{field}_list_field")
            list_len = self.field2seqlen[list_field]
            shape = (
                (new_length, list_len)
                if isinstance(list_len, int)
                else (new_length,) + list_len
            )
            if (
                self.field2type[field] in [FeatureType.FLOAT, FeatureType.FLOAT_SEQ]
                and field in self.config["numerical_features"]
            ):
                shape += (2,)
            new_dict[list_field] = torch.zeros(
                shape, dtype=self.inter_feat[field].dtype
            )

            value = self.inter_feat[field]
            
            # Optimized: use fast copy function for 1D sequences
            if isinstance(list_len, int) and len(value.shape) == 1 and NUMBA_AVAILABLE:
                # Use numba-optimized function for 1D data
                value_np = value.numpy()
                target_np = new_dict[list_field].numpy()
                _fast_copy_sequences_1d(value_np, target_np, seq_start_indices, seq_end_indices, list_len)
                new_dict[list_field] = torch.from_numpy(target_np)
            elif isinstance(list_len, int) and len(value.shape) == 1:
                # Fallback for when numba is not available - use vectorized slicing where possible
                for i in range(new_length):
                    start_idx = seq_start_indices[i]
                    end_idx = seq_end_indices[i]
                    length = item_list_length[i]
                    new_dict[list_field][i][:length] = value[start_idx:end_idx]
            else:
                # For multi-dimensional or complex cases, use standard approach
                for i in range(new_length):
                    start_idx = seq_start_indices[i]
                    end_idx = seq_end_indices[i]
                    length = item_list_length[i]
                    new_dict[list_field][i][:length] = value[start_idx:end_idx]

        new_data.update(Interaction(new_dict))
        self.inter_feat = new_data

    def _save_original_data_fingerprint(self):
        """Save original data fingerprint before augmentation for consistent cache key."""
        # This is called BEFORE augmentation, so inter_feat contains original data
        self._original_data_fingerprint = {
            'dataset': self.config["dataset"],
            'max_item_list_len': self.config["MAX_ITEM_LIST_LENGTH"],
            'user_inter_num_interval': str(getattr(self.config, "user_inter_num_interval", "")),
            'item_inter_num_interval': str(getattr(self.config, "item_inter_num_interval", "")),
            'original_data_len': len(self.inter_feat),
            'uid_sample': str(self.inter_feat[self.uid_field][:100].numpy().tobytes() if len(self.inter_feat) > 100 else ""),
            'iid_sample': str(self.inter_feat[self.iid_field][:100].numpy().tobytes() if len(self.inter_feat) > 100 else ""),
        }
    
    def _get_cache_key(self):
        """Generate a unique cache key based on original dataset configuration."""
        # Use the saved fingerprint from BEFORE augmentation
        if not hasattr(self, '_original_data_fingerprint'):
            # Fallback: this shouldn't happen if _save_original_data_fingerprint was called
            self._save_original_data_fingerprint()
        
        fp = self._original_data_fingerprint
        key_components = [
            fp['dataset'],
            fp['max_item_list_len'],
            fp['user_inter_num_interval'],
            fp['item_inter_num_interval'],
            fp['original_data_len'],
            fp['uid_sample'],
            fp['iid_sample'],
        ]
        key_string = "|".join(str(k) for k in key_components)
        cache_key = hashlib.md5(key_string.encode()).hexdigest()
        return cache_key
    
    def _get_cache_path(self):
        """Get the cache file path."""
        cache_key = self._get_cache_key()
        cache_filename = f"augmented_{self.config['dataset']}_{cache_key}.pkl"
        return os.path.join(self.cache_dir, cache_filename)
    
    def _save_augmented_cache(self):
        """Save augmented inter_feat to cache."""
        try:
            cache_path = self._get_cache_path()
            # Ensure cache directory exists
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            self.logger.info(f"Saving augmented data to cache: {cache_path}")
            
            # Estimate cache size
            import sys
            total_size = 0
            for field_name, field_data in self.inter_feat.interaction.items():
                if hasattr(field_data, 'element_size') and hasattr(field_data, 'nelement'):
                    total_size += field_data.element_size() * field_data.nelement()
            estimated_gb = total_size / (1024**3)
            self.logger.info(f"Estimated cache size: ~{estimated_gb:.2f} GB")
            
            # Save inter_feat and related attributes needed for reconstruction
            self.logger.info("Preparing cache data...")
            cache_data = {
                'inter_feat': self.inter_feat,
                'field2seqlen': self.field2seqlen,
                'field2type': self.field2type,
            }
            
            # Save list field attributes
            for field in self.inter_feat:
                if field != self.uid_field:
                    list_field_attr = f"{field}_list_field"
                    if hasattr(self, list_field_attr):
                        cache_data[list_field_attr] = getattr(self, list_field_attr)
            
            self.logger.info("Serializing data to disk (this may take several minutes for large datasets)...")
            with open(cache_path, 'wb') as f:
                # Use tqdm to show that something is happening
                with tqdm(total=1, desc="Saving cache", bar_format='{desc}: {elapsed}') as pbar:
                    pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
                    pbar.update(1)
            
            actual_size_gb = os.path.getsize(cache_path) / (1024**3)
            self.logger.info(f"Successfully saved augmented data cache ({actual_size_gb:.2f} GB)")
        except Exception as e:
            self.logger.warning(f"Failed to save augmented cache: {e}")
    
    def _load_augmented_cache(self):
        """Load augmented inter_feat from cache if available."""
        try:
            cache_path = self._get_cache_path()
            
            if not os.path.exists(cache_path):
                self.logger.info("No augmented cache found, will perform data augmentation.")
                return False
            
            cache_size_gb = os.path.getsize(cache_path) / (1024**3)
            self.logger.info(f"Loading augmented data from cache: {cache_path}")
            self.logger.info(f"Cache size: {cache_size_gb:.2f} GB (loading may take 1-2 minutes)")
            
            with open(cache_path, 'rb') as f:
                with tqdm(total=1, desc="Loading cache", bar_format='{desc}: {elapsed}') as pbar:
                    cache_data = pickle.load(f)
                    pbar.update(1)
            
            # Validate cache data structure
            self.logger.debug(f"Cache contains keys: {list(cache_data.keys())}")
            if 'inter_feat' in cache_data:
                inter_feat = cache_data['inter_feat']
                if hasattr(inter_feat, '__len__'):
                    self.logger.info(f"Cached inter_feat contains {len(inter_feat)} samples")
            
            
            # Restore inter_feat
            self.logger.info("Restoring dataset from cache...")
            self.inter_feat = cache_data['inter_feat']
            
            # Restore list field attributes
            for key, value in cache_data.items():
                if key.endswith('_list_field'):
                    setattr(self, key, value)
            
            self.logger.info(f"Successfully loaded augmented data from cache ({cache_size_gb:.2f} GB)")
            return True
            
        except Exception as e:
            self.logger.warning(f"Failed to load augmented cache: {e}")
            return False
    
    def _benchmark_presets(self):
        list_suffix = self.config["LIST_SUFFIX"]
        for field in self.inter_feat:
            if field + list_suffix in self.inter_feat:
                list_field = field + list_suffix
                setattr(self, f"{field}_list_field", list_field)
        self.set_field_property(
            self.item_list_length_field, FeatureType.TOKEN, FeatureSource.INTERACTION, 1
        )
        self.inter_feat[self.item_list_length_field] = self.inter_feat[
            self.item_id_list_field
        ].agg(len)

    def inter_matrix(self, form="coo", value_field=None):
        """Get sparse matrix that describe interactions between user_id and item_id.
        Sparse matrix has shape (user_num, item_num).
        For a row of <src, tgt>, ``matrix[src, tgt] = 1`` if ``value_field`` is ``None``,
        else ``matrix[src, tgt] = self.inter_feat[src, tgt]``.

        Args:
            form (str, optional): Sparse matrix format. Defaults to ``coo``.
            value_field (str, optional): Data of sparse matrix, which should exist in ``df_feat``.
                Defaults to ``None``.

        Returns:
            scipy.sparse: Sparse matrix in form ``coo`` or ``csr``.
        """
        if not self.uid_field or not self.iid_field:
            raise ValueError(
                "dataset does not exist uid/iid, thus can not converted to sparse matrix."
            )

        l1_idx = self.inter_feat[self.item_list_length_field] == 1
        l1_inter_dict = self.inter_feat[l1_idx].interaction
        new_dict = {}
        list_suffix = self.config["LIST_SUFFIX"]
        candidate_field_set = set()
        for field in l1_inter_dict:
            if field != self.uid_field and field + list_suffix in l1_inter_dict:
                candidate_field_set.add(field)
                new_dict[field] = torch.cat(
                    [self.inter_feat[field], l1_inter_dict[field + list_suffix][:, 0]]
                )
            elif (not field.endswith(list_suffix)) and (
                field != self.item_list_length_field
            ):
                new_dict[field] = torch.cat(
                    [self.inter_feat[field], l1_inter_dict[field]]
                )
        local_inter_feat = Interaction(new_dict)
        return self._create_sparse_matrix(
            local_inter_feat, self.uid_field, self.iid_field, form, value_field
        )

    def build(self):
        """Processing dataset according to evaluation setting, including Group, Order and Split.
        See :class:`~recbole.config.eval_setting.EvalSetting` for details.

        Args:
            eval_setting (:class:`~recbole.config.eval_setting.EvalSetting`):
                Object contains evaluation settings, which guide the data processing procedure.

        Returns:
            list: List of built :class:`Dataset`.
        """
        ordering_args = self.config["eval_args"]["order"]
        if ordering_args != "TO":
            raise ValueError(
                f"The ordering args for sequential recommendation has to be 'TO'"
            )

        return super().build()
