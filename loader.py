# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import TYPE_CHECKING, Literal, Optional, Union

import numpy as np
from datasets import Dataset, load_dataset, load_from_disk

from ..extras import logging
from ..extras.constants import FILEEXT2TYPE
from ..extras.misc import check_version, has_tokenized_data
from .converter import align_dataset
from .data_utils import get_dataset_module, merge_dataset, read_cloud_json, split_dataset
from .parser import get_dataset_list
from .processor import (
    FeedbackDatasetProcessor,
    PackedSupervisedDatasetProcessor,
    PairwiseDatasetProcessor,
    PretrainDatasetProcessor,
    SupervisedDatasetProcessor,
    UnsupervisedDatasetProcessor,
)


if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset
    from transformers import PreTrainedTokenizer, ProcessorMixin, Seq2SeqTrainingArguments

    from ..hparams import DataArguments, ModelArguments
    from .data_utils import DatasetModule
    from .parser import DatasetAttr
    from .processor import DatasetProcessor
    from .template import Template


logger = logging.get_logger(__name__)


def _load_single_dataset(
    dataset_attr: "DatasetAttr",
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
) -> Union["Dataset", "IterableDataset"]:
    r"""Load a single dataset and aligns it to the standard format."""
    logger.info_rank0(f"Loading dataset {dataset_attr}...")
    data_path, data_name, data_dir, data_files = None, None, None, None
    if dataset_attr.load_from in ["hf_hub", "ms_hub", "om_hub"]:
        data_path = dataset_attr.dataset_name
        data_name = dataset_attr.subset
        data_dir = dataset_attr.folder

    elif dataset_attr.load_from == "script":
        data_path = os.path.join(data_args.dataset_dir, dataset_attr.dataset_name)
        data_name = dataset_attr.subset
        data_dir = dataset_attr.folder

    elif dataset_attr.load_from == "cloud_file":
        data_path = dataset_attr.dataset_name

    elif dataset_attr.load_from == "file":
        data_files = []
        local_path = os.path.join(data_args.dataset_dir, dataset_attr.dataset_name)
        if os.path.isdir(local_path):  # is directory
            for file_name in os.listdir(local_path):
                data_files.append(os.path.join(local_path, file_name))
        elif os.path.isfile(local_path):  # is file
            data_files.append(local_path)
        else:
            raise ValueError(f"File {local_path} not found.")

        data_path = FILEEXT2TYPE.get(os.path.splitext(data_files[0])[-1][1:], None)
        if data_path is None:
            raise ValueError("Allowed file types: {}.".format(",".join(FILEEXT2TYPE.keys())))

        if any(data_path != FILEEXT2TYPE.get(os.path.splitext(data_file)[-1][1:], None) for data_file in data_files):
            raise ValueError("File types should be identical.")
    else:
        raise NotImplementedError(f"Unknown load type: {dataset_attr.load_from}.")

    if dataset_attr.load_from == "ms_hub":
        check_version("modelscope>=1.11.0", mandatory=True)
        from modelscope import MsDataset  # type: ignore
        from modelscope.utils.config_ds import MS_DATASETS_CACHE  # type: ignore

        cache_dir = model_args.cache_dir or MS_DATASETS_CACHE
        dataset = MsDataset.load(
            dataset_name=data_path,
            subset_name=data_name,
            data_dir=data_dir,
            data_files=data_files,
            split=dataset_attr.split,
            cache_dir=cache_dir,
            token=model_args.ms_hub_token,
            use_streaming=data_args.streaming,
        )
        if isinstance(dataset, MsDataset):
            dataset = dataset.to_hf_dataset()

    elif dataset_attr.load_from == "om_hub":
        check_version("openmind>=0.8.0", mandatory=True)
        from openmind import OmDataset  # type: ignore
        from openmind.utils.hub import OM_DATASETS_CACHE  # type: ignore

        cache_dir = model_args.cache_dir or OM_DATASETS_CACHE
        dataset = OmDataset.load_dataset(
            path=data_path,
            name=data_name,
            data_dir=data_dir,
            data_files=data_files,
            split=dataset_attr.split,
            cache_dir=cache_dir,
            token=model_args.om_hub_token,
            streaming=data_args.streaming,
        )
    elif dataset_attr.load_from == "cloud_file":
        dataset = Dataset.from_list(read_cloud_json(data_path), split=dataset_attr.split)
    else:
        dataset = load_dataset(
            path=data_path,
            name=data_name,
            data_dir=data_dir,
            data_files=data_files,
            split=dataset_attr.split,
            cache_dir=model_args.cache_dir,
            token=model_args.hf_hub_token,
            num_proc=data_args.preprocessing_num_workers,
            trust_remote_code=model_args.trust_remote_code,
            streaming=data_args.streaming and dataset_attr.load_from != "file",
        )
        if data_args.streaming and dataset_attr.load_from == "file":
            dataset = dataset.to_iterable_dataset(num_shards=training_args.dataloader_num_workers)

    if dataset_attr.num_samples is not None and not data_args.streaming:
        target_num = dataset_attr.num_samples
        indexes = np.random.permutation(len(dataset))[:target_num]  # all samples should be included
        target_num -= len(indexes)
        if target_num > 0:
            expand_indexes = np.random.choice(len(dataset), target_num)
            indexes = np.concatenate((indexes, expand_indexes), axis=0)

        assert len(indexes) == dataset_attr.num_samples, "Sample num mismatched."
        dataset = dataset.select(indexes)
        logger.info_rank0(f"Sampled {dataset_attr.num_samples} examples from dataset {dataset_attr}.")

    if data_args.max_samples is not None:  # truncate dataset
        max_samples = min(data_args.max_samples, len(dataset))
        dataset = dataset.select(range(max_samples))

    return align_dataset(dataset, dataset_attr, data_args, training_args)


def _get_merged_dataset(
    dataset_names: Optional[list[str]],
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    stage: Literal["pt", "sft", "rm", "ppo", "kto"],
    return_dict: bool = False,
) -> Optional[Union["Dataset", "IterableDataset", dict[str, "Dataset"]]]:
    r"""Return the merged datasets in the standard format."""
    if dataset_names is None:
        return None

    datasets = {}
    for dataset_name, dataset_attr in zip(dataset_names, get_dataset_list(dataset_names, data_args.dataset_dir)):
        if (stage == "rm" and dataset_attr.ranking is False) or (stage != "rm" and dataset_attr.ranking is True):
            raise ValueError("The dataset is not applicable in the current training stage.")

        datasets[dataset_name] = _load_single_dataset(dataset_attr, model_args, data_args, training_args)

    if return_dict:
        return datasets
    else:
        return merge_dataset(list(datasets.values()), data_args, seed=training_args.seed)


def _get_dataset_processor(
    data_args: "DataArguments",
    stage: Literal["pt", "sft", "rm", "ppo", "kto"],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"],
    do_generate: bool = False,
) -> "DatasetProcessor":
    r"""Return the corresponding dataset processor."""
    if stage == "pt":
        dataset_processor_class = PretrainDatasetProcessor
    elif stage == "sft" and not do_generate:
        if data_args.packing:
            if data_args.neat_packing:  # hack datasets to have int32 attention mask
                from datasets.arrow_writer import OptimizedTypedSequence, TypedSequence

                def __init__(self, data, **kwargs):
                    return TypedSequence.__init__(
                        self,
                        data,
                        type=kwargs.pop("type", None),
                        try_type=kwargs.pop("try_type", None),
                        optimized_int_type=kwargs.pop("optimized_int_type", None),
                    )

                OptimizedTypedSequence.__init__ = __init__
            dataset_processor_class = PackedSupervisedDatasetProcessor
        else:
            dataset_processor_class = SupervisedDatasetProcessor

    elif stage == "rm":
        dataset_processor_class = PairwiseDatasetProcessor
    elif stage == "kto":
        dataset_processor_class = FeedbackDatasetProcessor
    else:
        dataset_processor_class = UnsupervisedDatasetProcessor

    return dataset_processor_class(template=template, tokenizer=tokenizer, processor=processor, data_args=data_args)


def _get_preprocessed_dataset(
    dataset: Optional[Union["Dataset", "IterableDataset"]],
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    stage: Literal["pt", "sft", "rm", "ppo", "kto"],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"] = None,
    is_eval: bool = False,
) -> Optional[Union["Dataset", "IterableDataset"]]:
    r"""Preprocesses the dataset, including format checking and tokenization."""
    if dataset is None:
        return None

    dataset_processor = _get_dataset_processor(
        data_args, stage, template, tokenizer, processor, do_generate=(training_args.predict_with_generate and is_eval)
    )
    def replace_before_subsequence(lst, subsequence, replace_value=-100):
        """
        将列表中指定子序列前的所有元素替换为指定值。

        :param lst: 原始列表
        :param subsequence: 需要查找的子序列
        :param replace_value: 替换的值，默认为 -100
        :return: 修改后的列表
        """
        # 获取子序列的长度
        # import pdb; pdb.set_trace()
        sub_len = len(subsequence)
        # 遍历列表，查找子序列
        for i in range(len(lst) - sub_len + 1):
            # 检查当前窗口是否匹配子序列
            if lst[i:i + sub_len] == subsequence:
                # 将子序列前的所有元素替换为 replace_value
                # lst[:i + sub_len] = [replace_value] * (i + sub_len)
                # 将子序列的所有元素替换为 replace_value
                lst[i:i + sub_len] = [replace_value] * sub_len
                # lst[:i] = [replace_value] * i
                # break
                
                return lst
        # 如果未找到子序列，返回原列表
        return lst
    def replace_after_subsequence(lst, replace_value=-100):
        """
        将列表中指定子序列前的所有元素替换为指定值。

        :param lst: 原始列表
        :param subsequence: 需要查找的子序列
        :param replace_value: 替换的值，默认为 -100
        :return: 修改后的列表
        """
        # 获取子序列的长度
        # import pdb; pdb.set_trace()
        # 遍历列表，查找子序列
        for i in range(len(lst)):
            # 检查当前窗口是否匹配子序列
            if lst[i] != -100:
                max_len = min(i + 200, len(lst))
                # lst[i: max_len] = replace_value
                for j in range(i, max_len):
                    lst[j] = replace_value
                break
        # 如果未找到子序列，返回原列表
        return lst
    # import pdb; pdb.set_trace()

    # import pdb; pdb.set_trace()
    if type(dataset) == dict:
        merge_dataset_dict = {}
        for dataset_name, sub_dataset in dataset.items():
            column_names = list(next(iter(sub_dataset)).keys())
            kwargs = {}
            if not data_args.streaming:
                kwargs = dict(
                    num_proc=data_args.preprocessing_num_workers,
                    load_from_cache_file=(not data_args.overwrite_cache) or (training_args.local_process_index != 0),
                    desc="Running tokenizer on dataset",
                )
            # import pdb; pdb.set_trace()
            sub_dataset = sub_dataset.map(
                dataset_processor.preprocess_dataset,
                batched=True,
                batch_size=data_args.preprocessing_batch_size,
                remove_columns=column_names,
                **kwargs,
            )
            
            # ############################################################################################  
            # mask_dataset = None
            # import torch
            if dataset_name in [
                                "wildjailbreak_train_R1_Qwen_DA_sft_harmful_prefix_1_benign_DA", 
                                "wildjailbreak_train_R1_Qwen_DA_sft_benign_prefix_1_harmful_DA", 
                                "wildjailbreak_train_R1_Qwen_DA_sft_prefix_max_t2",
                                "wildjailbreak_train_R1_Qwen_DA_sft_prefix_1",
                                "wildjailbreak_train_R1_Qwen_DA_sft_prefix_max_t3",
                                "wildjailbreak_train_R1_Qwen_ERPO_sft_prefix_1",
                                "wildjailbreak_train_R1_Qwen_DA_sft_prefix_max_t4",
                                "wildjailbreak_train_R1_Qwen_DA_sft_llama3_prefix_1",
                                "wildjailbreak_train_R1_Qwen_DA_only_benign_sft_v2_prefix_1",
                                "wildjailbreak_train_R1_Qwen_DA_only_benign_sft_prefix_1",
                                "wildjailbreak_train_R1_Qwen_DA_only_harmful_sft_prefix_1",
                                "wildjailbreak_train_R1_Qwen_DA_only_harmful_sft_v2_prefix_1",
                                "wildjailbreak_train_Qwen3-14B_DA_sft_Qwen3-14B_prefix_1",
                                "wildjailbreak_train_Qwen3-8B_DA_sft_Qwen3-8B_prefix_1",
                                "wildjailbreak_train_Qwen3-14B_DA_sft_prefix_max",
                                "wildjailbreak_train_R1_Qwen_DA_sft_prefix_rnd",
                                "wildjailbreak_train_Qwen3-14B_DA_sft_prefix_rnd",
                                "wildjailbreak_train_R1_Qwen_DA_sft_prefix_gcg",
                                "wildjailbreak_train_R1_Qwen_SA_sft_rnd_mask_iter_1",
                                "wildjailbreak_train_R1_Qwen_SA_sft_rnd_mask_iter_2",
                                "wildjailbreak_train_R1_Qwen_SA_sft_rnd_mask_iter_3",
                                "wildjailbreak_train_R1_Qwen_SA_sft_rnd_mask_iter_4",
                                "wildjailbreak_train_R1_Qwen_SA_sft_rnd_mask_iter_4_STaR",
                                "wildjailbreak_train_R1_Qwen_SA_sft_rnd_mask",
                                "wildjailbreak_train_R1_Qwen_SA_sft_rnd_mask_STaR",
                                "wildjailbreak_train_R1_Qwen_SA_sft_rnd_mask_STaR_iter_1",
                                "wildjailbreak_train_R1_Qwen_self_align_sft_rnd_mask_with_hint",
                                "wildjailbreak_train_R1_Qwen_self_align_sft_rnd_mask",
                                "wildjailbreak_train_R1_Qwen_self_align_sft_rnd_mask_with_hint_iter_1",
                                "wildjailbreak_train_R1_Qwen_self_align_sft_rnd_mask_with_hint_v2",
                                "wildjailbreak_train_R1_Qwen_self_align_sft_rnd_mask_with_hint_iter_2",
                                "wildjailbreak_train_R1_Qwen_self_align_sft_rnd_mask_with_hint_v2_iter_1",
                                "wildjailbreak_train_R1_Qwen_self_align_sft_rnd_mask_with_hint_v2_iter_2",
                                "wildjailbreak_train_Qwen3-14B_self_align_sft_rnd_mask_with_hint",
                                "wildjailbreak_train_Qwen3-14B_self_align_sft_rnd_mask_with_hint_iter_1",
                                "wildjailbreak_train_Qwen3-14B_self_align_sft_rnd_mask_with_hint_iter_2",
                                "wildjailbreak_train_R1_Qwen_self_align_v2_sft_rnd_mask_with_hint_v3_3",
                                "wildjailbreak_train_R1_Qwen_self_align_v2_sft_rnd_mask_with_hint_v5",
                                "wildjailbreak_train_Qwen3-14B_self_align_v2_sft_rnd_mask_with_hint5-6_helpful_v4_wo_sharegpt",
                                "wildjailbreak_train_Qwen3-14B_self_align_v2_sft_rnd_mask_with_hint5-6_helpful_v4_wo_evol_instruct",
                                "wildjailbreak_train_R1_Qwen_self_align_v3_rnd_mask_hint5_sft",
                                "wildjailbreak_train_Qwen3-14B_self_align_v3_rnd_mask_hint5_sft",
                                "wildjailbreak_train_R1_Qwen_self_align_v2_random_rnd_mask_hint5_sft",
                                "wildjailbreak_train_R1_Qwen_self_align_v2_random_rnd_mask_hint5_sft_iter_1",
                                "wildjailbreak_train_R1_Qwen_self_align_v2_random_t0-6_rnd_mask_hint5_sft",
                                "wildjailbreak_train_R1_Qwen_self_align_v2_random_t0-6_rnd_mask_hint5_sft_iter_1",
                                "wildjailbreak_train_R1_Qwen_self_align_v2_random_rnd_mask_hint5_sft_iter_2",
                                "wildjailbreak_train_R1_Qwen_self_align_v3_rnd_mask_hint5_sft_iter_1",
                                "wildjailbreak_train_R1_Qwen_self_align_v4_rnd_mask_hint5_sft",
                                "wildjailbreak_train_R1_Qwen_self_align_v4_rnd_mask_hint5_sft_iter_1",
                                "wildjailbreak_train_R1_Qwen_self_align_v4_rnd_mask_hint5_sft_iter_2",
                                "wildjailbreak_train_R1_Qwen_self_align_v4-3_rnd_mask_hint5_sft_iter_1",
                                "wildjailbreak_train_R1_Qwen_self_align_v4-3_rnd_mask_hint5_sft_iter_2",
                                "wildjailbreak_train_R1_Qwen_self_align_v5_rnd_mask_hint5_sft",
                                "wildjailbreak_train_R1_Qwen_self_align_v5_rnd_mask_hint5_sft_iter_1",
                                "wildjailbreak_train_Qwen3-14B_self_align_v3_sft_rnd_mask_with_hint5_helpful_replay",
                                "wildjailbreak_train_Qwen3-14B_self_align_v3_sft_rnd_mask_with_hint5_helpful_replay_it_1",
                                "wildjailbreak_train_Qwen3-14B_self_align_v6_sft_rnd_mask_hint5_uf_replay",
                                "wildjailbreak_train_Qwen3-14B_self_align_v6_sft_rnd_mask_hint7_uf_replay",
                                "wildjailbreak_train_Qwen3-14B_self_align_v6_sft_rnd_mask_hint7_uf_replay_100",
                                "wildjailbreak_train_Qwen3-14B_self_align_v6_sft_rnd_mask_hint7_uf_replay_50",
                                "wildjailbreak_train_Qwen3-14B_self_align_v6_sft_rnd_mask_hint7_uf_replay_50_mix",
                                "wildjailbreak_train_Qwen3-14B_self_align_v6_sft_rnd_mask_hint7_uf_replay_50_mix_it_1",
                                "wildjailbreak_train_Qwen3-14B_self_align_v3_rnd_mask_hint5_sft_iter_1",
                                "wildjailbreak_train_Qwen3-14B_self_align_v3_rnd_mask_hint5_sft_iter_1_replay_100",
                                "wildjailbreak_train_Qwen3-14B_self_align_v3_rnd_mask_hint5_sft_iter_1_replay_500",
                                "wildjailbreak_train_Qwen3-14B_self_align_v3_rnd_mask_hint5_sft_iter_1_replay_1k",
                                "wildjailbreak_train_Qwen3-14B_self_align_v3_rnd_mask_hint5_sft_iter_1_replay_5k",
                                "wildjailbreak_train_Qwen3-14B_self_align_v2_rnd_mask_hint5_uf_sft",
                                "wildjailbreak_train_Qwen3-14B_self_align_v2_rnd_mask_hint5_uf_sft_replay_iter_1",
                                "wildjailbreak_train_Qwen3-14B_self_align_v2_rnd_mask_hint5_uf_sft_replay_iter_2",
                                "wildjailbreak_train_Qwen3-14B_self_align_v8_rnd_mask_hint7_uf_sft",
                                "wildjailbreak_train_Qwen3-14B_self_align_v8_rnd_mask_hint7_uf_sft_iter_1",
                                "wildjailbreak_train_Qwen3-14B_self_align_v8_rnd_mask_hint7_ufv1_sft_iter_1",
                                "wildjailbreak_train_Qwen3-14B_self_align_v8_rnd_mask_hint7_ufv2_sft_iter_1",
                                "wildjailbreak_train_Qwen3-14B_self_align_v8_rnd_mask_hint7_ufv2_sft_iter_2",
                                "wildjailbreak_train_Qwen3-14B_self_align_v8_rnd_mask_hint7_ufv3_sft_iter_1",
                                "wildjailbreak_train_R1_Qwen_self_align_v8_rnd_mask_hint8_ufv3_sft",
                                "wildjailbreak_train_Qwen3-14B_self_align_v8_rnd_mask_hint8_ufv3_sft",
                                "wildjailbreak_train_R1_Qwen_self_align_v9_rnd_mask_hint8_ufv3_sft",
                                "wildjailbreak_train_Qwen3-14B_self_align_v9_rnd_mask_hint8_ufv3_sft",
                                "wildjailbreak_train_Qwen3-14B_self_align_v2_rnd_mask_hint8_ufv3_sft",
                                "wildjailbreak_train_Qwen3-14B_self_align_v2_rnd_mask_hint8_ufv3_sft_iter_1",
                                "wildjailbreak_train_Qwen3-14B_self_align_v2_rnd_mask_hint8_ufv4_sft",
                                "wildjailbreak_train_Qwen3-14B_self_align_v2_rnd_mask_hint9_ufv3_sft",
                                "wildjailbreak_train_Qwen3-14B_self_align_v2_rnd_mask_hint9_ufv3_sft_iter_1",
                                "wildjailbreak_train_Qwen3-14B_self_align_v10_rnd_mask_hint10_ufv4_sft",
                                "wildjailbreak_train_GLM-Z1-9B_self_align_v10_rnd_mask_hint10_ufv4_sft",
                                "wildjailbreak_train_Qwen3-14B_self_align_v10_rnd_mask_hint10_ufv4_sft_iter_1",
                                "wildjailbreak_train_GLM-Z1-9B_self_align_v10_rnd_mask_hint10_ufv4_sft_iter_1",
                                "wildjailbreak_train_Qwen3-14B_self_align_v2_rnd_mask_hint11_ufv2_sft",
                                "wildjailbreak_train_Qwen3-14B_self_align_v2_rnd_mask_hint11_ufv2_sft_iter_1",
                                "wildjailbreak_train_Qwen3-14B_self_align_v2_rnd_mask_hint11_ufv2_sft_iter_2",
                                "wildjailbreak_train_Qwen3-14B_self_align_v10_rnd_mask_hint10_ufv4_sft_iter_2",
                                "wildjailbreak_train_Qwen3-14B_self_align_v8_rnd_mask_hint11_ufv2_sft",
                                "wildjailbreak_train_Qwen3-14B_self_align_v8_rnd_mask_hint11_ufv5_sft",
                                "wildjailbreak_train_Qwen3-14B_self_align_v2_sft_rnd_mask_hint5_helpful_v3-1_UF",
                                "wildjailbreak_train_Qwen3-14B_self_align_v2_sft_rnd_mask_hint5_helpful_v3-2_UF",
                                "wildjailbreak_train_R1_Qwen_self_align_v2_sft_rnd_mask_hint5_helpful_v3-0_UF",
                                "wildjailbreak_train_R1_Qwen_self_align_v2_sft_rnd_mask_helpful_v3-0_UF",
                                "wildjailbreak_train_R1_Qwen_self_align_v2_sft_rnd_mask_hint5_helpful_v3-1_UF",
                                "wildjailbreak_train_R1_Qwen_self_align_v2_sft_rnd_mask_hint5_helpful_v3-2_UF",
                                "wildjailbreak_train_Qwen3-14B_self_align_v2_sft_rnd_mask_hint5_helpful_v3-3_UF",
                                "wildjailbreak_train_R1_Qwen_self_align_v2_sft_rnd_mask_hint5_helpful_v3-0_UFv1",
                                "wildjailbreak_train_R1_Qwen_self_align_v2_sft_rnd_mask_hint15_helpful_v3-0_UF",
                                "wildjailbreak_train_R1_Qwen_self_align_v2_sft_rnd_mask_hint15_helpful_v3-1_UF",
                                "wildjailbreak_train_R1_Qwen_self_align_v2_sft_rnd_mask_hint15_helpful_v3-2_UF",
                                "wildjailbreak_train_Qwen3-14B_self_align_v2_sft_rnd_mask_hint15_helpful_v3-0_UF",
                                "wildjailbreak_train_Qwen3-14B_self_align_v2_sft_rnd_mask_helpful_v3-0_UF",
                                "wildjailbreak_train_Qwen3-14B_self_align_v2_sft_rnd_mask_hint15_helpful_v3-1_UF",
                                "wildjailbreak_train_R1_Qwen_self_align_v2_sft_rnd_mask_helpful_v3-1_UF",
                                "wildjailbreak_train_Qwen3-14B_self_align_v2_sft_rnd_mask_hint16_helpful_v3-0_UF",
                                "wildjailbreak_train_R1_Qwen_self_align_v2_sft_rnd_mask_hint16_helpful_v3-0_UF",
                                "wildjailbreak_train_Qwen3-14B_self_align_v2_sft_rnd_mask_hint16_helpful_v3-1_UF",
                                "wildjailbreak_train_Qwen3-14B_self_align_v2_sft_rnd_mask_hint16_helpful_v3-2_UF",
                                "wildjailbreak_train_R1_Qwen_self_align_v2_sft_rnd_mask_hint16_helpful_v3-1_UF",
                                "wildjailbreak_train_R1_Qwen_self_align_v2_sft_rnd_mask_hint16_helpful_v3-2_UF",
                                "wildjailbreak_train_Qwen3-14B_self_align_v2_sft_rnd_mask_helpful_v3-1_UF",
                                "wildjailbreak_train_Qwen3-14B_self_align_v2_sft_rnd_mask_helpful_v3-2_UF",
                                "wildjailbreak_train_R1_Qwen_self_align_v2_sft_rnd_mask_helpful_v3-2_UF",
                                "wildjailbreak_train_R1_Qwen_self_align_v2_1_sft_rnd_mask_hint16_helpful_v3-0_UF",
                                "wildjailbreak_train_Qwen3-14B_self_align_v2_1_sft_rnd_mask_hint16_helpful_v3-0_UF",
                                "wildjailbreak_train_R1_Qwen_self_align_v2_1_sft_rnd_mask_hint16_helpful_v3-1_UF",
                                "wildjailbreak_train_Qwen3-14B_self_align_v2_1_sft_rnd_mask_hint16_helpful_v3-1_UF",
                                "wildjailbreak_train_R1_Qwen_self_align_v2_sft_0_mask_hint16_helpful_v3-0_UF",
                                "wildjailbreak_train_R1_Qwen_self_align_v2_sft_25_mask_hint16_helpful_v3-0_UF",
                                "wildjailbreak_train_R1_Qwen_self_align_v2_sft_50_mask_hint16_helpful_v3-0_UF",
                                "wildjailbreak_train_R1_Qwen_self_align_v2_sft_75_mask_hint16_helpful_v3-0_UF",
                                ]:
            # if dataset_name == "wildjailbreak_train_R1_Qwen_DA_sft_llama3_prefix_1":
                tmp_dataset = sub_dataset.to_dict()
                if dataset_name in ["wildjailbreak_train_R1_Qwen_DA_sft_harmful_prefix_1_benign_DA", "wildjailbreak_train_R1_Qwen_DA_sft_benign_prefix_1_harmful_DA"]:
                    prefix_info_file = "/home/dwu/LLaMA-Factory/data/DeepSeek-R1-Distill-Qwen-14B_sys_0_sys_0_deliberative_alignment_v2_prefix_1_wildguard_as_judge_labelled.json"
                elif dataset_name == "wildjailbreak_train_R1_Qwen_DA_sft_prefix_max_t2":
                    prefix_info_file = "/home/dwu/LLaMA-Factory/data/wildjailbreak_train_R1_Qwen_DA_sft_prefix_max_t2_info.json"
                elif dataset_name == "wildjailbreak_train_R1_Qwen_DA_sft_prefix_max_t3":
                    prefix_info_file = "/home/dwu/LLaMA-Factory/data/wildjailbreak_train_R1_Qwen_DA_sft_prefix_max_t3_info.json"
                elif dataset_name == "wildjailbreak_train_R1_Qwen_DA_sft_prefix_max_t4":
                    prefix_info_file = "/home/dwu/LLaMA-Factory/data/wildjailbreak_train_R1_Qwen_DA_sft_prefix_max_t4_info.json"
                elif dataset_name == "wildjailbreak_train_R1_Qwen_DA_sft_prefix_1":
                    prefix_info_file = "/home/dwu/LLaMA-Factory/data/DeepSeek-R1-Distill-Qwen-14B_sys_0_sys_0_deliberative_alignment_v2_prefix_1_wildguard_as_judge_labelled.json"
                elif dataset_name == "wildjailbreak_train_R1_Qwen_ERPO_sft_prefix_1":
                    prefix_info_file = "/home/dwu/LLaMA-Factory/data/wildjailbreak_train_R1_Qwen_ERPO_sft_prefix_1_info.json"
                elif dataset_name == "wildjailbreak_train_R1_Qwen_DA_sft_llama3_prefix_1":
                    prefix_info_file = "/home/dwu/LLaMA-Factory/data/DeepSeek-R1-Distill-Llama-8B_sys_0_sys_0_deliberative_alignment_v2_prefix_1_wildguard_as_judge_labelled.json"
                elif dataset_name == "wildjailbreak_train_R1_Qwen_DA_only_benign_sft_v2_prefix_1":
                    prefix_info_file = "/home/dwu/LLaMA-Factory/data/wildjailbreak_train_R1_Qwen_DA_only_benign_sft_v2_prefix_1_info.json"
                elif dataset_name == "wildjailbreak_train_R1_Qwen_DA_only_benign_sft_prefix_1":
                    prefix_info_file = "/home/dwu/LLaMA-Factory/data/wildjailbreak_train_R1_Qwen_DA_only_benign_sft_prefix_1_info.json"
                elif dataset_name in ["wildjailbreak_train_R1_Qwen_DA_only_harmful_sft_prefix_1", "wildjailbreak_train_R1_Qwen_DA_only_harmful_sft_v2_prefix_1"]:
                    prefix_info_file = "/home/dwu/LLaMA-Factory/data/wildjailbreak_train_R1_Qwen_DA_only_harmful_sft_v2_prefix_1_info.json"
                elif dataset_name == "wildjailbreak_train_Qwen3-14B_DA_sft_Qwen3-14B_prefix_1":
                    prefix_info_file = "/home/dwu/LLaMA-Factory/data/wildjailbreak_train_Qwen3-14B_DA_sft_Qwen3-14B_prefix_1_info.json"
                elif dataset_name == "wildjailbreak_train_Qwen3-8B_DA_sft_Qwen3-8B_prefix_1":
                    prefix_info_file = "/home/dwu/LLaMA-Factory/data/wildjailbreak_train_Qwen3-8B_DA_sft_Qwen3-8B_prefix_1_info.json"
                elif dataset_name == "wildjailbreak_train_Qwen3-14B_DA_sft_prefix_max":
                    prefix_info_file = "/home/dwu/LLaMA-Factory/data/wildjailbreak_train_Qwen3-14B_DA_sft_prefix_max_info.json"
                elif dataset_name == "wildjailbreak_train_R1_Qwen_DA_sft_prefix_rnd":
                    prefix_info_file = "/home/dwu/LLaMA-Factory/data/wildjailbreak_train_R1_Qwen_DA_sft_prefix_rnd_info.json"
                elif dataset_name == "wildjailbreak_train_Qwen3-14B_DA_sft_prefix_rnd":
                    prefix_info_file = "/home/dwu/LLaMA-Factory/data/wildjailbreak_train_Qwen3-14B_DA_sft_prefix_rnd_info.json"
                elif dataset_name == "wildjailbreak_train_R1_Qwen_DA_sft_prefix_gcg":
                    prefix_info_file = "/home/dwu/LLaMA-Factory/data/wildjailbreak_train_R1_Qwen_DA_sft_prefix_gcg_info.json"
                elif dataset_name == "wildjailbreak_train_R1_Qwen_SA_sft_rnd_mask_iter_1":
                    prefix_info_file = "/home/dwu/LLaMA-Factory/data/wildjailbreak_train_R1_Qwen_SA_sft_rnd_mask_iter_1_info.json"
                elif dataset_name == "wildjailbreak_train_R1_Qwen_SA_sft_rnd_mask_iter_2":
                    prefix_info_file = "/home/dwu/LLaMA-Factory/data/wildjailbreak_train_R1_Qwen_SA_sft_rnd_mask_iter_2_info.json"
                elif dataset_name == "wildjailbreak_train_R1_Qwen_SA_sft_rnd_mask_iter_3":
                    prefix_info_file = "/home/dwu/LLaMA-Factory/data/wildjailbreak_train_R1_Qwen_SA_sft_rnd_mask_iter_3_info.json"
                elif dataset_name == "wildjailbreak_train_R1_Qwen_SA_sft_rnd_mask_iter_4":
                    prefix_info_file = "/home/dwu/LLaMA-Factory/data/wildjailbreak_train_R1_Qwen_SA_sft_rnd_mask_iter_4_info.json"
                elif dataset_name == "wildjailbreak_train_R1_Qwen_SA_sft_rnd_mask_iter_4_STaR":
                    prefix_info_file = "/home/dwu/LLaMA-Factory/data/wildjailbreak_train_R1_Qwen_SA_sft_rnd_mask_iter_4_STaR_info.json"
                elif dataset_name == "wildjailbreak_train_R1_Qwen_SA_sft_rnd_mask":
                    prefix_info_file = "/home/dwu/LLaMA-Factory/data/wildjailbreak_train_R1_Qwen_SA_sft_rnd_mask_info.json"
                elif dataset_name == "wildjailbreak_train_R1_Qwen_SA_sft_rnd_mask_STaR":
                    prefix_info_file = "/home/dwu/LLaMA-Factory/data/wildjailbreak_train_R1_Qwen_SA_sft_rnd_mask_STaR_info.json"
                elif dataset_name == "wildjailbreak_train_R1_Qwen_SA_sft_rnd_mask_STaR_iter_1":
                    prefix_info_file = "/home/dwu/LLaMA-Factory/data/wildjailbreak_train_R1_Qwen_SA_sft_rnd_mask_STaR_iter_1_info.json"
                else:
                    prefix_info_file = f"/home/dwu/LLaMA-Factory/data/{dataset_name}_info.json"
                # prefix_info_file = "/home/dwu/LLaMA-Factory/data/DeepSeek-R1-Distill-Llama-8B_sys_0_sys_0_deliberative_alignment_v2_prefix_1_wildguard_as_judge_labelled.json"
                import json
                prefix_info_dataset = json.load(open(prefix_info_file, 'r'))
                prefix_info_dict = {}
                not_found_count=0
                for data in prefix_info_dataset:
                    if data.get("prefix", None) != None:
                        # if dataset_name == "wildjailbreak_train_R1_Qwen_DA_sft_prefix_1" and data['is_harmful_request'] == True:
                        #     prefix_info_dict[data['question']] = data['prefix']
                        # else:
                        #     prefix_info_dict[data['question']] = None
                        # data['prefix'] = data['prefix'].replace("<think>\nOkay", "")
                        # prefix_info_dict[data['question']] = data['prefix']
                        if prefix_info_dict.get(data['question'], None) == None:
                            prefix_info_dict[data['question']] = []
                        prefix_info_dict[data['question']].append(data['prefix'])
                    else:
                        # prefix_info_dict[data['question']] = None
                        pass
                # reset_subseq = tokenizer.encode("[RESET]")
                sucess_mask = 0
                import copy
                for idx, data in enumerate(tmp_dataset['labels']):
                    if idx in [0, 1, 2]:
                        print("before -100: {}".format(data.count(-100)))
                    input_tokens = tokenizer.decode(tmp_dataset['input_ids'][idx])
                    # query = input_tokens.split("<｜Assistant｜>")[0].split("<｜User｜>")[-1]
                    if "<｜Assistant｜>" in input_tokens:
                        query = input_tokens.split("<｜Assistant｜>")[0].split("<｜User｜>")[-1]
                    elif "<|im_end|>\n<|im_start|>assistant\n" in input_tokens:
                        query = input_tokens.split("<|im_end|>\n<|im_start|>assistant\n")[0].split("<|im_start|>user\n")[-1]
                    elif "<|im_end|><|im_start|>assistant<|im_sep|>" in input_tokens:
                        query = input_tokens.split("<|im_end|><|im_start|>assistant<|im_sep|>")[0].split("<|im_end|><|im_start|>user<|im_sep|>")[-1]
                    elif "[gMASK]<sop><|user|>" in input_tokens:
                        query = input_tokens.split("<|assistant|>")[0].split("[gMASK]<sop><|user|>\n")[-1]
                    if prefix_info_dict.get(query, None) == None:
                        query = query + " "
                    if prefix_info_dict.get(query, None) != None:
                        first_prefix = prefix_info_dict[query]
                    else:
                        first_prefix = None 
                        not_found_count += 1
                    # first_prefix = prefix_info_dict[query]
                    if first_prefix != None:
                        to_be_mask_prefix = []
                        for prefix in first_prefix:
                            reset_subseq = tokenizer.encode(prefix, add_special_tokens = False)
                            to_be_mask_prefix.append(reset_subseq)
                        to_be_mask_prefix = sorted(to_be_mask_prefix, key=lambda x: len(x), reverse=True)
                        for reset_subseq in to_be_mask_prefix:
                            # import pdb; pdb.set_trace()
                            tmp_data = copy.deepcopy(data)
                            tmp_mask_tokens = replace_before_subsequence(tmp_data, reset_subseq)
                            if data.count(-100) != tmp_mask_tokens.count(-100) or len(reset_subseq) == 0:
                                tmp_dataset['labels'][idx] = tmp_mask_tokens
                                sucess_mask += 1
                                break

                    if idx in range(3):
                        print("after -100: {}".format(tmp_dataset['labels'][idx].count(-100)))
                        print("-"*50)
                print("not_found_count: {}".format(not_found_count))
                print("sucess_mask: {}".format(sucess_mask))
                from datasets import Dataset
                sub_dataset = Dataset.from_dict(tmp_dataset)
                merge_dataset_dict[dataset_name] = sub_dataset
            else:
                merge_dataset_dict[dataset_name] = sub_dataset
            if dataset_name in ["CodeAlpaca-20k",]:
                tmp_dataset = sub_dataset.to_dict()
                import json
                import torch
                select_mask_tokens_info = torch.load("/home/dwu/Immunization/test/codealpaca_slm_tokens_p_0.6.pt")
                query_select_tokens_dict = {}
                for item in select_mask_tokens_info:
                    query = item['question']
                    query_select_tokens_dict[query] = item
                # import pdb; pdb.set_trace()
                # reset_subseq = tokenizer.encode("[RESET]")
                can_not_found_count = 0
                miss_match_labels = 0
                for idx, data in enumerate(tmp_dataset['labels']):
                    if idx in [0, 1, 2]:
                        print("before -100: {}".format(data.count(-100)))
                    input_tokens = tokenizer.decode(tmp_dataset['input_ids'][idx])
                    # query = input_tokens.split("<｜Assistant｜>")[0].split("<｜User｜>")[-1]
                    if " [/INST]" in input_tokens:
                        query = input_tokens.split(" [/INST]")[0].split("<</SYS>>\n\n")[-1]

                    if query_select_tokens_dict.get(query, None) != None:
                        if tmp_dataset['labels'][idx] == query_select_tokens_dict[query]['labels']:
                            tmp_dataset['labels'][idx] = query_select_tokens_dict[query]['replace_labels']
                        else:
                            miss_match_labels +=1
                    else:
                        can_not_found_count += 1
                    if idx in [0, 1, 2]:
                        print("after -100: {}".format(tmp_dataset['labels'][idx].count(-100)))
                        print("-"*50)
                from datasets import Dataset
                sub_dataset = Dataset.from_dict(tmp_dataset)
                merge_dataset_dict[dataset_name] = sub_dataset
                print("miss_match_labels: {}".format(miss_match_labels))
                print("can_not_found_count: {}".format(can_not_found_count))
            else:
                merge_dataset_dict[dataset_name] = sub_dataset
            # ############################################################################################

        merge_list = []
        for k, v in merge_dataset_dict.items():
            merge_list.append(v)

        from datasets import concatenate_datasets
        dataset = concatenate_datasets(merge_list)
    else:
        column_names = list(next(iter(dataset)).keys())
        kwargs = {}
        if not data_args.streaming:
            kwargs = dict(
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=(not data_args.overwrite_cache) or (training_args.local_process_index != 0),
                desc="Running tokenizer on dataset",
            )
        dataset = dataset.map(
            dataset_processor.preprocess_dataset,
            batched=True,
            batch_size=data_args.preprocessing_batch_size,
            remove_columns=column_names,
            **kwargs,
        )
    # import pdb; pdb.set_trace()

    if training_args.should_log:
        try:
            print("eval example:" if is_eval else "training example:")
            dataset_processor.print_data_example(next(iter(dataset)))
        except StopIteration:
            if stage == "pt":
                raise RuntimeError("Cannot find sufficient samples, consider increasing dataset size.")
            else:
                raise RuntimeError("Cannot find valid samples, check `data/README.md` for the data format.")

    return dataset


def get_dataset(
    template: "Template",
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    stage: Literal["pt", "sft", "rm", "ppo", "kto"],
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"] = None,
) -> "DatasetModule":
    r"""Get the train dataset and optionally gets the evaluation dataset."""
    # Load tokenized dataset if path exists
    if data_args.tokenized_path is not None:
        if has_tokenized_data(data_args.tokenized_path):
            logger.warning_rank0("Loading dataset from disk will ignore other data arguments.")
            tokenized_data = load_from_disk(data_args.tokenized_path)
            dataset_module = get_dataset_module(tokenized_data)
            if data_args.streaming:
                dataset_module["train_dataset"] = dataset_module["train_dataset"].to_iterable_dataset()

            logger.info_rank0(f"Loaded tokenized dataset from {data_args.tokenized_path}.")
            return dataset_module

        if data_args.streaming:
            raise ValueError("Turn off `streaming` when saving dataset to disk.")

    # Load and preprocess dataset
    with training_args.main_process_first(desc="load dataset"):
        dataset = _get_merged_dataset(data_args.dataset, model_args, data_args, training_args, stage, return_dict = True)
        eval_dataset = _get_merged_dataset(
            data_args.eval_dataset,
            model_args,
            data_args,
            training_args,
            stage,
            return_dict=data_args.eval_on_each_dataset,
        )

    with training_args.main_process_first(desc="pre-process dataset"):
        dataset = _get_preprocessed_dataset(
            dataset, data_args, training_args, stage, template, tokenizer, processor, is_eval=False
        )
        if isinstance(eval_dataset, dict):
            for eval_name, eval_data in eval_dataset.items():
                eval_dataset[eval_name] = _get_preprocessed_dataset(
                    eval_data, data_args, training_args, stage, template, tokenizer, processor, is_eval=True
                )
        else:
            eval_dataset = _get_preprocessed_dataset(
                eval_dataset, data_args, training_args, stage, template, tokenizer, processor, is_eval=True
            )

        dataset_dict = split_dataset(dataset, eval_dataset, data_args, seed=training_args.seed)
        if data_args.tokenized_path is not None:  # save tokenized dataset to disk
            if training_args.should_save:
                dataset_dict.save_to_disk(data_args.tokenized_path)
                logger.info_rank0(f"Tokenized dataset is saved at {data_args.tokenized_path}.")
                logger.info_rank0(f"Please launch the training with `tokenized_path: {data_args.tokenized_path}`.")

        return get_dataset_module(dataset_dict)
