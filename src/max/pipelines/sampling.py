# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

"""Token sampling algorithms."""

from max.dtype import DType
from max.graph import Dim, Graph, Shape, TensorType, TensorValue, ops
from max.pipelines.max_config import SamplingConfig


def _sampling_input_types(
    sampling_config: SamplingConfig, return_logits: bool
) -> dict[str, TensorType]:
    inputs = {}

    # Logits are always provided
    if sampling_config.enable_variable_logits:
        logits_in_type = TensorType(
            sampling_config.in_dtype, ["total_output_len", "vocab_size"]
        )
        inputs["logits"] = logits_in_type
    else:
        logits_in_type = TensorType(
            sampling_config.in_dtype, ["batch", "vocab_size"]
        )
        inputs["logits"] = logits_in_type

    # We are currently, always passing tokens through
    prev_tokens_type = TensorType(DType.int64, ["batch", "num_prev_steps"])
    inputs["prev_tokens"] = prev_tokens_type

    # If we need to return logits, introduce tensor to append to.
    if return_logits:
        logits_type = TensorType(DType.float32, ["batch", "num_prev_steps"])
        inputs["existing_logits"] = logits_type

    # If we have variable token logits enabled
    if sampling_config.enable_variable_logits:
        logit_offset_type = TensorType(DType.uint32, ["logit_offsets_len"])
        inputs["logit_offsets"] = logit_offset_type

    # If we have structured_outputs enabled
    if sampling_config.enable_structured_output:
        bitmask_type = TensorType(DType.bool, ["batch", "vocab_size"])
        inputs["bitmask"] = bitmask_type

    return inputs


def token_sampler(
    sampling_config: SamplingConfig, return_logits: bool = False
) -> Graph:
    _input_dict = _sampling_input_types(sampling_config, return_logits)
    with Graph("top_k_sampler", input_types=_input_dict.values()) as graph:
        # Deconstruct inputs
        # TODO: Explore better ways of indexing into these input values
        # tightly coupling the input order with element indices feels
        # quite brittle.
        logits = graph.inputs[list(_input_dict).index("logits")].tensor
        logits = ops.cast(logits, sampling_config.out_dtype)
        prev_tokens = graph.inputs[
            list(_input_dict).index("prev_tokens")
        ].tensor

        if "existing_logits" in _input_dict:
            existing_logits = graph.inputs[
                list(_input_dict).index("existing_logits")
            ].tensor

        if "logit_offsets" in _input_dict:
            logit_offsets = graph.inputs[
                list(_input_dict).index("logit_offsets")
            ].tensor
            logits = ops.gather(logits, logit_offsets[1:] - 1, axis=0)
            logits = ops.rebind(logits, shape=("batch", "vocab_size"))

        if "bitmask" in _input_dict:
            bitmask = graph.inputs[list(_input_dict).index("bitmask")].tensor
            logits = ops.select(
                bitmask, logits, ops.constant(-10000, dtype=DType.float32)
            )

        # Apply top_k sampling
        shape = Shape(logits.shape)
        shape[-1] = Dim(1)
        tokens = ops.custom(
            "topk_fused_sampling",
            [
                ops.constant(sampling_config.top_k, dtype=DType.int64),
                logits,
            ],
            [TensorType(DType.int64, shape)],
        )[0]
        assert isinstance(tokens, TensorValue)

        # Concat tokens to previous tokens.
        all_tokens = ops.concat([prev_tokens, tokens], -1)

        # Gather logits if needed to return.
        if "existing_logits" in _input_dict:
            token_range = ops.reshape(
                ops.range(
                    ops.constant(0, dtype=DType.int64),
                    tokens.shape[0],
                    ops.constant(1, dtype=DType.int64),
                    out_dim=Dim(tokens.shape[0]),
                ),
                shape=tokens.shape,
            )

            token_indices = ops.concat(
                [
                    token_range,
                    tokens,
                ],
                axis=1,
            )
            new_logits = ops.reshape(
                ops.gather_nd(logits, token_indices), shape=tokens.shape
            )

            all_logits = ops.concat([existing_logits, new_logits], -1)
            graph.output(tokens, all_tokens, all_logits)
        else:
            tokens = ops.squeeze(tokens, -1)
            graph.output(tokens, all_tokens)

        return graph
