# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import numpy as np
import torch
from torch.utils.data import default_collate

from deepaudio.tts.datasets.batch import batch_sequences


def tacotron2_single_spk_batch_fn(examples):
    # fields = ["text", "text_lengths", "speech", "speech_lengths"]
    text = [np.array(item["text"], dtype=np.int64) for item in examples]
    speech = [np.array(item["speech"], dtype=np.float32) for item in examples]
    text_lengths = [
        np.array(item["text_lengths"], dtype=np.int64) for item in examples
    ]
    speech_lengths = [
        np.array(item["speech_lengths"], dtype=np.int64) for item in examples
    ]

    text = batch_sequences(text)
    speech = batch_sequences(speech)

    # convert each batch to paddle.Tensor
    text = default_collate(text)
    speech = default_collate(speech)
    text_lengths = default_collate(text_lengths)
    speech_lengths = default_collate(speech_lengths)

    batch = {
        "text": text,
        "text_lengths": text_lengths,
        "speech": speech,
        "speech_lengths": speech_lengths,
    }
    return batch


def tacotron2_multi_spk_batch_fn(examples):
    # fields = ["text", "text_lengths", "speech", "speech_lengths"]
    text = [np.array(item["text"], dtype=np.int64) for item in examples]
    speech = [np.array(item["speech"], dtype=np.float32) for item in examples]
    text_lengths = [
        np.array(item["text_lengths"], dtype=np.int64) for item in examples
    ]
    speech_lengths = [
        np.array(item["speech_lengths"], dtype=np.int64) for item in examples
    ]

    text = batch_sequences(text)
    speech = batch_sequences(speech)

    # convert each batch to paddle.Tensor
    text = default_collate(text)
    speech = default_collate(speech)
    text_lengths = default_collate(text_lengths)
    speech_lengths = default_collate(speech_lengths)

    batch = {
        "text": text,
        "text_lengths": text_lengths,
        "speech": speech,
        "speech_lengths": speech_lengths,
    }
    # spk_emb has a higher priority than spk_id
    if "spk_emb" in examples[0]:
        spk_emb = [
            np.array(item["spk_emb"], dtype=np.float32) for item in examples
        ]
        spk_emb = batch_sequences(spk_emb)
        spk_emb = default_collate(spk_emb)
        batch["spk_emb"] = spk_emb
    elif "spk_id" in examples[0]:
        spk_id = [np.array(item["spk_id"], dtype=np.int64) for item in examples]
        spk_id = default_collate(spk_id)
        batch["spk_id"] = spk_id
    return batch


def speedyspeech_single_spk_batch_fn(examples):
    # fields = ["phones", "tones", "num_phones", "num_frames", "feats", "durations"]
    phones = [np.array(item["phones"], dtype=np.int64) for item in examples]
    tones = [np.array(item["tones"], dtype=np.int64) for item in examples]
    feats = [np.array(item["feats"], dtype=np.float32) for item in examples]
    durations = [
        np.array(item["durations"], dtype=np.int64) for item in examples
    ]
    num_phones = [
        np.array(item["num_phones"], dtype=np.int64) for item in examples
    ]
    num_frames = [
        np.array(item["num_frames"], dtype=np.int64) for item in examples
    ]

    phones = batch_sequences(phones)
    tones = batch_sequences(tones)
    feats = batch_sequences(feats)
    durations = batch_sequences(durations)

    # convert each batch to paddle.Tensor
    phones = default_collate(phones)
    tones = default_collate(tones)
    feats = default_collate(feats)
    durations = default_collate(durations)
    num_phones = default_collate(num_phones)
    num_frames = default_collate(num_frames)
    batch = {
        "phones": phones,
        "tones": tones,
        "num_phones": num_phones,
        "num_frames": num_frames,
        "feats": feats,
        "durations": durations,
    }
    return batch


def speedyspeech_multi_spk_batch_fn(examples):
    # fields = ["phones", "tones", "num_phones", "num_frames", "feats", "durations", "spk_id"]
    phones = [np.array(item["phones"], dtype=np.int64) for item in examples]
    tones = [np.array(item["tones"], dtype=np.int64) for item in examples]
    feats = [np.array(item["feats"], dtype=np.float32) for item in examples]
    durations = [
        np.array(item["durations"], dtype=np.int64) for item in examples
    ]
    num_phones = [
        np.array(item["num_phones"], dtype=np.int64) for item in examples
    ]
    num_frames = [
        np.array(item["num_frames"], dtype=np.int64) for item in examples
    ]

    phones = batch_sequences(phones)
    tones = batch_sequences(tones)
    feats = batch_sequences(feats)
    durations = batch_sequences(durations)

    # convert each batch to paddle.Tensor
    phones = default_collate(phones)
    tones = default_collate(tones)
    feats = default_collate(feats)
    durations = default_collate(durations)
    num_phones = default_collate(num_phones)
    num_frames = default_collate(num_frames)
    batch = {
        "phones": phones,
        "tones": tones,
        "num_phones": num_phones,
        "num_frames": num_frames,
        "feats": feats,
        "durations": durations,
    }
    if "spk_id" in examples[0]:
        spk_id = [np.array(item["spk_id"], dtype=np.int64) for item in examples]
        spk_id = default_collate(spk_id)
        batch["spk_id"] = spk_id
    return batch


def fastspeech2_single_spk_batch_fn(examples):
    # fields = ["text", "text_lengths", "speech", "speech_lengths", "durations", "pitch", "energy"]
    text = [np.array(item["text"], dtype=np.int64) for item in examples]
    speech = [np.array(item["speech"], dtype=np.float32) for item in examples]
    pitch = [np.array(item["pitch"], dtype=np.float32) for item in examples]
    energy = [np.array(item["energy"], dtype=np.float32) for item in examples]
    durations = [
        np.array(item["durations"], dtype=np.int64) for item in examples
    ]

    text_lengths = [
        np.array(item["text_lengths"], dtype=np.int64) for item in examples
    ]
    speech_lengths = [
        np.array(item["speech_lengths"], dtype=np.int64) for item in examples
    ]

    text = batch_sequences(text)
    pitch = batch_sequences(pitch)
    speech = batch_sequences(speech)
    durations = batch_sequences(durations)
    energy = batch_sequences(energy)

    # convert each batch to paddle.Tensor
    text = default_collate(text)
    pitch = default_collate(pitch)
    speech = default_collate(speech)
    durations = default_collate(durations)
    energy = default_collate(energy)
    text_lengths = default_collate(text_lengths)
    speech_lengths = default_collate(speech_lengths)

    batch = {
        "text": text,
        "text_lengths": text_lengths,
        "durations": durations,
        "speech": speech,
        "speech_lengths": speech_lengths,
        "pitch": pitch,
        "energy": energy
    }
    return batch


def fastspeech2_multi_spk_batch_fn(examples):
    # fields = ["text", "text_lengths", "speech", "speech_lengths", "durations", "pitch", "energy", "spk_id"/"spk_emb"]
    text = [np.array(item["text"], dtype=np.int64) for item in examples]
    speech = [np.array(item["speech"], dtype=np.float32) for item in examples]
    pitch = [np.array(item["pitch"], dtype=np.float32) for item in examples]
    energy = [np.array(item["energy"], dtype=np.float32) for item in examples]
    durations = [
        np.array(item["durations"], dtype=np.int64) for item in examples
    ]
    text_lengths = [
        np.array(item["text_lengths"], dtype=np.int64) for item in examples
    ]
    speech_lengths = [
        np.array(item["speech_lengths"], dtype=np.int64) for item in examples
    ]

    text = batch_sequences(text)
    pitch = batch_sequences(pitch)
    speech = batch_sequences(speech)
    durations = batch_sequences(durations)
    energy = batch_sequences(energy)

    # convert each batch to paddle.Tensor
    text = default_collate(text)
    pitch = default_collate(pitch)
    speech = default_collate(speech)
    durations = default_collate(durations)
    energy = default_collate(energy)
    text_lengths = default_collate(text_lengths)
    speech_lengths = default_collate(speech_lengths)

    batch = {
        "text": text,
        "text_lengths": text_lengths,
        "durations": durations,
        "speech": speech,
        "speech_lengths": speech_lengths,
        "pitch": pitch,
        "energy": energy
    }
    # spk_emb has a higher priority than spk_id
    if "spk_emb" in examples[0]:
        spk_emb = [
            np.array(item["spk_emb"], dtype=np.float32) for item in examples
        ]
        spk_emb = batch_sequences(spk_emb)
        spk_emb = default_collate(spk_emb)
        batch["spk_emb"] = spk_emb
    elif "spk_id" in examples[0]:
        spk_id = [np.array(item["spk_id"], dtype=np.int64) for item in examples]
        spk_id = default_collate(spk_id)
        batch["spk_id"] = spk_id
    return batch


def transformer_single_spk_batch_fn(examples):
    # fields = ["text", "text_lengths", "speech", "speech_lengths"]
    text = [np.array(item["text"], dtype=np.int64) for item in examples]
    speech = [np.array(item["speech"], dtype=np.float32) for item in examples]
    text_lengths = [
        np.array(item["text_lengths"], dtype=np.int64) for item in examples
    ]
    speech_lengths = [
        np.array(item["speech_lengths"], dtype=np.int64) for item in examples
    ]

    text = batch_sequences(text)
    speech = batch_sequences(speech)

    # convert each batch to paddle.Tensor
    text = default_collate(text)
    speech = default_collate(speech)
    text_lengths = default_collate(text_lengths)
    speech_lengths = default_collate(speech_lengths)

    batch = {
        "text": text,
        "text_lengths": text_lengths,
        "speech": speech,
        "speech_lengths": speech_lengths,
    }
    return batch


def vits_single_spk_batch_fn(examples):
    """
    Returns:
        Dict[str, Any]:
            - text (Tensor): Text index tensor (B, T_text).
            - text_lengths (Tensor): Text length tensor (B,).
            - feats (Tensor): Feature tensor (B, T_feats, aux_channels).
            - feats_lengths (Tensor): Feature length tensor (B,).
            - speech (Tensor): Speech waveform tensor (B, T_wav).

    """
    # fields = ["text", "text_lengths", "feats", "feats_lengths", "speech"]
    text = [np.array(item["text"], dtype=np.int64) for item in examples]
    feats = [np.array(item["feats"], dtype=np.float32) for item in examples]
    speech = [np.array(item["wave"], dtype=np.float32) for item in examples]
    text_lengths = [
        np.array(item["text_lengths"], dtype=np.int64) for item in examples
    ]
    feats_lengths = [
        np.array(item["feats_lengths"], dtype=np.int64) for item in examples
    ]

    text = batch_sequences(text)
    feats = batch_sequences(feats)
    speech = batch_sequences(speech)

    # convert each batch to paddle.Tensor
    text = default_collate(text)
    feats = default_collate(feats)
    text_lengths = default_collate(text_lengths)
    feats_lengths = default_collate(feats_lengths)

    batch = {
        "text": text,
        "text_lengths": text_lengths,
        "feats": feats,
        "feats_lengths": feats_lengths,
        "speech": speech
    }
    return batch
