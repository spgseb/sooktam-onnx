import os
import re
import time
from pathlib import Path

import numpy as np
import onnxruntime
import soundfile as sf
import torch
from pydub import AudioSegment

from infer.cls_tokenizer_v2 import cls_tokenize_text


DEFAULT_VOCAB_PATH = "./infer/vocab.txt"
DEFAULT_ONNX_MODEL_A = "../onnx/F5_Preprocess.onnx"
DEFAULT_ONNX_MODEL_B = "../onnx/F5_Transformer.onnx"
DEFAULT_ONNX_MODEL_C = "../onnx/F5_Decode.onnx"
DEFAULT_OUTPUT_PATH = "../outputs/generated_audio_gpu.wav"

DEFAULT_REFERENCE_AUDIO = "./infer/ref.wav"
DEFAULT_REF_TEXT = "सर, मैं तब से यह कह रहा हूँ कि मैंने अपना टिकट कैंसल कर दिया है, लेकिन अब तक मेरे पैसे वापस नहीं आए हैं। आप इस मामले को देखेंगे भी या नहीं"
DEFAULT_GEN_TEXT = "सर, मैं तब से यह कह रहा हूँ कि मैंने अपना टिकट कैंसल कर दिया है, लेकिन अब तक मेरे पैसे वापस नहीं आए हैं। आप इस मामले को देखेंगे भी या नहीं?"
DEFAULT_LANGUAGE = "hindi"

RANDOM_SEED = 9527
DEFAULT_NFE_STEP = 32
DEFAULT_FUSE_NFE = 1
DEFAULT_SPEED = 1.0
DEFAULT_MAX_THREADS = 8
DEFAULT_DEVICE_ID = int(os.getenv("ORT_DEVICE_ID", "0"))
MODEL_SAMPLE_RATE = 24000
HOP_LENGTH = 256

TENSORRT_PROVIDER = "TensorrtExecutionProvider"
CUDA_PROVIDER = "CUDAExecutionProvider"
CPU_PROVIDER = "CPUExecutionProvider"


def build_trt_provider_options(device_id):
    return {
        "device_id": device_id,
        "trt_engine_cache_enable": True,
        "trt_engine_cache_path": "../outputs",
        "trt_timing_cache_enable": True,
        "trt_timing_cache_path": "../outputs",
        "trt_max_workspace_size": 2147483648, # 2GB
    }

def build_cuda_provider_options(device_id):
    return {
        "device_id": device_id,
        "arena_extend_strategy": "kNextPowerOfTwo",
        "cudnn_conv_algo_search": "EXHAUSTIVE",
        "use_tf32": "1",
        "cudnn_conv_use_max_workspace": "1",
        "enable_cuda_graph": "0",
    }

def build_gpu_session_providers(device_id, trt_flag):
    if trt_flag: # Enable TensorRT Execution Provider
        return [(TENSORRT_PROVIDER, build_trt_provider_options(device_id)), (CUDA_PROVIDER, build_cuda_provider_options(device_id)), CPU_PROVIDER]
    else: # Use CUDA Execution Provider
        return [(CUDA_PROVIDER, build_cuda_provider_options(device_id)), CPU_PROVIDER]

def list_str_to_idx(
    text: list[str] | list[list[str]],
    vocab_char_map: dict[str, int],
    padding_value=-1,
):
    if text and isinstance(text[0], str):
        text = [text]
    get_idx = vocab_char_map.get
    list_idx_tensors = [torch.tensor([get_idx(c, 0) for c in t], dtype=torch.int32) for t in text]
    return torch.nn.utils.rnn.pad_sequence(list_idx_tensors, padding_value=padding_value, batch_first=True)


def load_vocab(vocab_path=DEFAULT_VOCAB_PATH):
    vocab_char_map = {}
    with open(vocab_path, "r", encoding="utf-8") as f:
        for i, char in enumerate(f):
            vocab_char_map[char[:-1]] = i
    return vocab_char_map


def build_session_options(max_threads=DEFAULT_MAX_THREADS):
    session_opts = onnxruntime.SessionOptions()
    session_opts.log_severity_level = 4
    session_opts.log_verbosity_level = 4
    session_opts.inter_op_num_threads = max_threads
    session_opts.intra_op_num_threads = max_threads
    session_opts.enable_cpu_mem_arena = True
    session_opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
    session_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_opts.add_session_config_entry("session.set_denormal_as_zero", "1")
    session_opts.add_session_config_entry("session.intra_op.allow_spinning", "1")
    session_opts.add_session_config_entry("session.inter_op.allow_spinning", "1")
    session_opts.add_session_config_entry("session.enable_quant_qdq_cleanup", "1")
    session_opts.add_session_config_entry("session.qdq_matmulnbits_accuracy_level", "4")
    session_opts.add_session_config_entry("optimization.enable_gelu_approximation", "1")
    session_opts.add_session_config_entry("disable_synchronize_execution_providers", "1")
    session_opts.add_session_config_entry("optimization.minimal_build_optimizations", "")
    session_opts.add_session_config_entry("session.use_device_allocator_for_initializers", "1")
    return session_opts


def prepare_inputs(
    vocab_char_map,
    reference_audio,
    ref_text,
    gen_text,
    speed=DEFAULT_SPEED,
    language=DEFAULT_LANGUAGE,
    verbose=True,
):
    if verbose:
        print(f"\nReference Audio: {reference_audio}")

    audio = np.array(
        AudioSegment.from_file(reference_audio)
        .set_channels(1)
        .set_frame_rate(MODEL_SAMPLE_RATE)
        .get_array_of_samples(),
        dtype=np.int16,
    ).reshape(1, 1, -1)

    audio_len = audio.shape[-1]
    zh_pause_punc = r"。，、；：？！"
    ref_text_len = len(ref_text.encode("utf-8")) + 3 * len(re.findall(zh_pause_punc, ref_text))
    gen_text_len = len(gen_text.encode("utf-8")) + 3 * len(re.findall(zh_pause_punc, gen_text))
    ref_audio_len = audio_len // HOP_LENGTH + 1
    max_duration = np.array([ref_audio_len + int(ref_audio_len / ref_text_len * gen_text_len / speed)], dtype=np.int64)
    tokenized = cls_tokenize_text(ref_text + gen_text, language=language)
    text_ids = list_str_to_idx(tokenized, vocab_char_map).numpy()
    time_step = np.array([0], dtype=np.int32)
    return audio, text_ids, max_duration, time_step


def to_cuda_ortvalue(array, device_id):
    return onnxruntime.OrtValue.ortvalue_from_numpy(np.ascontiguousarray(array), "cuda", device_id)


def run_transformer_on_gpu(session, model_inputs, nfe_step, fuse_nfe, device_id, verbose=True):
    input_names = [meta.name for meta in session.get_inputs()]
    output_names = [meta.name for meta in session.get_outputs()]

    ort_inputs = [to_cuda_ortvalue(value, device_id) for value in model_inputs]
    ort_outputs = [ort_inputs[0], ort_inputs[-1]]

    io_binding = session.io_binding()
    for name, value in zip(input_names, ort_inputs):
        io_binding.bind_ortvalue_input(name, value)
    for name, value in zip(output_names, ort_outputs):
        io_binding.bind_ortvalue_output(name, value)

    if verbose:
        print("NFE_STEP: 0")
    for i in range(0, nfe_step - 1, fuse_nfe):
        session.run_with_iobinding(io_binding)
        if verbose:
            print(f"NFE_STEP: {i + fuse_nfe}")

    noise_gpu, time_step_gpu = io_binding.get_outputs()
    return noise_gpu.numpy(), time_step_gpu.numpy()


def build_runner(
    device_id=DEFAULT_DEVICE_ID,
    max_threads=DEFAULT_MAX_THREADS,
    vocab_path=DEFAULT_VOCAB_PATH,
    onnx_model_A=DEFAULT_ONNX_MODEL_A,
    onnx_model_B=DEFAULT_ONNX_MODEL_B,
    onnx_model_C=DEFAULT_ONNX_MODEL_C,
    trt_flag=False,
):
    available_providers = onnxruntime.get_available_providers()
    if CUDA_PROVIDER not in available_providers:
        raise RuntimeError(
            f"{CUDA_PROVIDER} is not available in this environment. Available providers: {available_providers}"
        )

    onnxruntime.set_seed(RANDOM_SEED)
    session_opts = build_session_options(max_threads=max_threads)
    vocab_char_map = load_vocab(vocab_path=vocab_path)

    ort_session_A = onnxruntime.InferenceSession(
        onnx_model_A,
        sess_options=session_opts,
        providers=[CPU_PROVIDER],
    )
    ort_session_B = onnxruntime.InferenceSession(
        onnx_model_B,
        sess_options=session_opts,
        providers=build_gpu_session_providers(device_id, trt_flag),
    )
    ort_session_C = onnxruntime.InferenceSession(
        onnx_model_C,
        sess_options=session_opts,
        providers=[CPU_PROVIDER],
    )

    return {
        "device_id": device_id,
        "available_providers": available_providers,
        "transformer_providers": ort_session_B.get_providers(),
        "vocab_char_map": vocab_char_map,
        "ort_session_A": ort_session_A,
        "ort_session_B": ort_session_B,
        "ort_session_C": ort_session_C,
        "in_names_A": [meta.name for meta in ort_session_A.get_inputs()],
        "out_names_A": [meta.name for meta in ort_session_A.get_outputs()],
        "in_names_C": [meta.name for meta in ort_session_C.get_inputs()],
        "out_names_C": [meta.name for meta in ort_session_C.get_outputs()],
    }


def run_inference(
    runner,
    reference_audio=DEFAULT_REFERENCE_AUDIO,
    ref_text=DEFAULT_REF_TEXT,
    gen_text=DEFAULT_GEN_TEXT,
    generated_audio=DEFAULT_OUTPUT_PATH,
    speed=DEFAULT_SPEED,
    nfe_step=DEFAULT_NFE_STEP,
    fuse_nfe=DEFAULT_FUSE_NFE,
    language=DEFAULT_LANGUAGE,
    verbose=True,
):
    audio, text_ids, max_duration, time_step = prepare_inputs(
        runner["vocab_char_map"],
        reference_audio=reference_audio,
        ref_text=ref_text,
        gen_text=gen_text,
        speed=speed,
        language=language,
        verbose=verbose,
    )

    if verbose:
        print("\n\nRun F5-TTS by ONNX Runtime with GPU transformer.")
    start_count = time.time()
    noise, rope_cos_q, rope_sin_q, rope_cos_k, rope_sin_k, cat_mel_text, cat_mel_text_drop, ref_signal_len = runner[
        "ort_session_A"
    ].run(
        runner["out_names_A"],
        {
            runner["in_names_A"][0]: audio,
            runner["in_names_A"][1]: text_ids,
            runner["in_names_A"][2]: max_duration,
        },
    )

    noise, time_step = run_transformer_on_gpu(
        runner["ort_session_B"],
        [
            noise,
            rope_cos_q,
            rope_sin_q,
            rope_cos_k,
            rope_sin_k,
            cat_mel_text,
            cat_mel_text_drop,
            time_step,
        ],
        nfe_step=nfe_step,
        fuse_nfe=fuse_nfe,
        device_id=runner["device_id"],
        verbose=verbose,
    )

    generated_signal = runner["ort_session_C"].run(
        runner["out_names_C"],
        {
            runner["in_names_C"][0]: noise,
            runner["in_names_C"][1]: ref_signal_len,
        },
    )[0]
    end_count = time.time()

    flat_signal = generated_signal.reshape(-1)
    output_path = None
    if generated_audio:
        output_path = Path(generated_audio).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_path, flat_signal, MODEL_SAMPLE_RATE, format="WAVEX")

    audio_seconds = float(len(flat_signal) / MODEL_SAMPLE_RATE)
    elapsed_seconds = float(end_count - start_count)
    rtf = elapsed_seconds / audio_seconds if audio_seconds > 0 else float("inf")

    result = {
        "elapsed_seconds": elapsed_seconds,
        "audio_seconds": audio_seconds,
        "rtf": rtf,
        "num_samples": int(len(flat_signal)),
        "sample_rate": MODEL_SAMPLE_RATE,
        "output_path": str(output_path) if output_path else None,
        "device_id": runner["device_id"],
        "language": language,
        "nfe_step": nfe_step,
        "fuse_nfe": fuse_nfe,
        "time_step": time_step,
    }

    if verbose:
        print(f"\nAudio generation is complete.\n\nONNXRuntime Time Cost in Seconds:\n{elapsed_seconds:.3f}")
        print(f"Generated Audio Seconds: {audio_seconds:.3f}")
        print(f"RTF: {rtf:.4f}")
        if output_path:
            print(f"\nSaved: {output_path}")
    return result


def main():
    runner = build_runner(device_id=DEFAULT_DEVICE_ID, max_threads=DEFAULT_MAX_THREADS, trt_flag=True)
    print(f"\nAvailable Providers: {runner['available_providers']}")
    print(f"Transformer Providers: {runner['transformer_providers']}")
    print(f"Using CUDA device id: {runner['device_id']}")

    run_inference(
        runner,
        reference_audio=DEFAULT_REFERENCE_AUDIO,
        ref_text=DEFAULT_REF_TEXT,
        gen_text=DEFAULT_GEN_TEXT,
        generated_audio=DEFAULT_OUTPUT_PATH,
        speed=DEFAULT_SPEED,
        nfe_step=DEFAULT_NFE_STEP,
        fuse_nfe=DEFAULT_FUSE_NFE,
        language=DEFAULT_LANGUAGE,
        verbose=True,
    )


if __name__ == "__main__":
    main()
