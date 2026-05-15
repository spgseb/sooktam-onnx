import csv
import importlib.util
import json
import os
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
GPU_INFERENCE_SCRIPT = SCRIPT_DIR / "F5-TTS-ONNX-Inference-GPU.py"
OUTPUT_DIR = SCRIPT_DIR / "../outputs"
SUMMARY_CSV = OUTPUT_DIR / "results.csv"
SUMMARY_JSON = OUTPUT_DIR / "summary.json"

WARMUP_TEXT = "यह वार्मअप रन है ताकि पहला मापन मॉडल लोड होने के समय से प्रभावित न हो।"

HINDI_EXAMPLES = [
    "नमस्ते, आपका ऑर्डर आज शाम तक पहुंच जाएगा।",
    "कृपया बैठक शुरू होने से पहले सभी दस्तावेज साझा कर दीजिए।",
    "मुझे अगले हफ्ते दिल्ली से लखनऊ की टिकट बुक करनी है।",
    "अगर बारिश तेज हुई तो हमें कार्यक्रम भीतर शिफ्ट करना पड़ेगा।",
    "ग्राहक ने कहा कि भुगतान सफल हुआ है, लेकिन पुष्टि संदेश अभी तक नहीं मिला।",
    "कृपया जांच कर बताइए कि सर्वर रात दो बजे के बाद क्यों बंद हुआ।",
    "आज की प्रस्तुति में हमें केवल मुख्य निष्कर्ष और अगले कदम दिखाने हैं।",
    "जब आप खाली हों तो मुझे फोन करिए, मुझे परियोजना की समयसीमा पर बात करनी है।",
    "यह दवा खाने के बाद कम से कम आधा घंटा आराम करना बेहतर रहेगा।",
    "रेलवे स्टेशन पर भीड़ बहुत ज्यादा थी, इसलिए हमें प्लेटफॉर्म तक पहुंचने में समय लगा।",
    "हम चाहते हैं कि नया मॉडल कम विलंबता के साथ स्थिर आवाज भी बनाए रखे।",
    "अगर उपयोगकर्ता एक साथ कई अनुरोध भेजें तो हमें समग्र थ्रूपुट और औसत आरटीएफ दोनों मापने चाहिए।",
    "कृपया सुनिश्चित करें कि सभी लॉग फाइलें परीक्षण के बाद अलग फोल्डर में सुरक्षित हो जाएं।",
    "इस रिपोर्ट में पिछले तीन महीनों के बिक्री आंकड़े, क्षेत्रीय तुलना और प्रमुख जोखिम शामिल हैं।",
    "मेरा मानना है कि सही अनुकूलन के बाद यह प्रणाली वास्तविक समय से तेज प्रदर्शन दे सकती है।",
]


def load_gpu_inference_module():
    spec = importlib.util.spec_from_file_location("f5_tts_onnx_inference_gpu", GPU_INFERENCE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load GPU inference module from {GPU_INFERENCE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def percentile(values, q):
    if not values:
        return float("nan")
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def shorten(text, limit=72):
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def write_csv(rows):
    fieldnames = [
        "example_id",
        "elapsed_seconds",
        "audio_seconds",
        "rtf",
        "num_samples",
        "sample_rate",
        "text_length_chars",
        "output_path",
        "text",
    ]
    with open(SUMMARY_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    module = load_gpu_inference_module()

    runner = module.build_runner(
        device_id=module.DEFAULT_DEVICE_ID,
        max_threads=module.DEFAULT_MAX_THREADS,
        trt_flag=True
    )
    print(f"Available Providers: {runner['available_providers']}")
    print(f"Transformer Providers: {runner['transformer_providers']}")
    print(f"Using CUDA device id: {runner['device_id']}")
    print(f"Load-test output dir: {OUTPUT_DIR}")

    print("\nRunning warmup request...")
    module.run_inference(
        runner,
        reference_audio=module.DEFAULT_REFERENCE_AUDIO,
        ref_text=module.DEFAULT_REF_TEXT,
        gen_text=WARMUP_TEXT,
        generated_audio=None,
        language="hindi",
        verbose=False,
    )

    rows = []
    total_elapsed_seconds = 0.0
    total_audio_seconds = 0.0

    print(f"\nRunning {len(HINDI_EXAMPLES)} Hindi examples...")
    for index, text in enumerate(HINDI_EXAMPLES, start=1):
        output_path = OUTPUT_DIR / f"hindi_{index:02d}.wav"
        result = module.run_inference(
            runner,
            reference_audio=module.DEFAULT_REFERENCE_AUDIO,
            ref_text=module.DEFAULT_REF_TEXT,
            gen_text=text,
            generated_audio=str(output_path),
            language="hindi",
            verbose=False,
        )

        row = {
            "example_id": index,
            "elapsed_seconds": round(result["elapsed_seconds"], 6),
            "audio_seconds": round(result["audio_seconds"], 6),
            "rtf": round(result["rtf"], 6),
            "num_samples": result["num_samples"],
            "sample_rate": result["sample_rate"],
            "text_length_chars": len(text),
            "output_path": result["output_path"],
            "text": text,
        }
        rows.append(row)
        total_elapsed_seconds += result["elapsed_seconds"]
        total_audio_seconds += result["audio_seconds"]

        print(
            f"[{index:02d}/{len(HINDI_EXAMPLES)}] "
            f"elapsed={result['elapsed_seconds']:.3f}s "
            f"audio={result['audio_seconds']:.3f}s "
            f"rtf={result['rtf']:.4f} "
            f"text={shorten(text)}"
        )

    write_csv(rows)

    rtfs = [row["rtf"] for row in rows]
    overall_rtf = total_elapsed_seconds / total_audio_seconds if total_audio_seconds > 0 else float("inf")
    summary = {
        "num_examples": len(rows),
        "language": "hindi",
        "warmup_excluded_from_metrics": True,
        "device_id": runner["device_id"],
        "available_providers": runner["available_providers"],
        "transformer_providers": runner["transformer_providers"],
        "total_elapsed_seconds": round(total_elapsed_seconds, 6),
        "total_audio_seconds": round(total_audio_seconds, 6),
        "overall_rtf": round(overall_rtf, 6),
        "avg_rtf": round(sum(rtfs) / len(rtfs), 6),
        "p50_rtf": round(percentile(rtfs, 0.50), 6),
        "p95_rtf": round(percentile(rtfs, 0.95), 6),
        "min_rtf": round(min(rtfs), 6),
        "max_rtf": round(max(rtfs), 6),
        "x_realtime": round(1.0 / overall_rtf, 6) if overall_rtf > 0 else float("inf"),
        "results_csv": str(SUMMARY_CSV),
        "results_json": str(SUMMARY_JSON),
        "output_dir": str(OUTPUT_DIR),
    }

    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": rows}, f, ensure_ascii=False, indent=2)

    print("\nSummary")
    print(f"Examples: {summary['num_examples']}")
    print(f"Total elapsed seconds: {summary['total_elapsed_seconds']:.3f}")
    print(f"Total audio seconds: {summary['total_audio_seconds']:.3f}")
    print(f"Overall RTF: {summary['overall_rtf']:.4f}")
    print(f"Average RTF: {summary['avg_rtf']:.4f}")
    print(f"P50 RTF: {summary['p50_rtf']:.4f}")
    print(f"P95 RTF: {summary['p95_rtf']:.4f}")
    print(f"Realtime multiple: {summary['x_realtime']:.2f}x")
    print(f"CSV: {SUMMARY_CSV}")
    print(f"JSON: {SUMMARY_JSON}")


if __name__ == "__main__":
    main()
