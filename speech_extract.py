import datetime
import json
import os
import re
import subprocess
import tempfile
import uuid
from pathlib import Path

import torch

from ffmpeg_utils import find_ffmpeg, find_ffprobe, subprocess_kwargs

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
except ImportError:
    AutoModelForCausalLM = None
    AutoTokenizer = None
    pipeline = None
try:
    from qwen_asr import Qwen3ASRModel
except ImportError:
    Qwen3ASRModel = None


class ASREngine:
    DEFAULT_QWEN_MAX_NEW_TOKENS = 1024

    def __init__(self, model_id, device=None, asr_terms=None, asr_corrections=None):
        self.model_id = model_id
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.qwen_max_new_tokens = self._resolve_qwen_max_new_tokens()
        self.asr_terms = list(asr_terms or [])
        self.asr_corrections = dict(asr_corrections or {})
        self.qwen_prompt = self._build_qwen_prompt(self.asr_terms)
        self._backend = "transformers"
        self._pipeline = None
        self._qwen_model = None
        self._build_asr_backend()

    def _resolve_qwen_max_new_tokens(self):
        raw_value = os.getenv("QWEN_ASR_MAX_NEW_TOKENS", str(self.DEFAULT_QWEN_MAX_NEW_TOKENS)).strip()
        try:
            parsed = int(raw_value)
        except Exception:
            parsed = self.DEFAULT_QWEN_MAX_NEW_TOKENS
        return max(256, parsed)

    def _is_qwen3_asr(self):
        model = (self.model_id or "").lower()
        return "qwen3-asr" in model

    @staticmethod
    def _build_qwen_prompt(terms):
        cleaned = []
        seen = set()
        for term in terms:
            value = str(term).strip()
            if not value:
                continue
            key = value.lower()
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(value)
        if not cleaned:
            return None
        joined = "、".join(cleaned[:32])
        return (
            "这是日语语音转写任务。请保持专有名词和术语拼写稳定，优先使用以下词汇："
            f"{joined}"
        )

    def _build_asr_backend(self):
        if self._is_qwen3_asr():
            if Qwen3ASRModel is None:
                raise ImportError(
                    "使用 Qwen3-ASR 模型需要安装 qwen-asr 包。"
                    "请运行 pip install qwen-asr 后重试。"
                )
            self._backend = "qwen_asr"
            self._qwen_model = self._build_qwen_model()
            return
        if pipeline is None:
            raise ImportError("缺少 transformers 包。请运行 pip install -r requirements.txt")
        self._backend = "transformers"
        self._pipeline = self._build_pipeline()

    def _build_qwen_model(self):
        use_cuda = self.device.startswith("cuda") and torch.cuda.is_available()
        dtype = torch.bfloat16 if use_cuda else torch.float32
        device_map = self.device if use_cuda else "cpu"
        try:
            return Qwen3ASRModel.from_pretrained(
                self.model_id,
                dtype=dtype,
                device_map=device_map,
                max_inference_batch_size=32,
                max_new_tokens=self.qwen_max_new_tokens,
            )
        except Exception as load_error:
            if use_cuda:
                print(f"Qwen ASR 模型在 CUDA 加载失败，已回退到 CPU：{load_error}")
                return Qwen3ASRModel.from_pretrained(
                    self.model_id,
                    dtype=torch.float32,
                    device_map="cpu",
                    max_inference_batch_size=16,
                    max_new_tokens=self.qwen_max_new_tokens,
                )
            raise RuntimeError(f"加载 Qwen ASR 模型失败 {self.model_id}：{load_error}") from load_error

    def _build_pipeline(self):
        use_cuda = self.device.startswith("cuda") and torch.cuda.is_available()
        pipe_device = 0 if use_cuda else -1
        dtype = torch.float16 if use_cuda else torch.float32
        try:
            return pipeline(
                task="automatic-speech-recognition",
                model=self.model_id,
                device=pipe_device,
                torch_dtype=dtype,
            )
        except Exception as load_error:
            if use_cuda:
                print(f"ASR 模型在 CUDA 加载失败，已回退到 CPU：{load_error}")
                return pipeline(
                    task="automatic-speech-recognition",
                    model=self.model_id,
                    device=-1,
                    torch_dtype=torch.float32,
                )
            raise RuntimeError(f"加载 ASR 模型失败 {self.model_id}：{load_error}") from load_error

    def _normalize_segments(self, asr_output, audio_duration):
        chunks = asr_output.get("chunks") if isinstance(asr_output, dict) else None
        if not chunks:
            text = (asr_output.get("text", "") if isinstance(asr_output, dict) else str(asr_output)).strip()
            if not text:
                return []
            return [{"start": 0.0, "end": max(0.5, audio_duration), "text": text, "confidence": 1.0}]

        normalized = []
        for chunk in chunks:
            timestamp = chunk.get("timestamp") or (0.0, 0.0)
            start = float(timestamp[0] or 0.0)
            end = float(timestamp[1] or start + 0.5)
            text = str(chunk.get("text", "")).strip()
            if not text:
                continue
            score = chunk.get("score")
            normalized.append(
                {
                    "start": max(0.0, start),
                    "end": max(start + 0.1, end),
                    "text": text,
                    "confidence": float(score) if isinstance(score, (int, float)) else 1.0,
                }
            )
        return normalized

    def _normalize_qwen_segments(self, results, audio_duration):
        if not results:
            return []

        if isinstance(results, (list, tuple)):
            entries = list(results)
        else:
            entries = [results]

        normalized = []
        untimed_texts = []

        for entry in entries:
            raw_segments = None
            for attr_name in ("segments", "chunks", "time_stamps", "timestamps", "timestamp"):
                if hasattr(entry, attr_name):
                    raw_segments = getattr(entry, attr_name)
                    break
                if isinstance(entry, dict) and attr_name in entry:
                    raw_segments = entry.get(attr_name)
                    break

            if raw_segments:
                for segment in raw_segments:
                    if isinstance(segment, dict):
                        start = segment.get("start", segment.get("begin", segment.get("from", 0.0)))
                        end = segment.get("end", segment.get("stop", segment.get("to", start)))
                        text = segment.get("text", segment.get("token", ""))
                        score = segment.get("score", segment.get("confidence"))
                    elif isinstance(segment, (list, tuple)) and len(segment) >= 3:
                        start, end, text = segment[0], segment[1], segment[2]
                        score = segment[3] if len(segment) > 3 else 1.0
                    else:
                        continue
                    text = str(text).strip()
                    if not text:
                        continue
                    try:
                        start_f = float(start or 0.0)
                        end_f = float(end or start_f + 0.5)
                    except Exception:
                        continue
                    normalized.append(
                        {
                            "start": max(0.0, start_f),
                            "end": max(start_f + 0.1, end_f),
                            "text": text,
                            "confidence": float(score) if isinstance(score, (int, float)) else 1.0,
                        }
                    )
                continue

            text = ""
            if hasattr(entry, "text"):
                text = str(getattr(entry, "text", "")).strip()
            elif isinstance(entry, dict):
                text = str(entry.get("text", "")).strip()
            if text:
                untimed_texts.append(text)

        if normalized:
            normalized.sort(key=lambda x: (x["start"], x["end"]))
            return normalized

        if not untimed_texts:
            return []

        total_duration = max(0.5, float(audio_duration))
        span = total_duration / max(1, len(untimed_texts))
        fallback_segments = []
        for index, text in enumerate(untimed_texts):
            start = index * span
            end = total_duration if index == len(untimed_texts) - 1 else (index + 1) * span
            fallback_segments.append(
                {"start": start, "end": max(start + 0.1, end), "text": text, "confidence": 1.0}
            )
        return fallback_segments

    def _apply_asr_corrections(self, text):
        output = str(text or "")
        if not output or not self.asr_corrections:
            return output
        for source, target in self.asr_corrections.items():
            if source and target:
                output = output.replace(source, target)
        return output

    def transcribe(self, audio_path, quality_mode="fast"):
        if self._backend == "qwen_asr":
            audio_duration = max(0.5, JapaneseVideoSubtitleGenerator.get_audio_duration(audio_path))
            attempts = [
                {"language": "Japanese", "return_time_stamps": True},
                {"language": None, "return_time_stamps": True},
                {"language": "Japanese"},
                {"language": None},
            ]
            results = None
            last_error = None
            prompt_variants = [None]
            if self.qwen_prompt:
                prompt_variants = ["prompt", "system_prompt", "text_prompt", None]
            for kwargs in attempts:
                for prompt_key in prompt_variants:
                    merged = dict(kwargs)
                    if prompt_key and self.qwen_prompt:
                        merged[prompt_key] = self.qwen_prompt
                    try:
                        results = self._qwen_model.transcribe(audio=audio_path, **merged)
                        break
                    except Exception as err:
                        last_error = err
                        # Qwen requires forced_aligner for timestamps in some setups.
                        # Fallback to non-timestamp mode instead of failing the whole chunk.
                        continue
                if results is not None:
                    break
            if results is None:
                raise RuntimeError(f"Qwen ASR 转写失败：{last_error}") from last_error
            normalized = self._normalize_qwen_segments(
                results,
                audio_duration,
            )
            for segment in normalized:
                segment["text"] = self._apply_asr_corrections(segment.get("text", ""))
            return normalized

        generate_kwargs = {"language": "ja"}
        if quality_mode == "accurate":
            generate_kwargs.update({"num_beams": 4, "temperature": 0.0})
        try:
            output = self._pipeline(audio_path, return_timestamps=True, generate_kwargs=generate_kwargs)
        except Exception:
            output = self._pipeline(audio_path, return_timestamps=True)
        normalized = self._normalize_segments(
            output,
            max(0.5, JapaneseVideoSubtitleGenerator.get_audio_duration(audio_path)),
        )
        for segment in normalized:
            segment["text"] = self._apply_asr_corrections(segment.get("text", ""))
        return normalized

    def transcribe_region(
        self,
        audio_path,
        start_seconds,
        end_seconds,
        quality_mode="accurate",
        audio_filter=None,
    ):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_clip = temp_file.name
        try:
            JapaneseVideoSubtitleGenerator.extract_audio_span(
                audio_path,
                temp_clip,
                start_seconds,
                end_seconds,
                audio_filter=audio_filter,
            )
            region_segments = self.transcribe(temp_clip, quality_mode=quality_mode)
            duration = max(0.1, end_seconds - start_seconds)
            fixed_segments = []
            for segment in region_segments:
                fixed_segments.append(
                    {
                        "start": min(duration, max(0.0, segment["start"])) + start_seconds,
                        "end": min(duration, max(0.0, segment["end"])) + start_seconds,
                        "text": segment["text"],
                        "confidence": segment.get("confidence", 1.0),
                    }
                )
            return fixed_segments
        finally:
            if os.path.exists(temp_clip):
                os.remove(temp_clip)


class TranslationEngine:
    def __init__(self, primary_model_id, advanced_model_id, use_advanced=False, device=None):
        if pipeline is None and (AutoModelForCausalLM is None or AutoTokenizer is None):
            raise ImportError("缺少 transformers 包。请运行 pip install -r requirements.txt")
        self.primary_model_id = primary_model_id
        self.advanced_model_id = advanced_model_id
        self.use_advanced = use_advanced
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.memory = {}
        self._backend = "pipeline"
        self._pipeline = None
        self._tokenizer = None
        self._model = None
        self.effective_model_id = None
        self._load_pipeline()

    def _is_hy_mt_model(self, model_id):
        text = (model_id or "").lower()
        return "hy-mt" in text

    def _load_hy_model(self, model_id):
        if AutoModelForCausalLM is None or AutoTokenizer is None:
            raise RuntimeError("transformers 的 AutoModel/AutoTokenizer 不可用。")

        use_cuda = self.device.startswith("cuda") and torch.cuda.is_available()
        model_kwargs = {"device_map": self.device if use_cuda else "cpu", "trust_remote_code": True}
        if use_cuda:
            model_kwargs["torch_dtype"] = torch.bfloat16

        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        return tokenizer, model

    def _load_translation_pipeline(self, model_id):
        use_cuda = self.device.startswith("cuda") and torch.cuda.is_available()
        pipe_device = 0 if use_cuda else -1
        dtype = torch.float16 if use_cuda else torch.float32
        tasks = ["translation", "text2text-generation", "text-generation"]
        last_error = None
        for task_name in tasks:
            try:
                return pipeline(task=task_name, model=model_id, device=pipe_device, torch_dtype=dtype), task_name
            except Exception as err:
                last_error = err
        raise RuntimeError(f"加载翻译模型失败 {model_id}：{last_error}")

    def _load_pipeline(self):
        model_candidates = []
        if self.use_advanced:
            model_candidates.append(self.advanced_model_id)
        model_candidates.append(self.primary_model_id)
        if "Helsinki-NLP/opus-mt-ja-zh" not in model_candidates:
            model_candidates.append("Helsinki-NLP/opus-mt-ja-zh")
        errors = []
        for model_id in model_candidates:
            try:
                if self._is_hy_mt_model(model_id):
                    self._tokenizer, self._model = self._load_hy_model(model_id)
                    self._backend = "hy_chat"
                    self._task_name = "chat-generation"
                else:
                    self._pipeline, self._task_name = self._load_translation_pipeline(model_id)
                    self._backend = "pipeline"
                self.effective_model_id = model_id
                if model_id == self.advanced_model_id and self.use_advanced:
                    print(f"使用高级 MT 模型：{model_id}")
                elif self.use_advanced and model_id == self.primary_model_id:
                    print(f"高级 MT 模型加载失败，已自动回退到：{model_id}")
                return
            except Exception as err:
                errors.append(str(err))
        raise RuntimeError("无法加载任何 MT 模型：\n" + "\n".join(errors))

    def _make_prompt(self, text):
        return (
            "Translate the following Japanese subtitle into natural Simplified Chinese. "
            "Keep names and special terms stable.\n"
            f"Japanese: {text}\nChinese:"
        )

    def _translate_with_hy_chat(self, text):
        messages = [
            {
                "role": "user",
                "content": (
                    "Translate the following segment into Chinese, without additional explanation.\n\n"
                    f"{text}"
                ),
            }
        ]

        tokenized_chat = self._tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors="pt",
        )
        input_ids = tokenized_chat.to(self._model.device)

        outputs = self._model.generate(
            input_ids,
            max_new_tokens=256,
            do_sample=True,
            top_k=20,
            top_p=0.6,
            temperature=0.7,
            repetition_penalty=1.05,
        )

        generated_tokens = outputs[0][input_ids.shape[-1]:]
        translated = self._tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        return translated or text

    def translate(self, text, glossary=None):
        key = text.strip()
        if not key:
            return ""
        if key in self.memory:
            return self.memory[key]

        if self._backend == "hy_chat":
            translated = self._translate_with_hy_chat(key)
        else:
            prompt = self._make_prompt(key)
            kwargs = {"max_new_tokens": 128, "do_sample": False}
            try:
                result = self._pipeline(prompt, **kwargs)
            except Exception:
                result = self._pipeline(prompt)
            translated = key
            if isinstance(result, list) and result:
                candidate = result[0]
                translated = (
                    candidate.get("translation_text")
                    or candidate.get("generated_text")
                    or candidate.get("summary_text")
                    or key
                )
            translated = str(translated).strip()
            if translated.startswith(prompt):
                translated = translated[len(prompt):].strip()
        translated = JapaneseVideoSubtitleGenerator.apply_glossary(key, translated, glossary)
        self.memory[key] = translated or key
        return self.memory[key]


class JapaneseVideoSubtitleGenerator:
    DEFAULT_ASR_MODEL = "Qwen/Qwen3-ASR-1.7B"
    DEFAULT_MT_MODEL = "tencent/HY-MT1.5-1.8B"
    DEFAULT_ADV_MT_MODEL = "tencent/HY-MT1.5-7B"
    CHECKPOINT_VERSION = 5
    AUDIO_PRESET_STANDARD = "standard"
    AUDIO_PRESET_DENOISE = "denoise"
    AUDIO_PRESET_AGGRESSIVE = "aggressive"
    AUDIO_FILTER_PRESETS = {
        AUDIO_PRESET_STANDARD: "highpass=f=80,lowpass=f=8000,volume=1.5,dynaudnorm",
        AUDIO_PRESET_DENOISE: (
            "highpass=f=90,lowpass=f=7800,"
            "afftdn=nr=14:nf=-28:tn=1,"
            "acompressor=threshold=-21dB:ratio=2.2:attack=15:release=220:makeup=2,"
            "dynaudnorm=f=200:g=11:p=0.85"
        ),
        AUDIO_PRESET_AGGRESSIVE: (
            "highpass=f=100,lowpass=f=7600,"
            "afftdn=nr=20:nf=-30:tn=1,"
            "acompressor=threshold=-24dB:ratio=3.0:attack=10:release=260:makeup=3,"
            "dynaudnorm=f=150:g=13:p=0.9"
        ),
    }

    def __init__(
        self,
        model_name="Qwen/Qwen3-ASR-1.7B",
        device=None,
        asr_model_id=None,
        mt_model_id=None,
        mt_advanced_model_id=None,
        use_advanced_mt=False,
        quality_mode="fast",
        glossary_path=None,
        asr_terms_path=None,
        audio_preset=AUDIO_PRESET_STANDARD,
        progress_callback=None,
    ):
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.asr_model_id = asr_model_id or self._resolve_asr_model(model_name)
        self.mt_model_id = self._normalize_mt_model_id(mt_model_id or self.DEFAULT_MT_MODEL)
        self.mt_advanced_model_id = self._normalize_mt_model_id(mt_advanced_model_id or self.DEFAULT_ADV_MT_MODEL)
        self.quality_mode = quality_mode if quality_mode in {"fast", "accurate"} else "fast"
        self.glossary_path = glossary_path
        self.glossary = self.load_glossary(glossary_path)
        self.asr_terms_path = asr_terms_path
        self.audio_preset = self.resolve_audio_preset(audio_preset)
        self.asr_terms, self.asr_corrections = self.load_asr_terms(asr_terms_path)
        self.progress_callback = progress_callback
        print(f"ASR 模型：{self.asr_model_id}")
        print(f"MT 模型：{self.mt_model_id}（高级：{self.mt_advanced_model_id}）")
        print(f"设备偏好：{self.device}")
        print(f"音频增强预设：{self.audio_preset}")
        if self.asr_terms:
            print(f"ASR 术语条目：{len(self.asr_terms)}")
        if self.asr_corrections:
            print(f"ASR 修正规则：{len(self.asr_corrections)}")
        self.asr_engine = ASREngine(
            model_id=self.asr_model_id,
            device=self.device,
            asr_terms=self.asr_terms,
            asr_corrections=self.asr_corrections,
        )
        self.translation_engine = TranslationEngine(
            primary_model_id=self.mt_model_id,
            advanced_model_id=self.mt_advanced_model_id,
            use_advanced=use_advanced_mt,
            device=self.device,
        )
        print(f"最终使用的 MT 模型：{self.translation_engine.effective_model_id}")

    def _emit_progress(self, percent, message):
        callback = self.progress_callback
        if callback is None:
            return
        try:
            callback(float(max(0.0, min(100.0, percent))), str(message))
        except Exception:
            pass

    @staticmethod
    def _resolve_asr_model(model_name):
        whisper_aliases = {"tiny", "base", "small", "medium", "large", "large-v2", "large-v3"}
        if model_name in whisper_aliases:
            return JapaneseVideoSubtitleGenerator.DEFAULT_ASR_MODEL
        return model_name or JapaneseVideoSubtitleGenerator.DEFAULT_ASR_MODEL

    @staticmethod
    def _normalize_mt_model_id(model_id):
        if not model_id:
            return JapaneseVideoSubtitleGenerator.DEFAULT_MT_MODEL
        text = str(model_id).strip()
        alias_map = {
            "HY-MT1.5-1.8B": "tencent/HY-MT1.5-1.8B",
            "HY-MT1.5-7B": "tencent/HY-MT1.5-7B",
        }
        return alias_map.get(text, text)

    @staticmethod
    def _subprocess_kwargs():
        return subprocess_kwargs()

    @staticmethod
    def _ffmpeg_bin():
        return find_ffmpeg()

    @staticmethod
    def _ffprobe_bin():
        return find_ffprobe()

    @staticmethod
    def get_audio_duration(audio_path):
        ffprobe_bin = JapaneseVideoSubtitleGenerator._ffprobe_bin()
        cmd = [
            ffprobe_bin,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            audio_path,
        ]
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            **JapaneseVideoSubtitleGenerator._subprocess_kwargs(),
        )
        return float(result.stdout.strip())

    @staticmethod
    def extract_audio_span(source_audio_path, output_audio_path, start_seconds, end_seconds, audio_filter=None):
        ffmpeg_bin = JapaneseVideoSubtitleGenerator._ffmpeg_bin()
        cmd = [
            ffmpeg_bin,
            "-hide_banner",
            "-i",
            source_audio_path,
            "-ss",
            str(max(0.0, start_seconds)),
            "-to",
            str(max(start_seconds + 0.1, end_seconds)),
        ]
        if audio_filter:
            cmd.extend(["-af", str(audio_filter)])
        cmd.extend(
            [
                "-acodec",
                "pcm_s16le",
                "-ar",
                "16000",
                "-ac",
                "1",
                output_audio_path,
                "-y",
            ]
        )
        subprocess.run(cmd, check=True, capture_output=True, **JapaneseVideoSubtitleGenerator._subprocess_kwargs())

    @staticmethod
    def apply_glossary(japanese_text, chinese_text, glossary):
        if not glossary:
            return chinese_text
        output = chinese_text
        for ja_term, zh_term in glossary.items():
            if ja_term and zh_term and ja_term in japanese_text:
                output = output.replace(ja_term, zh_term)
        return output

    @staticmethod
    def load_glossary(glossary_path):
        if not glossary_path:
            return {}
        glossary = {}
        try:
            with open(glossary_path, "r", encoding="utf-8") as f:
                for raw_line in f:
                    line = raw_line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" in line:
                        source, target = line.split("=", 1)
                    elif "\t" in line:
                        source, target = line.split("\t", 1)
                    else:
                        continue
                    glossary[source.strip()] = target.strip()
        except Exception as err:
            print(f"警告：加载术语表失败 {glossary_path}：{err}")
        return glossary

    @classmethod
    def resolve_audio_preset(cls, preset):
        normalized = str(preset or "").strip().lower()
        if normalized in cls.AUDIO_FILTER_PRESETS:
            return normalized
        return cls.AUDIO_PRESET_STANDARD

    @classmethod
    def get_audio_filter_chain(cls, preset):
        normalized = cls.resolve_audio_preset(preset)
        return cls.AUDIO_FILTER_PRESETS.get(normalized, cls.AUDIO_FILTER_PRESETS[cls.AUDIO_PRESET_STANDARD])

    @staticmethod
    def load_asr_terms(asr_terms_path):
        if not asr_terms_path:
            return [], {}
        terms = []
        corrections = {}
        seen = set()
        try:
            with open(asr_terms_path, "r", encoding="utf-8") as f:
                for raw_line in f:
                    line = raw_line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=>" in line:
                        source, target = line.split("=>", 1)
                        source = source.strip()
                        target = target.strip()
                        if source and target:
                            corrections[source] = target
                        continue
                    term = line.strip()
                    key = term.lower()
                    if term and key not in seen:
                        seen.add(key)
                        terms.append(term)
        except Exception as err:
            print(f"警告：加载 ASR 术语失败 {asr_terms_path}：{err}")
        return terms, corrections

    def extract_audio(self, video_path, audio_path, audio_preset=None):
        print("正在从视频中提取音频...")
        ffmpeg_bin = self._ffmpeg_bin()
        selected_preset = self.resolve_audio_preset(audio_preset or self.audio_preset)
        audio_filter = self.get_audio_filter_chain(selected_preset)
        cmd = [
            ffmpeg_bin,
            "-hide_banner",
            "-i",
            video_path,
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            "-af",
            audio_filter,
            audio_path,
            "-y",
        ]
        subprocess.run(cmd, check=True, capture_output=True, **self._subprocess_kwargs())
        print(f"音频提取完成：{audio_path}")
        return True

    def split_audio_chunks(self, audio_path, chunk_size_seconds, overlap_seconds):
        duration = self.get_audio_duration(audio_path)
        chunks = []
        chunk_size_seconds = max(30, min(600, int(chunk_size_seconds)))
        overlap_seconds = max(0.0, min(10.0, float(overlap_seconds)))
        start = 0.0
        index = 0
        while start < duration:
            end = min(duration, start + chunk_size_seconds)
            chunks.append(
                {
                    "index": index,
                    "start": max(0.0, start - overlap_seconds if index > 0 else start),
                    "end": end,
                    "content_start": start,
                    "content_end": end,
                }
            )
            index += 1
            start += chunk_size_seconds
        return chunks

    def format_time(self, seconds):
        td = datetime.timedelta(seconds=max(0.0, seconds))
        hours = int(td.total_seconds() // 3600)
        minutes = int((td.total_seconds() % 3600) // 60)
        sec = td.total_seconds() % 60
        milliseconds = int((sec % 1) * 1000)
        sec = int(sec)
        return f"{hours:02d}:{minutes:02d}:{sec:02d},{milliseconds:03d}"

    @staticmethod
    def _checkpoint_path(checkpoint_dir, chunk_index):
        return os.path.join(checkpoint_dir, f"chunk_{chunk_index:04d}.json")

    def _load_chunk_checkpoint(self, checkpoint_dir, chunk_index):
        file_path = self._checkpoint_path(checkpoint_dir, chunk_index)
        if not os.path.exists(file_path):
            return None
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if payload.get("checkpoint_version") != self.CHECKPOINT_VERSION:
                return None
            if payload.get("status") == "completed":
                return payload
        except Exception as err:
            print(f"警告：读取检查点失败 {file_path}：{err}")
        return None

    def _save_chunk_checkpoint(self, checkpoint_dir, chunk_index, payload):
        os.makedirs(checkpoint_dir, exist_ok=True)
        file_path = self._checkpoint_path(checkpoint_dir, chunk_index)
        payload["checkpoint_version"] = self.CHECKPOINT_VERSION
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def _segment_needs_second_pass(self, segment, previous_segment):
        confidence = segment.get("confidence", 1.0)
        text = segment.get("text", "").strip()
        overlap_like = previous_segment is not None and previous_segment["end"] > segment["start"]
        too_short = len(text) <= 2
        low_confidence = confidence < 0.45
        duration = max(0.1, float(segment.get("end", 0.0)) - float(segment.get("start", 0.0)))
        return overlap_like or too_short or low_confidence or self._looks_like_bad_asr(text, duration)

    @staticmethod
    def _looks_like_bad_asr(text, duration_seconds):
        cleaned = str(text or "").strip()
        if not cleaned:
            return True
        if re.search(r"(.)\1{5,}", cleaned):
            return True
        duration = max(0.1, float(duration_seconds))
        jp_chars = re.findall(r"[ぁ-んァ-ン一-龯々ー]", cleaned)
        jp_ratio = len(jp_chars) / max(1, len(cleaned))
        if duration >= 1.2 and jp_ratio < 0.18:
            return True
        if len(cleaned) > int(duration * 20):
            return True
        punct_ratio = len(re.findall(r"[^\w\sぁ-んァ-ン一-龯々ー。、「」！？!?\-]", cleaned)) / max(1, len(cleaned))
        return punct_ratio > 0.25

    @classmethod
    def _next_audio_preset(cls, preset):
        current = cls.resolve_audio_preset(preset)
        if current == cls.AUDIO_PRESET_STANDARD:
            return cls.AUDIO_PRESET_DENOISE
        if current == cls.AUDIO_PRESET_DENOISE:
            return cls.AUDIO_PRESET_AGGRESSIVE
        return cls.AUDIO_PRESET_AGGRESSIVE

    @staticmethod
    def _looks_like_single_block(segments, chunk_duration):
        if len(segments) != 1:
            return False
        segment = segments[0]
        text = segment.get("text", "").strip()
        if len(text) < 30:
            return False
        start = float(segment.get("start", 0.0))
        end = float(segment.get("end", 0.0))
        span = max(0.0, end - start)
        return span >= max(20.0, chunk_duration * 0.8)

    def _recover_segments_by_windows(self, chunk_audio_path, chunk_duration, quality_mode, audio_filter=None):
        window_seconds = 30.0 if quality_mode != "accurate" else 20.0
        overlap_seconds = 2.0
        recovered = []
        start = 0.0
        while start < chunk_duration:
            end = min(chunk_duration, start + window_seconds)
            region_segments = self.asr_engine.transcribe_region(
                chunk_audio_path,
                start,
                end,
                quality_mode=quality_mode,
                audio_filter=audio_filter,
            )
            for segment in region_segments:
                text = segment.get("text", "").strip()
                if not text:
                    continue
                if recovered:
                    prev = recovered[-1]
                    same_text = prev["text"] == text
                    near = abs(float(segment["start"]) - float(prev["end"])) <= 0.6
                    if same_text and near:
                        prev["end"] = max(float(prev["end"]), float(segment["end"]))
                        prev["confidence"] = max(prev.get("confidence", 1.0), segment.get("confidence", 1.0))
                        continue
                recovered.append(
                    {
                        "start": max(0.0, float(segment["start"])),
                        "end": min(chunk_duration, max(float(segment["start"]) + 0.1, float(segment["end"]))),
                        "text": text,
                        "confidence": float(segment.get("confidence", 1.0)),
                    }
                )
            start += max(1.0, window_seconds - overlap_seconds)
        return recovered

    def _merge_boundary_segments(self, segments):
        if not segments:
            return []
        ordered = sorted(segments, key=lambda x: (x["start"], x["end"]))
        merged = [ordered[0]]
        for current in ordered[1:]:
            prev = merged[-1]
            same_text = prev["text"].strip() == current["text"].strip()
            near_boundary = current["start"] - prev["end"] <= 1.0
            if same_text and near_boundary:
                prev["end"] = max(prev["end"], current["end"])
                prev["confidence"] = max(prev.get("confidence", 1.0), current.get("confidence", 1.0))
            elif current["start"] < prev["end"] and same_text:
                prev["end"] = max(prev["end"], current["end"])
            else:
                merged.append(current)
        return merged

    @staticmethod
    def _split_text_units(text, max_chars=40):
        if not text:
            return []
        normalized = re.sub(r"\s+", " ", str(text)).strip()
        if not normalized:
            return []

        sentence_parts = [p.strip() for p in re.split(r"(?<=[。！？!?])\s*", normalized) if p.strip()]
        units = []
        for part in sentence_parts:
            if len(part) <= max_chars:
                units.append(part)
                continue
            cursor = 0
            while cursor < len(part):
                units.append(part[cursor: cursor + max_chars].strip())
                cursor += max_chars
        return [u for u in units if u]

    def _expand_long_segment(self, segment):
        text = segment.get("text", "").strip()
        start = float(segment.get("start", 0.0))
        end = float(segment.get("end", start + 0.1))
        duration = max(0.1, end - start)

        # Split only when the segment is clearly too long.
        should_split = duration >= 12.0 or len(text) >= 70
        if not should_split:
            return [segment]

        units = self._split_text_units(text)
        if len(units) <= 1:
            return [segment]

        weights = [max(1, len(unit)) for unit in units]
        total_weight = float(sum(weights))
        cursor = start
        expanded = []
        for idx, unit in enumerate(units):
            if idx == len(units) - 1:
                unit_end = end
            else:
                unit_end = cursor + (duration * (weights[idx] / total_weight))
            expanded.append(
                {
                    "start": max(start, cursor),
                    "end": max(cursor + 0.1, min(end, unit_end)),
                    "text": unit,
                    "confidence": float(segment.get("confidence", 1.0)),
                    "quality_flagged": bool(segment.get("quality_flagged", False)),
                }
            )
            cursor = expanded[-1]["end"]
        return expanded

    def _merge_short_context_segments(self, segments):
        if not segments:
            return []

        ordered = sorted(segments, key=lambda x: (x["start"], x["end"]))
        merged = [ordered[0].copy()]

        for current in ordered[1:]:
            prev = merged[-1]
            prev_text = prev.get("text", "").strip()
            curr_text = current.get("text", "").strip()
            if not prev_text:
                merged[-1] = current.copy()
                continue
            if not curr_text:
                continue

            gap = float(current["start"]) - float(prev["end"])
            prev_duration = float(prev["end"]) - float(prev["start"])
            combined_duration = float(current["end"]) - float(prev["start"])
            combined_text = f"{prev_text}{curr_text}"
            prev_short = len(prev_text) < 16 or prev_duration < 1.6
            prev_incomplete = not re.search(r"[。！？!?]$", prev_text)
            curr_tiny = len(curr_text) < 10
            can_merge = (
                gap <= 0.45
                and len(combined_text) <= 52
                and combined_duration <= 8.0
                and (prev_short or prev_incomplete or curr_tiny)
            )
            if can_merge:
                prev["text"] = combined_text
                prev["end"] = max(float(prev["end"]), float(current["end"]))
                prev["confidence"] = max(prev.get("confidence", 1.0), current.get("confidence", 1.0))
                prev["quality_flagged"] = bool(prev.get("quality_flagged", False) or current.get("quality_flagged", False))
            else:
                merged.append(current.copy())

        return merged

    def process_video(
        self,
        video_path,
        output_dir=None,
        chunk_size_seconds=120,
        overlap_seconds=1.5,
        quality_mode=None,
        glossary_path=None,
        asr_terms_path=None,
        audio_preset=None,
    ):
        video_path = str(Path(video_path).resolve())
        if not video_path.lower().endswith((".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv")):
            print(f"错误：{video_path} 不是支持的视频格式")
            return False

        if output_dir is None:
            output_dir = os.path.dirname(video_path)
        output_dir = str(Path(output_dir).resolve())
        os.makedirs(output_dir, exist_ok=True)

        active_quality = quality_mode if quality_mode in {"fast", "accurate"} else self.quality_mode
        active_audio_preset = self.resolve_audio_preset(audio_preset or self.audio_preset)
        active_audio_filter = self.get_audio_filter_chain(active_audio_preset)
        active_chunk_size = int(chunk_size_seconds)
        active_overlap = float(overlap_seconds)
        if glossary_path and glossary_path != self.glossary_path:
            self.glossary_path = glossary_path
            self.glossary = self.load_glossary(glossary_path)
        if asr_terms_path and asr_terms_path != self.asr_terms_path:
            self.asr_terms_path = asr_terms_path
            self.asr_terms, self.asr_corrections = self.load_asr_terms(asr_terms_path)
            self.asr_engine.asr_terms = list(self.asr_terms)
            self.asr_engine.asr_corrections = dict(self.asr_corrections)
            self.asr_engine.qwen_prompt = self.asr_engine._build_qwen_prompt(self.asr_terms)
        if active_quality == "accurate" and active_chunk_size >= 120:
            active_chunk_size = 60
            print("精确模式已自动调整分块时长为 60 秒，以提升复杂噪声音频识别稳定性。")
        if active_quality == "accurate" and active_overlap < 2.0:
            active_overlap = 2.0

        video_name = os.path.splitext(os.path.basename(video_path))[0]
        audio_path = os.path.join(output_dir, f"{video_name}_{uuid.uuid4().hex}_temp.wav")
        subtitle_path = os.path.join(output_dir, f"{video_name}_bilingual.srt")
        checkpoint_dir = os.path.join(output_dir, f"{video_name}_checkpoints")

        print(
            f"分块：{active_chunk_size}s，重叠：{active_overlap}s，质量：{active_quality}，"
            f"音频预设：{active_audio_preset}"
        )
        print(f"检查点目录：{checkpoint_dir}")

        all_segments = []
        try:
            self._emit_progress(0.0, "正在提取音频...")
            self.extract_audio(video_path, audio_path, audio_preset=active_audio_preset)
            chunks = self.split_audio_chunks(audio_path, active_chunk_size, active_overlap)
            total_chunks = max(1, len(chunks))
            for chunk in chunks:
                existing = self._load_chunk_checkpoint(checkpoint_dir, chunk["index"])
                if existing:
                    print(f"继续处理：分块 {chunk['index']} 已完成")
                    all_segments.extend(existing.get("segments", []))
                    for source, target in existing.get("translation_memory", {}).items():
                        self.translation_engine.memory[source] = target
                    progress = ((chunk["index"] + 1) / total_chunks) * 95.0
                    self._emit_progress(progress, f"已续跑分块 {chunk['index'] + 1}/{total_chunks}")
                    continue

                print(
                    f"正在处理分块 {chunk['index'] + 1}/{len(chunks)} "
                    f"({chunk['start']:.2f}s - {chunk['end']:.2f}s)"
                )
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_chunk:
                    chunk_audio_path = temp_chunk.name

                try:
                    self._emit_progress(
                        (chunk["index"] / total_chunks) * 95.0,
                        f"正在处理分块 {chunk['index'] + 1}/{total_chunks}",
                    )
                    self.extract_audio_span(audio_path, chunk_audio_path, chunk["start"], chunk["end"])
                    local_segments = self.asr_engine.transcribe(chunk_audio_path, quality_mode=active_quality)
                    chunk_duration = max(0.1, chunk["end"] - chunk["start"])
                    if self._looks_like_single_block(local_segments, chunk_duration):
                        print("检测到 ASR 时间戳较粗糙，使用短窗口重试该分块...")
                        recovered = self._recover_segments_by_windows(
                            chunk_audio_path,
                            chunk_duration=chunk_duration,
                            quality_mode=active_quality,
                            audio_filter=active_audio_filter,
                        )
                        if recovered:
                            local_segments = recovered

                    chunk_segments = []
                    flagged = []
                    prev_segment = None
                    for segment in local_segments:
                        global_start = segment["start"] + chunk["start"]
                        global_end = segment["end"] + chunk["start"]
                        if global_end <= chunk["content_start"] or global_start >= chunk["content_end"]:
                            prev_segment = segment
                            continue
                        clipped_start = max(chunk["content_start"], global_start)
                        clipped_end = min(chunk["content_end"], max(global_start + 0.1, global_end))
                        normalized = {
                            "start": clipped_start,
                            "end": clipped_end,
                            "text": segment.get("text", "").strip(),
                            "confidence": float(segment.get("confidence", 1.0)),
                            "quality_flagged": False,
                        }
                        if not normalized["text"]:
                            prev_segment = segment
                            continue
                        expanded_segments = self._expand_long_segment(normalized)
                        for expanded in expanded_segments:
                            if self._segment_needs_second_pass(expanded, prev_segment):
                                expanded["quality_flagged"] = True
                                flagged.append(expanded)
                            chunk_segments.append(expanded)
                            prev_segment = expanded

                    if flagged:
                        for flagged_segment in flagged:
                            local_start = max(0.0, flagged_segment["start"] - chunk["start"])
                            local_end = max(local_start + 0.1, flagged_segment["end"] - chunk["start"])
                            refined = self.asr_engine.transcribe_region(
                                chunk_audio_path,
                                local_start,
                                local_end,
                                quality_mode="accurate",
                                audio_filter=self.get_audio_filter_chain(
                                    self._next_audio_preset(active_audio_preset)
                                ),
                            )
                            if refined:
                                best = refined[0]
                                flagged_segment["text"] = best.get("text", flagged_segment["text"]).strip()
                                flagged_segment["confidence"] = max(
                                    flagged_segment["confidence"],
                                    best.get("confidence", flagged_segment["confidence"]),
                                )

                    chunk_segments = self._merge_short_context_segments(chunk_segments)

                    for segment in chunk_segments:
                        segment["zh_text"] = self.translation_engine.translate(segment["text"], glossary=self.glossary)

                    merged_chunk_segments = self._merge_boundary_segments(chunk_segments)
                    checkpoint_payload = {
                        "status": "completed",
                        "chunk_index": chunk["index"],
                        "chunk_meta": chunk,
                        "segments": merged_chunk_segments,
                        "translation_memory": self.translation_engine.memory,
                    }
                    self._save_chunk_checkpoint(checkpoint_dir, chunk["index"], checkpoint_payload)
                    all_segments.extend(merged_chunk_segments)
                    progress = ((chunk["index"] + 1) / total_chunks) * 95.0
                    self._emit_progress(progress, f"已处理分块 {chunk['index'] + 1}/{total_chunks}")
                except Exception as chunk_error:
                    self._save_chunk_checkpoint(
                        checkpoint_dir,
                        chunk["index"],
                        {"status": "failed", "chunk_index": chunk["index"], "error": str(chunk_error)},
                    )
                    print(f"分块 {chunk['index']} 失败：{chunk_error}")
                    continue
                finally:
                    if os.path.exists(chunk_audio_path):
                        os.remove(chunk_audio_path)

            merged_segments = self._merge_boundary_segments(all_segments)
            self._emit_progress(97.0, "正在写入字幕文件...")
            self.generate_bilingual_srt(merged_segments, subtitle_path)
            self._emit_progress(100.0, "字幕生成完成")
            print(f"处理完成。字幕文件：{subtitle_path}")
            return subtitle_path
        except Exception as e:
            print(f"处理过程中发生错误：{e}")
            return False
        finally:
            if os.path.exists(audio_path):
                os.remove(audio_path)

    def generate_bilingual_srt(self, segments, output_path):
        print("正在生成双语字幕...")
        cleaned = [s for s in segments if s.get("text", "").strip()]
        cleaned.sort(key=lambda x: x["start"])
        with open(output_path, "w", encoding="utf-8") as f:
            for index, segment in enumerate(cleaned, start=1):
                f.write(f"{index}\n")
                f.write(f"{self.format_time(segment['start'])} --> {self.format_time(segment['end'])}\n")
                f.write(f"{segment['text'].strip()}\n{segment.get('zh_text', '').strip()}\n\n")
                if index % 20 == 0:
                    print(f"字幕写入进度：{index}/{len(cleaned)}")
