# 1. 패키지 import
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset
import librosa

# 2. 모델과 포로세서 로드
# load model and processor
processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
model.config.forced_decoder_ids = None

# # 3. 데이터셋 로드
# ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
# sample = ds[0]["audio"]

wav_file = "The-Ivory-Coast-inte.wav"
audio_array, sampling_rate = librosa.load(wav_file, sr=16000)
sample = {
    "array": audio_array,
    "sampling_rate": sampling_rate
}

#  4. 전처리
input_features = processor(sample["array"], sampling_rate=sample["sampling_rate"], return_tensors="pt").input_features 

# 5. 모델 추론
# generate token ids
predicted_ids = model.generate(input_features)
# decode token ids to text
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=False)

transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

print(transcription)