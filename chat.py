import os
from dotenv import load_dotenv
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Settings,
)
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone

# Voice imports
import sounddevice as sd
import soundfile as sf
from transformers import pipeline


load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

Settings.llm = OpenAILike(
    model="meta-llama/llama-3.3-70b-instruct",
    api_base="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    is_chat_model=True,
)

pc = Pinecone(
    api_key=PINECONE_API_KEY
)

pinecone_index = pc.Index("simplyrag")

vector_store = PineconeVectorStore(
    pinecone_index=pinecone_index
)

storage_context = StorageContext.from_defaults(
    vector_store=vector_store
)

index = VectorStoreIndex.from_vector_store(
    vector_store,
    storage_context=storage_context,
)

chat_engine = index.as_chat_engine(
    chat_mode="context",
    system_prompt=(
        "You are a helpful assistant that answers questions "
        "using the provided documents. "
        "If the answer cannot be found in the documents, "
        "say that you don't know rather than making something up."
    )
)

print("Loading speech recognition model...")

whisper = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-base",
)

print("Speech recognition model loaded!")

def record_audio():
    sample_rate = 16000

    print("\n🎙 Recording...")
    print("Press Enter to stop recording.")

    audio_chunks = []

    def callback(indata, frames, time, status):
        if status:
            print(status)

        audio_chunks.append(indata.copy())

    with sd.InputStream(
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        callback=callback,
    ):
        input()

    if not audio_chunks:
        return None

    import numpy as np

    audio = np.concatenate(audio_chunks, axis=0)
    audio = audio.flatten()

    # Remove leading/trailing silence
    threshold = 0.01

    non_silent = np.where(np.abs(audio) > threshold)[0]

    if len(non_silent) == 0:
        return None

    start = non_silent[0]
    end = non_silent[-1]

    padding = int(0.2 * sample_rate)

    start = max(0, start - padding)
    end = min(len(audio), end + padding)

    audio = audio[start:end]

    return audio, sample_rate

def voice_input():
    while True:

        result = record_audio()

        if result is None:
            print("No audio recorded.")
            continue

        audio, sample_rate = result

        print("Transcribing...")

        transcription = whisper(
            {
                "raw": audio,
                "sampling_rate": sample_rate,
            },
            generate_kwargs={
                "language": "en",
                "task": "transcribe",
            },
        )

        text = transcription["text"].strip()

        print(f"\nYou said: {text}")

        print("\n[Enter] send   [r] retry   [x] cancel")

        choice = input("> ").strip().lower()

        if choice == "r":
            continue

        if choice == "x":
            return None

        # Empty input means send
        return text


print("\nRAG chatbot ready!")
print("Type 'exit' to quit.")
print("Type 'v' for voice input.\n")


while True:
    question = input("You: ").strip()
    if question.lower() == "exit":
        break

    if question.lower() == "v":

        question = voice_input()

        if not question:
            print("Voice input cancelled.\n")
            continue

        print()

    response = chat_engine.chat(question)

    print("\nBot:", response)
    print()