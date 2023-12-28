import asyncio
import queue
import signal
from typing import Optional

import vocode
from vocode.streaming.streaming_conversation import StreamingConversation
from vocode.helpers import create_streaming_microphone_input_and_speaker_output
from vocode.streaming.models.transcriber import (
    DeepgramTranscriberConfig,
    PunctuationEndpointingConfig,
)
from vocode.streaming.agent.chat_gpt_agent import ChatGPTAgent
from vocode.streaming.models.agent import ChatGPTAgentConfig
from vocode.streaming.models.message import BaseMessage
from vocode.streaming.synthesizer.play_ht_synthesizer import PlayHtSynthesizer
from vocode.streaming.synthesizer.play_ht_synthesizer import PlayHtSynthesizerConfig
from vocode.streaming.transcriber.deepgram_transcriber import DeepgramTranscriber


# get_audio
def get_audio(self) -> Optional[bytes]:
    try:
        return self.queue.get_nowait()
    except queue.Empty:
        return None


# Set environment variables

async def main():
    MicrophoneInput, speaker_output = create_streaming_microphone_input_and_speaker_output(
        use_blocking_speaker_output=False
    )

    # Create PlayHtSynthesizerConfig
    synthesizer_config = PlayHtSynthesizerConfig(
        voice_id="s3://voice-cloning-zero-shot/1f44b3e7-22ea-4c2e-87d0-b4d9c8f1d47d/sophia/manifest.json",
        speed=None,  # You can set a value if needed
        preset=None,  # You can set a value if needed
        sampling_rate=8000,  # mulaw set 8k sapling rate
        audio_encoding="mulaw",  # Add the required field

    )

    # Create StreamingConversation
    conversation = StreamingConversation(
        output_device=speaker_output,
        transcriber=DeepgramTranscriber(
            DeepgramTranscriberConfig.from_input_device(
                MicrophoneInput,
                endpointing_config=PunctuationEndpointingConfig(),
            )
        ),
        agent=ChatGPTAgent(
            ChatGPTAgentConfig(
                initial_message=BaseMessage(text="Hello!"),
                prompt_preamble="Have a pleasant conversation about life",
            )
        ),
        synthesizer=PlayHtSynthesizer(synthesizer_config),
    )

    await conversation.start()
    print("Conversation started, press Ctrl+C to end")
    while conversation.is_active():
        chunk = MicrophoneInput.get_audio()  # Adjust buffer size as needed
        conversation.receive_audio(chunk)


if __name__ == "__main__":
    asyncio.run(main())
