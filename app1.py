import asyncio
import signal

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
from vocode.streaming.transcriber.deepgram_transcriber import DeepgramTranscriber
from vocode.streaming.synthesizer.play_ht_synthesizer import PlayHtSynthesizer
from vocode.streaming.synthesizer.play_ht_synthesizer import PlayHtSynthesizerConfig
# Set environment variables
vocode.setenv(
    OPEN_APT_KEY="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",
    DEEPGRAM_API_KEY="ab1769cd27df5dc51f453c7ade74b77da7d452f9",
    PLAY_HT_API_KEY="4eb0bf3bfc3a4f02bf6e52578b1c1bdb",
    PLAY_HT_USER_ID="lpsjDH7W95VguTGS2LInizhdabQ2",
)


async def main():
    microphone_input, speaker_output = create_streaming_microphone_input_and_speaker_output(
        use_blocking_speaker_output=False
    )

    # Create PlayHtSynthesizerConfig
    synthesizer_config = PlayHtSynthesizerConfig(
        voice_id="s3://voice-cloning-zero-shot/1f44b3e7-22ea-4c2e-87d0-b4d9c8f1d47d/sophia/manifest.json",
        speed=None,  # You can set a value if needed
        preset=None,  # You can set a value if needed
    )

    # Create StreamingConversation
    conversation = StreamingConversation(
        output_device=speaker_output,
        transcriber=DeepgramTranscriber(
            DeepgramTranscriberConfig.from_input_device(
                microphone_input,
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
        chunk = await microphone_input.get_audio()
        conversation.receive_audio(chunk)


if __name__ == "__main__":
    asyncio.run(main(), debug=True)
