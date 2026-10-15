# Synthetic test question

`question.wav` is a 24 kHz mono PCM test utterance generated with MiniCPM-o 4.5 and its bundled reference voice during this project's integration tests. It asks, in Chinese, what the main color of the input picture is. It is not a recording of a user's conversation.

`python3 scripts/setup.py fixtures` copies it to the ignored runtime directory without replacing an existing question. The browser demo uses this file together with a synthetic color image. Model weights and reference voice assets remain in their upstream repositories; see [third-party notices](../../THIRD_PARTY_NOTICES.md).
