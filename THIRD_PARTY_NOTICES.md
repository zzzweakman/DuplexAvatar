# Third-party components and assets

DuplexAvatar's integration code is licensed under [Apache-2.0](LICENSE).

DuplexAvatar connects separately installed projects. This repository does not distribute their model weights or copy their source trees. The exact tested source/model revisions are in [deploy/sources.json](deploy/sources.json).

| Component / asset | Source and terms |
| --- | --- |
| MiniCPM-o 4.5 model, custom model code and original speech | [Model card and license](https://huggingface.co/openbmb/MiniCPM-o-4_5) |
| MiniCPM backend, protocol and reference voice | [MiniCPM-o-Demo](https://github.com/OpenBMB/MiniCPM-o-Demo) |
| SoulX-FlashHead inference code and example avatar | [Repository / Apache-2.0 license](https://github.com/Soul-AILab/SoulX-FlashHead/blob/main/LICENSE) |
| SoulX-FlashHead Lite and VAE weights | [Model card and terms](https://huggingface.co/Soul-AILab/SoulX-FlashHead-1_3B) |
| wav2vec2-base-960h | [Model card and license](https://huggingface.co/facebook/wav2vec2-base-960h) |
| ASR service implementation | [Omni-Interaction-Agent / Apache-2.0](https://github.com/Omni-Interaction-Gander/Omni-Interaction-Agent/blob/main/LICENSE) |
| faster-whisper and converted Whisper weights | [Code / MIT](https://github.com/SYSTRAN/faster-whisper/blob/master/LICENSE), [model card](https://huggingface.co/Systran/faster-whisper-large-v3) |
| FlashAttention | [Dao-AILab / BSD-3-Clause](https://github.com/Dao-AILab/flash-attention/blob/main/LICENSE) |
| Noto Sans SC subset font | [Bundled SIL Open Font License](frontend/fonts/noto-sans-sc-LICENSE.txt) |

`assets/fixtures/question.wav` is synthetic test speech generated during this project's MiniCPM integration; provenance is described [beside the fixture](assets/fixtures/README.md). The default avatar image and reference voice are read from the separately installed upstream repositories, not included here.

Upstream projects and dependencies retain their own licenses. Installing the integration does not replace the terms for models, voices, images, fonts or media codecs. Refer to the linked upstream notices for the versions you deploy.
