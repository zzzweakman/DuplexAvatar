# 测试与诊断

所有命令从仓库根目录执行；使用与启动器一致的 `DUPLEX_AVATAR_CONFIG`。默认端口 8443。完整模型测试一次只运行一个；GPU 0–3 配置仅通过配置逻辑测试，本项目实测 GPU 为 4–7。

## CPU 回归

```bash
../MiniCPM-o-Demo/.venv/base/bin/python -m unittest discover -s tests -v
../SoulX-FlashHead/.venv/bin/python -m unittest discover -s tests -p test_video_transport.py -v
node --test tests/*.mjs
```

第二条验证 PyAV 编码后独立解码的帧数、顺序和时戳；MiniCPM 环境没有 PyAV，会明确跳过这项。JavaScript 检查 5 FPS 采样、有界输入、弱网流控、倾听动作及打断过渡，不加载 GPU 模型。

## 服务与真实模型

```bash
python3 scripts/setup.py doctor
python3 scripts/manage.py start
curl -k https://127.0.0.1:8443/health
../MiniCPM-o-Demo/.venv/base/bin/python scripts/smoke.py
../MiniCPM-o-Demo/.venv/base/bin/python scripts/smoke.py --audio-input
../MiniCPM-o-Demo/.venv/base/bin/python scripts/smoke.py --audio-input --audio-output
```

`smoke.py` 使用合成语音和色块画面；检查真实视觉 token、回复、非静音原声、交付 PCM 逐采样一致、数字人运动，以及理解 / 视频服务区间重叠。仅音频输入应明确表示没有图像。模型回复具有随机性，文字断言失败时也需要查看完整回复，区分模型回答与传输故障。

## 浏览器回归

Node.js 20+。测试工具不参与生产推理；前端本身不需要 npm 构建。

```bash
npm install
npx playwright install chromium
npm run test:browser
```

精简 Linux 环境还需要 Playwright 所需的系统浏览器库，可参考 [Playwright 安装文档](https://playwright.dev/docs/browsers)。测试支持 `PLAYWRIGHT_PACKAGE` 和 `CHROMIUM_PATH` 覆盖已有工具路径。

测试启动无头 Chromium，使用模拟设备驱动真实服务，覆盖摄像头、麦克风、H.264、媒体 / 控制连接、重复打断、倾听过渡、仅音频输入、纯音频输出、字幕与时间戳、响度、句尾画面和重连。输出位于 `runtime/browser-*/`。

### 5 FPS 与长会话视觉验证

额外准备一段 WAV，清晰朗读一句：**“请描述一下你现在看到的画面。”** 不要在同一段录音里连续问两个问题；模型可能回答第一个问题时被第二个问题打断。

```bash
export VISION_QUESTION_WAV=/absolute/path/to/describe.wav
npm run test:vision
node scripts/vision_browser_smoke.cjs --audio-output
```

数字人测试至少 90 秒：同一会话切换蓝色、绿色、示例头像，再经过上下文淘汰后切回蓝色；检查回答、实际采样时戳、批内帧差异、摄像头暂停 / 恢复和最终轨道释放。纯音频输出的视觉测试至少 20 秒。头像描述断言使用上游默认 `girl.png`，自定义头像需要相应调整断言。

测试结果不是所有真人场景的识别率；不同光照、噪声、语言、网络和 GPU 会影响结果。

## 显存复测

终端 A 先启动只读采样器，再在终端 B 启动服务并依次运行带视觉 / 仅音频测试：

```bash
python3 scripts/profile_memory.py --gpus 4 5 6 7 --interval 0.2 \
  --output runtime/memory-my-run
```

`--output` 必须是新目录。四卡 0–3 机器将 `--gpus` 改为 `0 1 2 3`。采样器使用 `nvidia-smi`；实际间隔受命令执行时间影响，时间戳保存在 `samples.jsonl`。完成后在终端 A 按 Ctrl-C，写出 `summary.json`。

它记录各 GPU 的整卡使用量和本项目进程使用量，并分别统计：

- 每张卡的观察峰值。
- 同一次采样中四卡使用量相加的最大值。
- 每张卡独立峰值之和，不能当成同时发生的峰值。

`nvidia-smi` 包含 CUDA 上下文和保留的分配，不等于 `torch.cuda.max_memory_allocated()`。采样可能漏掉短暂峰值；这不是显存理论上限或最低硬件保证。采样器不会启动或停止模型，最终执行：

```bash
python3 scripts/manage.py stop
python3 scripts/manage.py status
```

## 编译基准与实验

运行独立编译基准前停止正式服务，避免重复加载占用 GPU：

```bash
python3 scripts/manage.py stop
../MiniCPM-o-Demo/.venv/base/bin/python scripts/benchmark_compile.py brain
../SoulX-FlashHead/.venv/bin/python scripts/benchmark_compile.py avatar
```

脑服务基准固定为倾听决策；它不能代表 TTS 或完整端到端首音延迟。视频基准测量一块生成，排除网络 / 播放。`benchmark_parallel_avatar.py` 是独立实验入口，未作为正式双卡视频工作器启用。不要对并行语音服务使用 `turn_based` 协议；分卡部署使用 `full_duplex`。

原始诊断文件可能含测试音频、视频和会话标识，因此 `runtime/` 不进入 Git。对外分享结果时使用去除媒体载荷与会话信息的汇总。
