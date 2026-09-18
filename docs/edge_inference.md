# Edge inference: plan and measurements

## Where the time goes

Thesis measurement on one NVIDIA T4 (Table 4.8): the network takes **34 ms** per
16-frame clip, the whole pipeline **433 ms**. The network is under 10% of the cost;
video decoding, person detection and filtering are the rest. Optimise in that order:

1. **Decode less.** A decision uses 16 frames spread over a 5 s window, i.e. about
   3 frames per second of a 30 fps stream. Decode only those frames (skip non-keyframe
   decoding, or a hardware decoder that outputs at reduced rate and resolution).
2. **Detect people less often.** The crop is one static box per window. The builder runs
   YOLOv8n on 12 frames; 2–4 frames, or reusing the previous window's box while it
   still covers the detections, gives nearly the same box. Run YOLOv8n at 320 px, as
   TensorRT/OpenVINO INT8.
3. **Filter only what is used.** Apply the bilateral filter and CLAHE to the 16 sampled
   frames, not all 64, and on the crop only. The bilateral filter is the slowest step on
   a CPU; whether it can be dropped needs an ablation (rebuild the data without it).
4. **Batch cameras.** Throughput grows with batch size, so run one forward pass per
   window for all cameras together.
5. **Decide over windows.** Slide the 5 s window with a 1–2 s stride and raise an alarm
   when k of the last n windows are above a threshold tuned for recall on validation.

## The network

`ma-x3d export RUN --int8` writes `model_fp32.onnx` (and `model_int8.onnx`) with:

- wide kernels fused into plain 5×5 depthwise convolutions;
- the motion map and input normalisation inside the graph, so the input is the raw
  [0, 1] clip `[B, 3, 16, 224, 224]` and there is no extra preprocessing step;
- dynamic batch size.

It checks the ONNX output against PyTorch, times it with ONNX Runtime on the CPU, and
evaluates accuracy on the test split for FP32 and INT8. BatchNorm is folded into the
convolutions by the runtime.

Options for the network, roughly in order of effort:

| Option | Where | Expected effect | Accuracy risk |
| --- | --- | --- | --- |
| FP16 | TensorRT (Jetson), OpenVINO (Intel GPU) | ~2× vs FP32 | very low |
| INT8 post-training | TensorRT, OpenVINO, ONNX Runtime | ~2× vs FP16 where 3D INT8 kernels exist | check: depthwise 3D + SE + swish are sensitive |
| INT8 quantisation-aware training | same | recovers PTQ loss | low, costs a short fine-tune |
| Fewer frames (T=8 or 12) | retrain | ~linear in T | E1 row gives the trend |
| Smaller input (160×160) | retrain; head pool must become adaptive | ~2× | unknown |
| Distil into X3D-S/XS | retrain | 2–4× | unknown |

Targets:

- **NVIDIA Jetson (Orin Nano/NX):** ONNX → TensorRT (`trtexec --onnx=model_fp32.onnx
  --fp16`), INT8 with a calibration cache from training clips. Check that 3D depthwise
  convolutions run in INT8 and do not fall back to FP16.
- **Intel CPU / iGPU:** OpenVINO (`ovc model_fp32.onnx`), then NNCF post-training
  quantisation.
- **Plain x86/ARM CPU:** ONNX Runtime, numbers below.

## Measured

Fused MA-X3D, one 16-frame clip (3.015M parameters, 5.46 GFLOPs). The accuracy
column is from the 2-epoch smoke checkpoint and only shows the *change* between
FP32 and INT8; re-run `ma-x3d export` on a trained model.

| Backend | Precision | Latency per clip | Test acc. (smoke model) |
| --- | --- | --- | --- |
| PyTorch, 1 MI250 GCD, batch 1 | FP32 | 20.3 ms | – |
| PyTorch, 1 MI250 GCD, batch 16 | FP16 | 6.4 ms (156 clips/s) | – |
| ONNX Runtime, CPU (EPYC 7413, 16 threads) | FP32 | 123 ms | 81.0% (PyTorch: 81.0%) |
| ONNX Runtime, CPU, static INT8 (QDQ) | INT8 | 280 ms | 75.0% |

The ONNX graph matches PyTorch (max logit difference 1.5e-6). Static INT8 in ONNX
Runtime is **slower** than FP32 here, because its CPU kernels do not run 3D
convolutions in INT8 and add quantise/dequantise steps, and it lost 6 points. INT8
is therefore only worth trying on a runtime with INT8 3D-convolution kernels
(TensorRT), with quantisation-aware training if post-training quantisation loses
accuracy. FP16 is the safe first step on a GPU.

For comparison, the thesis measured 33.8 ms per clip on an NVIDIA T4.
