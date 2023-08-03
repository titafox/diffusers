<p align="center">
    <br>
    <img src="https://raw.githubusercontent.com/huggingface/diffusers/main/docs/source/en/imgs/diffusers_library.jpg" width="400"/>
    <br>
<p>
<p align="center">
    <a href="https://github.com/huggingface/diffusers/blob/main/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/github/license/huggingface/datasets.svg?color=blue">
    </a>
    <a href="https://github.com/huggingface/diffusers/releases">
        <img alt="GitHub release" src="https://img.shields.io/github/release/huggingface/diffusers.svg">
    </a>
    <a href="CODE_OF_CONDUCT.md">
        <img alt="Contributor Covenant" src="https://img.shields.io/badge/Contributor%20Covenant-2.0-4baaaa.svg">
    </a>
</p>


🤗 Diffusers是用于生成图像、音频甚至分子的3D结构的最先进预训练扩散模型的首选库。无论您是寻找简单的推理解决方案还是训练自己的扩散模型，🤗 Diffusers都是一个支持两者的模块化工具箱。
我们的库专注[易用性优先于性能](https://huggingface.co/docs/diffusers/conceptual/philosophy#usability-over-performance), [简约而不及简单](https://huggingface.co/docs/diffusers/conceptual/philosophy#simple-over-easy), 以及 [可定制性优于抽象性](https://huggingface.co/docs/diffusers/conceptual/philosophy#tweakable-contributorfriendly-over-abstraction).

🤗 Diffusers提供三个核心组件：

* 最先进的[扩散流水线](https://huggingface.co/docs/diffusers/api/pipelines/overview)，只需几行代码即可进行推理。
* 可互换的噪声[调度器](https://huggingface.co/docs/diffusers/api/schedulers/overview)，用于不同的扩散速度和输出质量。
* 预训练的[模型](https://huggingface.co/docs/diffusers/api/models)，可用作构建模块，并与调度器结合，创建您自己的端到端扩散系统。

## 安装

我们建议从PyPi或Conda在虚拟环境中安装🤗 Diffusers。有关安装[PyTorch](https://pytorch.org/get-started/locally/)和[Flax](https://flax.readthedocs.io/en/latest/#installation)的更多详细信息，请参阅它们的官方文档。


### PyTorch

使用 `pip` (official package):

```bash
pip install --upgrade diffusers[torch]
```

使用 `conda` (maintained by the community):

```sh
conda install -c conda-forge diffusers
```

### Flax

使用 `pip` (official package):

```bash
pip install --upgrade diffusers[flax]
```

### 苹果芯片（M1/M2）支持。

请参阅[在苹果芯片（M1/M2）上使用稳定扩散的指南](https://huggingface.co/docs/diffusers/optimization/mps)。

## 快速上手

使用🤗 Diffusers生成输出非常简单。要从文本生成图像，请使用`from_pretrained`方法加载任何预训练的扩散模型（浏览[Hub](https://huggingface.co/models?library=diffusers&sort=downloads)查看4000多个检查点）：

```python
from diffusers import DiffusionPipeline
import torch

pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipeline.to("cuda")
pipeline("An image of a squirrel in Picasso style").images[0]
```

您还可以深入了解模型和调度器工具箱，构建自己的扩散系统：

```python
from diffusers import DDPMScheduler, UNet2DModel
from PIL import Image
import torch
import numpy as np

scheduler = DDPMScheduler.from_pretrained("google/ddpm-cat-256")
model = UNet2DModel.from_pretrained("google/ddpm-cat-256").to("cuda")
scheduler.set_timesteps(50)

sample_size = model.config.sample_size
noise = torch.randn((1, 3, sample_size, sample_size)).to("cuda")
input = noise

for t in scheduler.timesteps:
    with torch.no_grad():
        noisy_residual = model(input, t).sample
        prev_noisy_sample = scheduler.step(noisy_residual, t, input).prev_sample
        input = prev_noisy_sample

image = (input / 2 + 0.5).clamp(0, 1)
image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
image = Image.fromarray((image * 255).round().astype("uint8"))
image
```

立即查看[快速入门](https://huggingface.co/docs/diffusers/quicktour)，开始您的扩散之旅吧！

## 如何使用文档
| **文档**                                                                           | **我能学到什么？**                                                                                             |
| ------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------- |
| [教程](https://huggingface.co/docs/diffusers/tutorials/tutorial_overview)                   | 一个基础的速成课程，教您如何使用库的最重要功能，比如使用模型和调度器构建自己的扩散系统，以及如何训练自己的扩散模型。 |
| [加载](https://huggingface.co/docs/diffusers/using-diffusers/loading_overview)              | 指南介绍了如何加载和配置库的所有组件（流水线、模型和调度器），以及如何使用不同的调度器。                             |
| [用于推理的流水线](https://huggingface.co/docs/diffusers/using-diffusers/pipeline_overview) | 指南介绍了如何在不同的推理任务中使用流水线，批量生成、控制生成的输出和随机性，以及如何为库贡献一个流水线。           |
| [优化](https://huggingface.co/docs/diffusers/optimization/opt_overview)                     | 指南介绍了如何优化您的扩散模型，使其运行更快，占用更少的内存。                                                       |
| [训练](https://huggingface.co/docs/diffusers/training/overview)                             | 指南介绍了如何使用不同的训练技术，为不同任务训练扩散模型。                                                           |

## 贡献

我们 ❤️ 来自开源社区的贡献！
如果您想对该库做出贡献，请查阅我们的[贡献指南](https://github.com/huggingface/diffusers/blob/main/CONTRIBUTING.md)。
您可以找到您感兴趣的[问题](https://github.com/huggingface/diffusers/issues)来为该库做出贡献。

* 查看[Good first issues](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22)，以获取一般的贡献机会
* 查看[New model/pipeline](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22New+pipeline%2Fmodel%22)，贡献令人兴奋的新扩散模型/扩散流水线
* 查看[New scheduler](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22New+scheduler%22)

此外，在我们的公开Discord频道里打个招呼👋 <a href="https://discord.gg/G7tWnz98XR"><img alt="Join us on Discord" src="https://img.shields.io/discord/823813159592001537?color=5865F2&logo=discord&logoColor=white"></a>。我们讨论有关扩散模型的最热门趋势，相互帮助贡献、个人项目，或者只是闲聊喝杯咖啡☕。



## 热门任务和流水线（Popular Tasks & Pipelines）

<table>
  <tr>
    <th>任务</th>
    <th>流水线</th>
    <th>🤗 Hub</th>
  </tr>
  <tr style="border-top: 2px solid black">
    <td>无条件图像生成</td>
    <td><a href="https://huggingface.co/docs/diffusers/api/pipelines/ddpm"> DDPM </a></td>
    <td><a href="https://huggingface.co/google/ddpm-ema-church-256"> google/ddpm-ema-church-256 </a></td>
  </tr>
  <tr style="border-top: 2px solid black">
    <td>文本到图像</td>
    <td><a href="https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/text2img">Stable Diffusion 文本到图像</a></td>
      <td><a href="https://huggingface.co/runwayml/stable-diffusion-v1-5"> runwayml/stable-diffusion-v1-5 </a></td>
  </tr>
  <tr>
    <td>文本到图像</td>
    <td><a href="https://huggingface.co/docs/diffusers/api/pipelines/unclip">unclip</a></td>
      <td><a href="https://huggingface.co/kakaobrain/karlo-v1-alpha"> kakaobrain/karlo-v1-alpha </a></td>
  </tr>
  <tr>
    <td>文本到图像</td>
    <td><a href="https://huggingface.co/docs/diffusers/api/pipelines/if">DeepFloyd IF</a></td>
      <td><a href="https://huggingface.co/DeepFloyd/IF-I-XL-v1.0"> DeepFloyd/IF-I-XL-v1.0 </a></td>
  </tr>
  <tr>
    <td>文本到图像</td>
    <td><a href="https://huggingface.co/docs/diffusers/api/pipelines/kandinsky">Kandinsky</a></td>
      <td><a href="https://huggingface.co/kandinsky-community/kandinsky-2-2-decoder"> kandinsky-community/kandinsky-2-2-decoder </a></td>
  </tr>
  <tr style="border-top: 2px solid black">
    <td>文本指导图像到图像</td>
    <td><a href="https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/controlnet">Controlnet</a></td>
      <td><a href="https://huggingface.co/lllyasviel/sd-controlnet-canny"> lllyasviel/sd-controlnet-canny </a></td>
  </tr>
  <tr>
    <td>文本指导图像到图像</td>
    <td><a href="https://huggingface.co/docs/diffusers/api/pipelines/pix2pix">Instruct Pix2Pix</a></td>
      <td><a href="https://huggingface.co/timbrooks/instruct-pix2pix"> timbrooks/instruct-pix2pix </a></td>
  </tr>
  <tr>
    <td>文本指导图像到图像</td>
    <td><a href="https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/img2img">Stable Diffusion 图像到图像</a></td>
      <td><a href="https://huggingface.co/runwayml/stable-diffusion-v1-5"> runwayml/stable-diffusion-v1-5 </a></td>
  </tr>
  <tr style="border-top: 2px solid black">
    <td>文本指导图像修复</td>
    <td><a href="https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/inpaint">Stable Diffusion 图像修复</a></td>
      <td><a href="https://huggingface.co/runwayml/stable-diffusion-inpainting"> runwayml/stable-diffusion-inpainting </a></td>
  </tr>
  <tr style="border-top: 2px solid black">
    <td>图像变异</td>
    <td><a href="https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/image_variation">Stable Diffusion 图像变异</a></td>
      <td><a href="https://huggingface.co/lambdalabs/sd-image-variations-diffusers"> lambdalabs/sd-image-variations-diffusers </a></td>
  </tr>
  <tr style="border-top: 2px solid black">
    <td>超分辨率</td>
    <td><a href="https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/upscale">Stable Diffusion 超分辨率</a></td>
      <td><a href="https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler"> stabilityai/stable-diffusion-x4-upscaler </a></td>
  </tr>
  <tr>
    <td>超分辨率</td>
    <td><a href="https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/latent_upscale">Stable Diffusion 潜在超分辨率</a></td>
      <td><a href="https://huggingface.co/stabilityai/sd-x2-latent-upscaler"> stabilityai/sd-x2-latent-upscaler </a></td>
  </tr>
</table>

## 使用🧨 Diffusers 的热门库

- https://github.com/microsoft/TaskMatrix
- https://github.com/invoke-ai/InvokeAI
- https://github.com/apple/ml-stable-diffusion
- https://github.com/Sanster/lama-cleaner
- https://github.com/IDEA-Research/Grounded-Segment-Anything
- https://github.com/ashawkey/stable-dreamfusion
- https://github.com/deep-floyd/IF
- https://github.com/bentoml/BentoML
- https://github.com/bmaltais/kohya_ss
- +3000 other amazing GitHub repositories 💪

## 鸣谢

本库是许多不同作者之前工作的具体化，没有他们的伟大研究和实现，我们的库是不可能实现的。我们特别感谢以下实现，在我们的开发过程中帮助了我们，没有这些实现，我们的API今天也不可能如此完善：

* @CompVis 的潜在扩散模型库，可在[此处](https://github.com/CompVis/latent-diffusion)找到
* @hojonathanho 原始DDPM实现，可在[此处](https://github.com/hojonathanho/diffusion)找到，以及由@pesser将其翻译成PyTorch的非常有用的实现，可在[此处](https://github.com/pesser/pytorch_diffusion)找到
* @ermongroup 的DDIM实现，可在[此处](https://github.com/ermongroup/ddim)找到
* @yang-song 的Score-VE和Score-VP实现，可在[此处](https://github.com/yang-song/score_sde_pytorch)找到

我们还要感谢 @heejkoo 提供的扩散模型论文、代码和资源的非常有用的概述，可在[此处](https://github.com/heejkoo/Awesome-Diffusion-Models)找到，以及 @crowsonkb 和 @rromb 进行有益的讨论和见解。

## 引用方式

```bibtex
@misc{von-platen-etal-2022-diffusers,
  author = {Patrick von Platen and Suraj Patil and Anton Lozhkov and Pedro Cuenca and Nathan Lambert and Kashif Rasul and Mishig Davaadorj and Thomas Wolf},
  title = {Diffusers: State-of-the-art diffusion models},
  year = {2022},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/huggingface/diffusers}}
}
```
