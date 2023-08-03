<!---
Copyright 2023 The HuggingFace Team. All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# 🧨 Diffusers 示例

Diffusers 示例是演示如何使用 `diffusers` 库进行各种训练或微调的脚本集合。

**注意** :如果你正在寻找 **官方** 示例来展示如何使用 `diffusers` 进行推理,请查看 [src/diffusers/pipelines](https://github.com/huggingface/diffusers/tree/main/src/diffusers/pipelines)。

我们的示例旨在具有 **自包含性** 、 **易于调整** 、**初学者友好性**和 **单一目的** 。
更具体地说,这意味着:

* **自包含** :示例脚本应该只依赖于可以在 `requirements.txt` 文件中找到的“可通过 pip 安装”的 Python 包。示例脚本**不应该**依赖任何本地文件。这意味着你可以简单地下载一个示例脚本,例如 [train_unconditional.py](https://github.com/huggingface/diffusers/blob/main/examples/unconditional_image_generation/train_unconditional.py),安装所需的依赖项,例如 [requirements.txt](https://github.com/huggingface/diffusers/blob/main/examples/unconditional_image_generation/requirements.txt),然后执行示例脚本。
* **易于调整** :虽然我们努力展示尽可能多的用例,但示例脚本只是例子。预计它们在你的特定问题上不会开箱即用,你需要更改几行代码将它们调整到你的需要。为了帮助你,大多数示例完全公开了数据的预处理和训练循环,以便根据需要进行调整和编辑。
* **初学者友好** :我们的目标不是提供最新模型的最先进的训练脚本,而是可以用来更好地理解扩散模型及如何将它们与 `diffusers` 库一起使用的示例。如果我们认为某些最先进的方法对初学者来说太复杂,我们通常会有意地省略它们。
* **单一目的** :示例应该展示一个任务,只展示一个任务。即使从建模的角度来看,某些任务非常相似,例如图像超分辨率和图像修改往往使用相同的模型和训练方法,但我们希望示例仅展示一个任务,以使它们保持最大的可读性和易理解性。

我们提供涵盖扩散模型最流行任务的**官方**示例。
**官方**示例由 `diffusers` 的维护人员**积极地**维护,我们努力严格遵循上面定义的示例理念。
如果您认为应该存在另一个重要的示例,我们非常欢迎您提出 [Feature Request](https://github.com/huggingface/diffusers/issues/new?assignees=&labels=&template=feature_request.md&title=) 或直接提出 [Pull Request](https://github.com/huggingface/diffusers/compare)。

培训示例展示了如何对各种任务预训练或微调扩散模型。目前我们支持:

| Task | 🤗 Accelerate | 🤗 Datasets | Colab
|---|---|:---:|:---:|
| [**无条件图像生成**](./unconditional_image_generation) | ✅ | ✅ | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/training_example.ipynb)
| [**文本到图像微调**](./text_to_image) | ✅ | ✅ | 
| [*文本反转**](./textual_inversion) | ✅ | - | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/sd_textual_inversion_training.ipynb)
| [**Dreambooth**](./dreambooth) | ✅ | - | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/sd_dreambooth_training.ipynb)
| [**ControlNet**](./controlnet) | ✅ | ✅ | -
| [**InstructPix2Pix**](./instruct_pix2pix) | ✅ | ✅ | -
| [**基于强化学习的控制**](https://github.com/huggingface/diffusers/blob/main/examples/reinforcement_learning/run_diffusers_locomotion.py)                    | - | - | 即将推出                                                                                                                                                                                 |

## 社区

此外,我们还提供由社区添加和维护的**社区**示例。
社区示例可以包括*训练*示例或*推理*管道。
对于这样的示例,我们对上述理念更宽松,也无法保证为每个问题提供维护。
对社区有用但可能还不被认为流行或尚未遵循我们的理念的示例应放入 [community examples](https://github.com/huggingface/diffusers/tree/main/examples/community) 文件夹。community 文件夹因此包括训练示例和推理管道。
**注意** :社区示例可以是一个很好的[首次贡献](https://github.com/huggingface/diffusers/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22),向社区展示你喜欢如何使用 `diffusers` 🪄。

## 研究项目

我们还提供由社区维护的**研究项目**示例,如各自研究项目文件夹中所定义。这些示例很有用,并提供了补充官方示例的扩展功能。您可以参阅 [research_projects](https://github.com/huggingface/diffusers/tree/main/examples/research_projects) 以了解详细信息。

## 重要提示

为了确保你可以成功运行示例脚本的最新版本,你必须**从源代码安装该库**并安装一些特定于示例的要求。要执行此操作,请在一个新的虚拟环境中执行以下步骤:

```bash
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install .
```
然后进入你选择的示例文件夹并运行:

```bash
pip install -r requirements.txt
```
