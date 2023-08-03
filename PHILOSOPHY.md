# 理念

🧨 Diffusers 提供了**最先进的**预训练的渐进式模型,跨多种模式。
它的目的是充当 **模块化工具箱** ,用于推理和训练。

我们的目标是构建一个经得起时间检验的库,因此我们非常认真地对待 API 设计。

简而言之,Diffusers 旨在成为 PyTorch 的自然扩展。因此,我们的大多数设计选择都基于 [PyTorch 的设计原则](https://pytorch.org/docs/stable/community/design.html#pytorch-design-philosophy)。让我们概述一下最重要的原则:

## 可用性高于性能

* 虽然 Diffusers 具有许多内置的性能增强功能(参见[内存和速度](https://huggingface.co/docs/diffusers/optimization/fp16)),但模型总是以最高精度和最低优化加载。因此,默认情况下,扩散管道总是在 CPU 上以 float32 精度实例化,除非用户另有定义。这可确保跨不同平台和加速器的可用性,意味着不需要复杂的安装才能运行该库。
* Diffusers 旨在成为一个**轻量级**包,因此几乎没有必需的依赖项,但有许多可以提高性能的软依赖项(如 `accelerate`、`safetensors`、`onnx`等)。我们努力保持该库尽可能轻量,以便它可以作为对其他包的依赖项而不需要太多考虑。
* Diffusers 更喜欢简单的、自解释的代码,而不是简洁的、神奇的代码。这意味着通常不希望看到诸如 lambda 函数和高级 PyTorch 操作符等速记代码语法。

## 简单高于容易

正如 PyTorch 所述, **显式优于隐式** ,并且 **简单优于复杂** 。这种设计哲学反映在库的多个部分:

* 我们遵循 PyTorch 的 API,例如 [`DiffusionPipeline.to`](https://huggingface.co/docs/diffusers/main/en/api/diffusion_pipeline#diffusers.DiffusionPipeline.to) 来让用户处理设备管理。
* 提出简明的错误消息优先于悄悄纠正错误的输入。 Diffusers 旨在教导用户,而不是使库尽可能容易使用。
* 公开复杂的模型与调度器逻辑,而不是在内部神奇地处理。 调度器/采样器与扩散模型分开,相互依赖性最小。这迫使用户编写展开的去噪循环。但是,这种分离允许更容易的调试,并让用户对调整去噪过程或切换扩散模型或调度器有更多控制。
* 分别训练的扩散管道组件,例如文本编码器、unet 和变分自动编码器,每个都有自己的模型类。这迫使用户处理不同模型组件之间的交互以及序列化格式将模型组件分隔到不同的文件中。但是,这允许更容易的调试和自定义。 Dreambooth 或文本反转训练由于 diffusers 能够分离扩散管道的单个组件而变得非常简单。

## 可调整性和友好的贡献性高于抽象化

对于该库的大部分,Diffusers采用了 [Transformers 库](https://github.com/huggingface/transformers)的一个重要设计原则,即优先复制粘贴代码而不是匆忙抽象。这个设计原则非常见仁见智,与流行的设计原则如 [Don't repeat yourself (DRY)](https://en.wikipedia.org/wiki/Don%27t_repeat_yourself) 形成鲜明对比。
简而言之,就像 Transformers 对建模文件所做的那样,diffusers 更喜欢保持极低的抽象级别和非常独立的代码用于流水线和调度器。
函数、长代码块甚至类可以跨多个文件复制,乍一看这可能看起来是一个糟糕的、混乱的设计选择,会使库难以维护。
**然而** ,这种设计对 Transformers 已经证明非常成功,并且对社区驱动的开源机器学习库很有意义,因为:

* 机器学习是一个发展极快的领域,其中范式、模型架构和算法都在迅速变化,因此很难定义长期有效的代码抽象。
* 机器学习从业者喜欢能够快速调整现有代码进行构思和研究,因此他们更喜欢独立的代码而不是包含许多抽象的代码。
* 开源库依赖社区贡献,因此必须构建一个易于贡献的库。代码越抽象,依赖性越多,可读性越差,贡献难度越大。过于抽象的代码会阻碍潜在新贡献者的贡献动力。如果向库贡献代码不会破坏其他基础代码,那么不仅对潜在新贡献者更具吸引力,而且可以更轻松地同时审查和贡献多个部分。

在 Hugging Face,我们称这个设计为 **单文件策略** ,这意味着某个类的几乎所有代码都应该编写在一个独立的文件中。要了解更多关于这种设计哲学的信息,您可以看看 [这篇博文](https://huggingface.co/blog/transformers-design-philosophy)。

在 diffusers 中,我们对流水线和调度器遵循这种哲学,但对扩散模型只是部分遵循。我们不完全遵循扩散模型的设计的原因是几乎所有扩散管道,例如 [DDPM](https://huggingface.co/docs/diffusers/v0.12.0/en/api/pipelines/ddpm)、[Stable Diffusion](https://huggingface.co/docs/diffusers/v0.12.0/en/api/pipelines/stable_diffusion/overview#stable-diffusion-pipelines)、[UnCLIP(Dalle-2)]([https://huggingface](https://huggingface/)) .co/docs/diffusers/v0.12.0/en/api/pipelines/unclip#overview) 和 [Imagen](https://imagen.research.google/) 都依赖于相同的扩散模型,[UNet](https://huggingface.co/docs/diffusers/api/models#diffusers.UNet2DConditionModel)。

很好,现在您应该已经大致了解为什么 🧨 Diffusers 以这种方式设计了 🤗。我们试图在整个库中一致地应用这些设计原则。不过,哲学仍有一些小的例外或一些不幸的设计选择。如果您对设计有任何反馈,我们很乐意直接在 [GitHub](https://github.com/huggingface/diffusers/issues/new?assignees=&labels=&template=feedback.md&title=) 上收到您的反馈。

## 详细的设计理念

现在,让我们深入一点,了解设计理念的细节。Diffusers 从本质上由三个主要类组成,[流水线](https://github.com/huggingface/diffusers/tree/main/src/diffusers/pipelines)、[模型](https://github.com/huggingface/diffusers/tree/main/src/diffusers/models)和 [调度器](https://github.com/huggingface/diffusers/tree/main/src/diffusers/schedulers)。让我们逐一查看每个类的更详细的设计决策。

### 流水线

流水线旨在易于使用(因此不完全遵循 [*简单高于容易*](https://claude.ai/chat/3d47d9a3-dd61-4f81-83c4-094245cc484a#%E7%AE%80%E5%8D%95%E9%AB%98%E4%BA%8E%E5%AE%B9%E6%98%93) 100%)),功能不完整,并且应该轻松地被视为如何使用 [模型](https://claude.ai/chat/3d47d9a3-dd61-4f81-83c4-094245cc484a#%E6%A8%A1%E5%9E%8B) 和 [调度器](https://claude.ai/chat/3d47d9a3-dd61-4f81-83c4-094245cc484a#%E8%B0%83%E5%BA%A6%E5%99%A8)进行推理的示例。

遵循以下设计原则:

* 流水线遵循单文件策略。所有流水线都可以在 src/diffusers/pipelines 下的各个目录中找到。一个流水线文件夹对应一个扩散论文/项目/版本。多个流水线文件可以聚集在一个流水线文件夹中,就像在 [`src/diffusers/pipelines/stable-diffusion`](https://github.com/huggingface/diffusers/tree/main/src/diffusers/pipelines/stable_diffusion) 中所做的那样。如果流水线共享类似的功能,可以使用 [#Copied from机制](https://github.com/huggingface/diffusers/blob/125d783076e5bd9785beb05367a2d2566843a271/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_img2img.py#L251)。
* 所有流水线都继承自 [`DiffusionPipeline`]
* 每个流水线由不同的模型和调度器组件组成,这些组件在 [`model_index.json` 文件](https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/model_index.json)中进行了文档化,可以通过流水线属性访问,并可以通过 [`DiffusionPipeline.components`](https://huggingface.co/docs/diffusers/main/en/api/diffusion_pipeline#diffusers.DiffusionPipeline.components) 函数在流水线之间共享。
* 每个流水线都应该可以通过 [`DiffusionPipeline.from_pretrained`](https://huggingface.co/docs/diffusers/main/en/api/diffusion_pipeline#diffusers.DiffusionPipeline.from_pretrained) 函数加载。
* 流水线**只**用于推理。
* 流水线应该非常可读,不言自明,并且易于调整。
* 流水线应该设计成能够互相构建,并且易于集成到更高级的 API 中。
* 流水线**不是**完整的用户接口。对于未来的完整用户接口,应该更倾向于查看 [InvokeAI](https://github.com/invoke-ai/InvokeAI)、[Diffuzers](https://github.com/abhishekkrthakur/diffuzers) 和 [lama-cleaner](https://github.com/Sanster/lama-cleaner)
* 每个流水线应该只有一种通过 `__call__` 方法运行它的方式。 `__call__` 参数的命名应该在所有流水线中共享。
* 流水线应该根据它们要解决的任务命名。
* 在几乎所有情况下,新的扩散流水线应该在一个新的流水线文件夹/文件中实现。

### 模型

模型被设计为可配置的工具箱,是 [PyTorch 的 Module 类](https://pytorch.org/docs/stable/generated/torch.nn.Module.html) 的自然扩展。它们只在一定程度上遵循 **单文件策略** 。

遵循以下设计原则:

* 模型对应于 **一种模型架构类型** 。例如,[`UNet2DConditionModel`] 类用于所有期望 2D 图像输入并根据某些上下文进行条件化的 UNet 变体。
* 所有模型都可以在 [`src/diffusers/models`](https://github.com/huggingface/diffusers/tree/main/src/diffusers/models) 中找到,每个模型架构都应在其各自的文件中定义,例如 [`unet_2d_condition.py`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/unet_2d_condition.py)、[`transformer_2d.py`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformer_2d.py)等。
* 模型**不遵循**单文件策略,应利用更小的模型构建块,如 [`attention.py`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention.py)、[`resnet.py`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/resnet.py)、[`embeddings.py`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/embeddings.py) 等。 **注意** :这与 Transformers 的建模文件形成了鲜明对比,显示模型实际上并没有真正遵循单文件策略。
* 模型旨在暴露复杂性,就像 PyTorch 的 module 一样,并给出清晰的错误消息。
* 所有模型都继承自 `ModelMixin` 和 `ConfigMixin`。
* 当不要求重大代码更改、保持向后兼容性并提供显著的内存或计算收益时,模型可以进行性能优化。
* 模型默认情况下应具有最高精度和最低性能设置。
* 要将新的模型检查点集成到 Diffusers 中,其一般架构可以归类为已经存在的架构,应适应现有的模型架构以使其与新检查点一起工作。只有当模型架构从根本上不同时,才应创建新文件。
* 模型应该设计得易于扩展以适应未来的更改。这可以通过限制公共函数参数、配置参数以及“预见”未来更改来实现,例如,添加字符串“...类型”参数通常比添加布尔“is_... _type”参数更好,因为它们可以轻松扩展到新的未来类型。应该对现有架构进行最少量的更改,以使新模型检查点正常工作。
* 模型设计需要在保持代码可读性、简洁性和支持许多模型检查点之间达到艰难的权衡。对于大部分建模代码,应根据新模型检查点调整类,而对一些例外情况,如 [UNet 块](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/unet_2d_blocks.py) 和 [Attention 处理器](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py),添加新类以确保长期保持代码简洁和可读性是首选的。

### 调度器

调度器负责指导推理的去噪过程以及定义训练的噪声时间表。它们被设计为具有可加载配置文件和强烈遵循**单文件策略**的单独类。

遵循以下设计原则:

* 所有调度器都可以在 [`src/diffusers/schedulers`](https://github.com/huggingface/diffusers/tree/main/src/diffusers/schedulers) 中找到。
* 调度器**不允许**从大型实用程序文件导入,并且应保持非常独立。
* 一个调度器 python 文件对应一个调度器算法(可能在论文中定义)。
* 如果调度器共享类似的功能,我们可以使用 `#Copied from` 机制。
* 所有调度器都继承自 `SchedulerMixin` 和 `ConfigMixin`。
* 调度器可以通过 [`ConfigMixin.from_config`](https://huggingface.co/docs/diffusers/main/en/api/configuration#diffusers.ConfigMixin.from_config) 方法轻松切换,如[这里](https://claude.ai/chat/using-diffusers/schedulers.md)所详细说明的。
* 每个调度器都必须有一个 `set_num_inference_steps` 和一个 `step` 函数。在每次去噪过程之前,也就是在调用 `step(...)` 之前,都必须调用 `set_num_inference_steps(...)`。
* 每个调度器通过一个 `timesteps` 属性公开要“循环”的时间步长,这是一个模型将被调用的时间步长数组。
* `step(...)` 函数获取预测的模型输出和“当前”样本(x_t),并返回略微更去噪的“前一个”样本(x_t-1)。
* 考虑到扩散调度器的复杂性,`step` 函数不会暴露所有复杂性,可以是一个有点“黑盒”的函数。
* 在几乎所有情况下,新的调度器都应该在一个新的调度文件中实现。

总而言之,Diffusers 在可用性、简单性和贡献者友好性方面做出了设计选择,这有助于构建一个可持续发展的开源机器学习库。我们试图在整个库中一致地应用这些设计原则。如果您对设计有任何反馈,我们非常欢迎您通过 GitHub 直接告知我们。

