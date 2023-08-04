# DreamBooth训练示例

[DreamBooth](https://arxiv.org/abs/2208.12242)是一种个性化文本到图像模型的方法,如stable diffusion,只需要极少量(3~5)的某个主题的图片。

`train_dreambooth.py` 脚本展示了如何实现训练过程并将其适配到stable diffusion。

## 在本地用PyTorch运行

### 安装依赖

在运行脚本之前,请确保安装库的训练依赖:

**重要**

为了确保你可以成功运行最新版本的示例脚本,我们强烈建议**从源代码安装**并保持安装更新,因为我们经常更新示例脚本并安装一些示例特定的要求。要做到这一点,请在一个新的虚拟环境中执行以下步骤:
```bash
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install -e .
```

Then cd in the example folder and run
然后cd 到example文件夹并run
```bash
pip install -r requirements.txt
```

并用以下命令初始化一个[🤗Accelerate]([https://github.com/huggingface/accelerate/)环境](https://github.com/huggingface/accelerate/)%E7%8E%AF%E5%A2%83):

```bash
accelerate config
```
或者什么问题不要回答，直接设置默认accelerate配置

```bash
accelerate config default
```

或者如果你的环境不支持交互式shell,例如notebook

```python
from accelerate.utils import write_basic_config
write_basic_config()
```

翻译:当运行`accelerate config`时,如果我们将torch编译模式设置为True,可以明显加速。

### 狗玩具示例

现在让我们获取数据集。对于这个示例,我们将使用一些狗的图片:[https://huggingface.co/datasets/diffusers/dog-example。](https://huggingface.co/datasets/diffusers/dog-example%E3%80%82)

首先让我们在本地下载它:

```python
from huggingface_hub import snapshot_download

local_dir = "./dog"
snapshot_download(
    "diffusers/dog-example",
    local_dir=local_dir, repo_type="dataset",
    ignore_patterns=".gitattributes",
)
```

然后使用以下命令启动训练:

**___注意:如果你使用的是 [stable-diffusion-2]([https://huggingface.co/stabilityai/stable-diffusion-2](https://huggingface.co/stabilityai/stable-diffusion-2)) 768x768 模型,请将 `resolution` 改为768。___**

```bash
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_DIR="dog"
export OUTPUT_DIR="path-to-save-model"

accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of sks dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=400 \
  --push_to_hub
```

#### --push_to_hub

--push_to_hub 是将训练好的DreamBooth模型推送到Hugging Face Hub来分享的选项。使用方法如下:

1. 在Hugging Face上注册账号,然后在网站的Settings中获取个人Access Token。
2. 在训练命令最后加上--push_to_hub参数,同时用--hub_token指定Access Token:

```bash
--hub_token="xxxxxxx"
````
3.  可选地用--hub_model_id指定模型在Hub上的名字,不指定会使用output_dir的名字:
```bash
--hub_model_id="titafox/test"
```
4. 训练结束后,脚本会自动创建Hub仓库,上传模型文件和生成示例图片。
5. 在Hub上刷新你的模型列表,就可以看到新上传的DreamBooth模型,可以直接使用。

### 使用先验保留损失进行训练

先验保留用于避免过拟合和语言漂移。参阅论文以了解更多信息。 对于先验保留,我们首先使用模型和一个类别提示生成图像,然后在训练时将这些图像与我们的数据一起使用。

根据论文,对于先验保留生成 `num_epochs * num_samples` 个图像是推荐的。对于大多数情况,200-300就足够了。`num_class_images` 标志设置使用类别提示生成的图像数量。您可以将现有图像放在`class_data_dir`中,训练脚本将生成任何额外的图像,以便在训练时`class_data_dir`中存在`num_class_images`。

```bash
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_DIR="dog"
export CLASS_DIR="path-to-class-images"
export OUTPUT_DIR="path-to-save-model"

accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$OUTPUT_DIR \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --instance_prompt="a photo of sks dog" \
  --class_prompt="a photo of dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=200 \
  --max_train_steps=800 \
  --push_to_hub
```


### 在16GB GPU上训练:

在梯度检查点和来自bitsandbytes的8位优化器的帮助下,可以在16GB GPU上运行train dreambooth。

要安装bitsandbytes请参考这个自述文件。

以下是在16GB GPU上训练的一个示例命令:

```bash
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_DIR="dog"
export CLASS_DIR="path-to-class-images"
export OUTPUT_DIR="path-to-save-model"

accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$OUTPUT_DIR \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --instance_prompt="a photo of sks dog" \
  --class_prompt="a photo of dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=2 --gradient_checkpointing \
  --use_8bit_adam \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=200 \
  --max_train_steps=800 \
  --push_to_hub
```
关键是使用 --mixed_precision="fp16" 和 --gradient_accumulation_steps=1 来减少GPU内存使用。

### 在12GB GPU上训练:

通过使用以下优化,可以在12GB GPU上运行dreambooth:

- [梯度检查点和8位优化器](#在16GB-GPU上训练)

- [xformers](#使用xformers训练)

- [设置grads为none](#设置grads为none)

以下是一个12GB GPU上的示例命令:

```bash
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_DIR="dog"
export CLASS_DIR="path-to-class-images"
export OUTPUT_DIR="path-to-save-model"

accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$OUTPUT_DIR \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --instance_prompt="a photo of sks dog" \
  --class_prompt="a photo of dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 --gradient_checkpointing \
  --use_8bit_adam \
  --enable_xformers_memory_efficient_attention \
  --set_grads_to_none \
  --learning_rate=2e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=200 \
  --max_train_steps=800 \
  --push_to_hub
```
关键是使用梯度累积、较低的学习率和xformers来减少GPU内存使用。

### 在 8 GB GPU 上训练:

通过使用 [DeepSpeed](https://www.deepspeed.ai/),可以将一些张量从 VRAM 卸载到 CPU 或 NVME,以便用更少的 VRAM 进行训练。

需要在 `accelerate config` 中启用 DeepSpeed。在配置过程中,对“您是否要使用 DeepSpeed?”问答“是”。使用 DeepSpeed 第2阶段,fp16 混合精度和将参数和优化器状态卸载到 cpu,可以在小于 8 GB VRAM 的条件下进行训练,代价是需要明显更多的 RAM(约 25 GB)。参见[文档](https://huggingface.co/docs/accelerate/usage_guides/deepspeed)以获取更多 DeepSpeed 配置选项。

将默认的 Adam 优化器更改为 DeepSpeed 的 Adam 特殊版本 `deepspeed.ops.adam.DeepSpeedCPUAdam` 可以明显加速,但启用它需要与 pytorch 相同版本的 CUDA 工具链。目前 8 比特优化器似乎与 DeepSpeed 不兼容。

以下是一个 DeepSpeed 示例命令:

```bash
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_DIR="dog"
export CLASS_DIR="path-to-class-images"
export OUTPUT_DIR="path-to-save-model"

accelerate launch --mixed_precision="fp16" train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$OUTPUT_DIR \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --instance_prompt="a photo of sks dog" \
  --class_prompt="a photo of dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --sample_batch_size=1 \
  --gradient_accumulation_steps=1 --gradient_checkpointing \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=200 \
  --max_train_steps=800 \
  --push_to_hub
```
关键是使用 DeepSpeed 和 fp16 混合精度来显著减少 VRAM 使用。

### class_data_dir和instance_data_dir

instance_data_dir和class_data_dir之间的主要区别是:

* instance_data_dir: 包含要训练的特定实例对象的图像,例如某个具体的人或者狗。这些图像会让模型学会生成该特定实例。
* class_data_dir: 包含同一类对象的通用图像,例如各种不同狗的图片。这些图像提供类的先验知识,避免模型过度拟合特定实例。

instance_data_dir是必需的,包含要训练的实例对象。这主要决定了模型能生成什么样的内容。

class_data_dir是可选的,包含更多同类别样本。这有助于模型学习整个类的特征分布,避免只记住特定实例的信息。

所以instance_data_dir是核心数据,class_data_dir是可选的辅助数据。former决定了特定生成内容,latter帮助模型学习类的先验知识。二者共同提高生成质量和多样性。

简单来说,instance_data_dir是“这个”,class_data_dir是“这类东西”。两者共同作用让模型既学会特定实例,又不忘记更广泛的先验知识。

#### instance_data_dir

instance_data_dir目录下应该包含用于DreamBooth训练的instance图像。

该目录的组织结构通常很简单,类似如下:

* INSTANCE_DIR
  * img1.png
  * img2.png
  * img3.png
  * ...
也就是说instance_data_dir直接放置instance对象(例如特定狗)的各种图片。

图像的数量和质量会直接影响训练效果。一般来说,数量越多效果越好,但即使只有几张图像也能工作。图像质量越高越好。

如果图像中包含复杂的背景,可以考虑先进行抠图来聚焦对象。

另外,如果目标是生成人像,由于需要避免生成不存在的人,实例图像不宜过少(一般需要10张以上)。

所以实例图像目录很简单,直接准备好instance对象的图片就可以,数量质量越高效果越好。这是DreamBooth训练最关键的数据。

#### class_data_dir
class_data_dir目录下应该包含一系列用于训练的类图像,用于提供类先验信息。

该目录的组织结构通常如下:

* class_data_dir
  * class1
    * img1.png
    * img2.png ...
  * class2
    * img1.png ...
  * ...

也就是说,class_data_dir下包含多个类别的子目录,每个子目录下是对应类别的图像。

图像的数量和质量会影响最终的模型效果。一般来说,每个类别至少需要100张以上的高质量图像,才能提供足够的先验知识。

如果class_data_dir中的图像不足,训练脚本在运行时会自动采样生成更多类图像以达到所需的数量。

另外,class_data_dir可以为空或者不提供,这时就不会使用类先验的损失,效果可能会略差一些。

所以在有限的数据条件下,提供class_data_dir可以帮助提升模型对类特征的学习。但即使没有也可以进行DreamBooth微调训练。


### 使用UNet微调文本编码器。

该脚本也允许细调`text_encoder`和`unet`。通过实验观察到,微调`text_encoder`可以获得更好的结果,特别是在人脸上。

传递`--train_text_encoder`参数给脚本以启用训练`text_encoder`。

___注意:训练文本编码器需要更多内存,使用此选项,训练将不适合16GB GPU。它至少需要24GB VRAM。___

```bash
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_DIR="dog"
export CLASS_DIR="path-to-class-images"
export OUTPUT_DIR="path-to-save-model"

accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --train_text_encoder \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$OUTPUT_DIR \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --instance_prompt="a photo of sks dog" \
  --class_prompt="a photo of dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --use_8bit_adam \
  --gradient_checkpointing \
  --learning_rate=2e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=200 \
  --max_train_steps=800 \
  --push_to_hub
```

### 将DreamBooth用于除Stable Diffusion之外的流程

[AltDiffusion流程](https://huggingface.co/docs/diffusers/api/pipelines/alt_diffusion)也支持dreambooth微调。过程与上述相同,您需要做的就是像这样替换`MODEL_NAME`:
```
export MODEL_NAME="CompVis/stable-diffusion-v1-4" --> export MODEL_NAME="BAAI/AltDiffusion-m9"
or
export MODEL_NAME="CompVis/stable-diffusion-v1-4" --> export MODEL_NAME="BAAI/AltDiffusion"
```

### 推理

一旦你使用上述命令训练了一个模型,你可以简单地使用`StableDiffusionPipeline`运行推理。确保在你的提示中包含`identifier`(例如上例中的sks)。

```python
from diffusers import StableDiffusionPipeline
import torch

model_id = "path-to-your-trained-model"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

prompt = "A photo of sks dog in a bucket"
image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]

image.save("dog-bucket.png")
```

### 推理训练检查点

如果你使用了--checkpointing_steps参数,也可以从训练过程中保存的检查点之一执行推理。请参阅文档以查看如何操作。

使用大语言模型的低秩适配(LoRA)进行训练
低秩大语言模型适配首先由Microsoft在[LoRA:大语言模型的低秩适配](https://arxiv.org/abs/2106.09685)中提出,作者是_Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen_

简而言之,LoRA允许通过向现有权重添加等级分解矩阵对来适配预训练模型,并且**只**训练新添加的权重。这有几个优点:

- 以前的预训练权重保持冻结,所以模型不太容易[灾难性遗忘]((https://www.pnas.org/doi/10.1073/pnas.1611835114))

- 等级分解矩阵的参数明显少于原始模型,这意味着训练好的LoRA权重很容易便携。

- LoRA注意力层允许通过scale参数控制模型适配新的训练图像的程度。

[cloneofsimo](https://github.com/cloneofsimo)是首次尝试在Stable Diffusion中进行LoRA训练的人

在流行的lora GitHub库中。

训练
让我们从一个简单的例子开始。我们将重用前一节中的狗示例。

首先,您需要按说明设置dreambooth训练示例安装部分。

接下来,让我们下载狗的数据集。从这里下载图像并保存到一个目录中。请确保在下面将INSTANCE_DIR设置为目录的名称。这将是我们的训练数据。

现在,您可以启动训练。这里我们将使用Stable Diffusion 1-5。

**___注意:如果您使用的是stable-diffusion-2 768x768 模型,请将resolution更改为768。___**

**___注意:通过在训练过程中定期生成示例图像来监控训练进度非常有用。 wandb是一个很好的解决方案,可以轻松地在训练过程中看到生成的图像。您需要做的就是在训练前运行pip install wandb,并传递--report_to="wandb"自动记录图像。___**


```bash
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export INSTANCE_DIR="dog"
export OUTPUT_DIR="path-to-save-model"
```

对于这个例子,我们想直接在Hub上存储训练好的LoRA嵌入,所以

我们需要登录并添加 --push_to_hub 参数。

```bash
huggingface-cli login
```

如果是在 colab 上你也可以这样做：

```bash
!mkdir -p ~/.huggingface
HUGGINGFACE_TOKEN = "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" #@param {type:"string"}
!echo -n "{HUGGINGFACE_TOKEN}" > ~/.huggingface/token
```

现在我们可以开始训练了

```bash
accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of sks dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --checkpointing_steps=100 \
  --learning_rate=1e-4 \
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=500 \
  --validation_prompt="A photo of sks dog in a bucket" \
  --validation_epochs=50 \
  --seed="0" \
  --push_to_hub
```

**___注意:在使用LoRA时,我们可以使用比普通梦之迹更高的学习率。这里我们使用_1e-4_,而不是通常的_2e-6_。___**

最终的LoRA嵌入权重已上传到[patrickvonplaten/lora_dreambooth_dog_example]([https://huggingface.co/patrickvonplaten/lora\\_dreambooth\\_dog\\_example)。](https://huggingface.co/patrickvonplaten/lora%5C%5C_dreambooth%5C%5C_dog%5C%5C_example)%E3%80%82) **___注意:[最终的权重]([https://huggingface.co/patrickvonplaten/lora/blob/main/pytorch\\_attn\\_procs.bin)只有3](https://huggingface.co/patrickvonplaten/lora/blob/main/pytorch%5C%5C_attn%5C%5C_procs.bin)%E5%8F%AA%E6%9C%893) MB大小,比原始模型小了几个数量级。**

训练结果总结[在这里](https://api.wandb.ai/report/patrickvonplaten/xm6cd5q5)。

您可以使用`Step`滑块来查看模型在训练过程中如何学习我们主题的特征。

可选地,我们也可以为文本编码器训练额外的LoRA层。为此,请指定上述的`--train_text_encoder`参数。如果您有兴趣了解我们如何

启用此支持,请查看此[PR](https://github.com/huggingface/diffusers/pull/2918)。

使用上述的默认超参数,训练似乎正在朝着正面方向发展。查看[这个面板](https://wandb.ai/sayakpaul/dreambooth-lora/reports/test-23-04-17-17-00-13---Vmlldzo0MDkwNjMy)。训练好的LoRA层可在[这里](https://huggingface.co/sayakpaul/dreambooth)获得。


### 推理

训练后,LoRA权重可以很容易地加载到原始管道中。首先,您需要加载原始管道:

```python
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
import torch

pipe = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.to("cuda")
```

接下来,我们可以使用[`load_attn_procs`函数](https://huggingface.co/docs/diffusers/api/loaders#diffusers.loaders.UNet2DConditionLoadersMixin.load_attn_procs)将适配器层加载到UNet中。

```python
pipe.unet.load_attn_procs("patrickvonplaten/lora_dreambooth_dog_example")
```

最后,我们可以在推理中运行该模型。

```python
image = pipe("A picture of a sks dog in a bucket", num_inference_steps=25).images[0]
```

如果你从Hub加载LoRA参数,并且Hub仓库有一个
`base_model` 标签(比如 [这个](https://huggingface.co/patrickvonplaten/lora_dreambooth_dog_example/blob/main/README.md?code=true#L4)),那么
你可以这样做:

```py 
from huggingface_hub.repocard import RepoCard

lora_model_id = "patrickvonplaten/lora_dreambooth_dog_example"
card = RepoCard.load(lora_model_id)
base_model_id = card.data.to_dict()["base_model"]

pipe = StableDiffusionPipeline.from_pretrained(base_model_id, torch_dtype=torch.float16)
...
```

如果你在训练期间使用了 --train_text_encoder,那么使用 pipe.load_lora_weights() 来加载LoRA权重。例如:

```python
from huggingface_hub.repocard import RepoCard
from diffusers import StableDiffusionPipeline
import torch 

lora_model_id = "sayakpaul/dreambooth-text-encoder-test"
card = RepoCard.load(lora_model_id)
base_model_id = card.data.to_dict()["base_model"]

pipe = StableDiffusionPipeline.from_pretrained(base_model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")
pipe.load_lora_weights(lora_model_id)
image = pipe("A picture of a sks dog in a bucket", num_inference_steps=25).images[0]
```

请注意,与 [`\UNet2DConditionLoadersMixin.load_attn_procs\`](https://huggingface.co/docs/diffusers/main/en/api/loaders#diffusers.loaders.UNet2DConditionLoadersMixin.load_attn_procs) 相比,[`\LoraLoaderMixin.load_lora_weights\`](https://huggingface.co/docs/diffusers/main/en/api/loaders#diffusers.loaders.LoraLoaderMixin.load_lora_weights)是加载LoRA参数的首选方法。这是因为`LoraLoaderMixin.load_lora_weights` 可以处理以下情况:

* 没有单独标识符的LoRA参数,用于UNet和文本编码器(比如 [`\patrickvonplaten/lora_dreambooth_dog_example\`](https://huggingface.co/patrickvonplaten/lora_dreambooth_dog_example))。所以,你可以这样做:

  ```py 
  pipe.load_lora_weights(lora_model_path)
  ```
  它将自动加载文本编码器和UNet的LoRA参数。

* 对于只训练了文本编码器或UNet的情况,它也可以正常工作。

所以总的来说,LoraLoaderMixin.load_lora_weights提供了一个标准的API来加载LoRA参数,无需担心它们的确切组织方式。

* 具有UNet和文本编码器单独标识符的LoRA参数,例如:  [`"sayakpaul/dreambooth"`](https://huggingface.co/sayakpaul/dreambooth).

## Training with Flax/JAX

For faster training on TPUs and GPUs you can leverage the flax training example. Follow the instructions above to get the model and dataset before running the script.

____Note: The flax example don't yet support features like gradient checkpoint, gradient accumulation etc, so to use flax for faster training we will need >30GB cards.___


Before running the scripts, make sure to install the library's training dependencies:

```bash
pip install -U -r requirements_flax.txt
```


### Training without prior preservation loss

```bash
export MODEL_NAME="duongna/stable-diffusion-v1-4-flax"
export INSTANCE_DIR="dog"
export OUTPUT_DIR="path-to-save-model"

python train_dreambooth_flax.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of sks dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --learning_rate=5e-6 \
  --max_train_steps=400
```


### Training with prior preservation loss

```bash
export MODEL_NAME="duongna/stable-diffusion-v1-4-flax"
export INSTANCE_DIR="dog"
export CLASS_DIR="path-to-class-images"
export OUTPUT_DIR="path-to-save-model"

python train_dreambooth_flax.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$OUTPUT_DIR \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --instance_prompt="a photo of sks dog" \
  --class_prompt="a photo of dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --learning_rate=5e-6 \
  --num_class_images=200 \
  --max_train_steps=800
```


### Fine-tune text encoder with the UNet.

```bash
export MODEL_NAME="duongna/stable-diffusion-v1-4-flax"
export INSTANCE_DIR="dog"
export CLASS_DIR="path-to-class-images"
export OUTPUT_DIR="path-to-save-model"

python train_dreambooth_flax.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --train_text_encoder \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$OUTPUT_DIR \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --instance_prompt="a photo of sks dog" \
  --class_prompt="a photo of dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --learning_rate=2e-6 \
  --num_class_images=200 \
  --max_train_steps=800
```

### Training with xformers:
You can enable memory efficient attention by [installing xFormers](https://github.com/facebookresearch/xformers#installing-xformers) and padding the `--enable_xformers_memory_efficient_attention` argument to the script. This is not available with the Flax/JAX implementation.

You can also use Dreambooth to train the specialized in-painting model. See [the script in the research folder for details](https://github.com/huggingface/diffusers/tree/main/examples/research_projects/dreambooth_inpaint).

### Set grads to none

To save even more memory, pass the `--set_grads_to_none` argument to the script. This will set grads to None instead of zero. However, be aware that it changes certain behaviors, so if you start experiencing any problems, remove this argument.

More info: https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html

### Experimental results
You can refer to [this blog post](https://huggingface.co/blog/dreambooth) that discusses some of DreamBooth experiments in detail. Specifically, it recommends a set of DreamBooth-specific tips and tricks that we have found to work well for a variety of subjects. 

## IF

You can use the lora and full dreambooth scripts to train the text to image [IF model](https://huggingface.co/DeepFloyd/IF-I-XL-v1.0) and the stage II upscaler 
[IF model](https://huggingface.co/DeepFloyd/IF-II-L-v1.0).

Note that IF has a predicted variance, and our finetuning scripts only train the models predicted error, so for finetuned IF models we switch to a fixed
variance schedule. The full finetuning scripts will update the scheduler config for the full saved model. However, when loading saved LoRA weights, you
must also update the pipeline's scheduler config.

```py
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained("DeepFloyd/IF-I-XL-v1.0")

pipe.load_lora_weights("<lora weights path>")

# Update scheduler config to fixed variance schedule
pipe.scheduler = pipe.scheduler.__class__.from_config(pipe.scheduler.config, variance_type="fixed_small")
```

Additionally, a few alternative cli flags are needed for IF.

`--resolution=64`: IF is a pixel space diffusion model. In order to operate on un-compressed pixels, the input images are of a much smaller resolution. 

`--pre_compute_text_embeddings`: IF uses [T5](https://huggingface.co/docs/transformers/model_doc/t5) for its text encoder. In order to save GPU memory, we pre compute all text embeddings and then de-allocate
T5.

`--tokenizer_max_length=77`: T5 has a longer default text length, but the default IF encoding procedure uses a smaller number.

`--text_encoder_use_attention_mask`: T5 passes the attention mask to the text encoder.

### Tips and Tricks
We find LoRA to be sufficient for finetuning the stage I model as the low resolution of the model makes representing finegrained detail hard regardless.

For common and/or not-visually complex object concepts, you can get away with not-finetuning the upscaler. Just be sure to adjust the prompt passed to the
upscaler to remove the new token from the instance prompt. I.e. if your stage I prompt is "a sks dog", use "a dog" for your stage II prompt.

For finegrained detail like faces that aren't present in the original training set, we find that full finetuning of the stage II upscaler is better than 
LoRA finetuning stage II.

For finegrained detail like faces, we find that lower learning rates along with larger batch sizes work best.

For stage II, we find that lower learning rates are also needed.

We found experimentally that the DDPM scheduler with the default larger number of denoising steps to sometimes work better than the DPM Solver scheduler
used in the training scripts.

### Stage II additional validation images

The stage II validation requires images to upscale, we can download a downsized version of the training set:

```py
from huggingface_hub import snapshot_download

local_dir = "./dog_downsized"
snapshot_download(
    "diffusers/dog-example-downsized",
    local_dir=local_dir,
    repo_type="dataset",
    ignore_patterns=".gitattributes",
)
```

### IF stage I LoRA Dreambooth
This training configuration requires ~28 GB VRAM.

```sh
export MODEL_NAME="DeepFloyd/IF-I-XL-v1.0"
export INSTANCE_DIR="dog"
export OUTPUT_DIR="dreambooth_dog_lora"

accelerate launch train_dreambooth_lora.py \
  --report_to wandb \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a sks dog" \
  --resolution=64 \
  --train_batch_size=4 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --scale_lr \
  --max_train_steps=1200 \
  --validation_prompt="a sks dog" \
  --validation_epochs=25 \
  --checkpointing_steps=100 \
  --pre_compute_text_embeddings \
  --tokenizer_max_length=77 \
  --text_encoder_use_attention_mask
```

### IF stage II LoRA Dreambooth

`--validation_images`: These images are upscaled during validation steps.

`--class_labels_conditioning=timesteps`: Pass additional conditioning to the UNet needed for stage II.

`--learning_rate=1e-6`: Lower learning rate than stage I.

`--resolution=256`: The upscaler expects higher resolution inputs

```sh
export MODEL_NAME="DeepFloyd/IF-II-L-v1.0"
export INSTANCE_DIR="dog"
export OUTPUT_DIR="dreambooth_dog_upscale"
export VALIDATION_IMAGES="dog_downsized/image_1.png dog_downsized/image_2.png dog_downsized/image_3.png dog_downsized/image_4.png"

python train_dreambooth_lora.py \
    --report_to wandb \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --instance_data_dir=$INSTANCE_DIR \
    --output_dir=$OUTPUT_DIR \
    --instance_prompt="a sks dog" \
    --resolution=256 \
    --train_batch_size=4 \
    --gradient_accumulation_steps=1 \
    --learning_rate=1e-6 \ 
    --max_train_steps=2000 \
    --validation_prompt="a sks dog" \
    --validation_epochs=100 \
    --checkpointing_steps=500 \
    --pre_compute_text_embeddings \
    --tokenizer_max_length=77 \
    --text_encoder_use_attention_mask \
    --validation_images $VALIDATION_IMAGES \
    --class_labels_conditioning=timesteps
```

### IF Stage I Full Dreambooth
`--skip_save_text_encoder`: When training the full model, this will skip saving the entire T5 with the finetuned model. You can still load the pipeline
with a T5 loaded from the original model.

`use_8bit_adam`: Due to the size of the optimizer states, we recommend training the full XL IF model with 8bit adam. 

`--learning_rate=1e-7`: For full dreambooth, IF requires very low learning rates. With higher learning rates model quality will degrade. Note that it is 
likely the learning rate can be increased with larger batch sizes.

Using 8bit adam and a batch size of 4, the model can be trained in ~48 GB VRAM.

```sh
export MODEL_NAME="DeepFloyd/IF-I-XL-v1.0"

export INSTANCE_DIR="dog"
export OUTPUT_DIR="dreambooth_if"

accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of sks dog" \
  --resolution=64 \
  --train_batch_size=4 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-7 \
  --max_train_steps=150 \
  --validation_prompt "a photo of sks dog" \
  --validation_steps 25 \
  --text_encoder_use_attention_mask \
  --tokenizer_max_length 77 \
  --pre_compute_text_embeddings \
  --use_8bit_adam \
  --set_grads_to_none \
  --skip_save_text_encoder \
  --push_to_hub
```

### IF Stage II Full Dreambooth

`--learning_rate=5e-6`: With a smaller effective batch size of 4, we found that we required learning rates as low as
1e-8.

`--resolution=256`: The upscaler expects higher resolution inputs

`--train_batch_size=2` and `--gradient_accumulation_steps=6`: We found that full training of stage II particularly with
faces required large effective batch sizes.

```sh
export MODEL_NAME="DeepFloyd/IF-II-L-v1.0"
export INSTANCE_DIR="dog"
export OUTPUT_DIR="dreambooth_dog_upscale"
export VALIDATION_IMAGES="dog_downsized/image_1.png dog_downsized/image_2.png dog_downsized/image_3.png dog_downsized/image_4.png"

accelerate launch train_dreambooth.py \
  --report_to wandb \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a sks dog" \
  --resolution=256 \
  --train_batch_size=2 \
  --gradient_accumulation_steps=6 \
  --learning_rate=5e-6 \
  --max_train_steps=2000 \
  --validation_prompt="a sks dog" \
  --validation_steps=150 \
  --checkpointing_steps=500 \
  --pre_compute_text_embeddings \
  --tokenizer_max_length=77 \
  --text_encoder_use_attention_mask \
  --validation_images $VALIDATION_IMAGES \
  --class_labels_conditioning timesteps \
  --push_to_hub
```

## Stable Diffusion XL

We support fine-tuning of the UNet shipped in [Stable Diffusion XL](https://huggingface.co/papers/2307.01952) with DreamBooth and LoRA via the `train_dreambooth_lora_sdxl.py` script. Please refer to the docs [here](./README_sdxl.md). 
