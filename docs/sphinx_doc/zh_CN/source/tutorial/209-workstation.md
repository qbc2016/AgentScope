(209-workstation-en)=

AgentScope Workstation是一个基于AgentScope的零代码创建多智能体应用的平台。在这里，您可以通过拖拽的方式，轻松地创建您的应用。您可以在安装好AgentScope完整版本后在命令行输入`as_studio`使用离线版本；也可以不安装，进入 https://agentscope.io/体验AgentScope Workstation的在线版本。

在这个教程中，您将学会如何利用AgentScope Workstation零代码构建多智能应用。现在我们将按照以下5个部分来从浅入深介绍Workstation的各项功能以及如何使用它。

1. [基本原理](#1-基本原理)
2. [常用术语](#2-常用术语)
3. [实例演示](#3-实例演示)
4. [搭建多模态复杂应用](#4-搭建多模态复杂应用)
5. [工作流程节点对照表](#5-工作流程节点对照表)

### 1. 基本原理

AgentScope Workstation基于AgentScope，提供了一个拖拽式、零代码创建多智能体应用的方式，您可以在画布上通过拖拽并连接节点来创建各种有趣的应用。Workstation中每个节点都对应着AgentScope的某个特定模块的能力，通过串联组装各个节点成为一个流程图，一个多智能体应用就此产生。

此外，为了方便不同级别的用户分享、使用Workstation中创建的流程图，Workstation支持将画布上的内容导出为JSON代码。用户可以通过Workstation中的导入按钮，将此工作流重新导入，并拖拽式二次编辑。同时，Workstation还支持将流程图导出为Python代码，可以直接或经过有编程经验的开发者二次开发后通过Python命令来运行。

导出的JSON代码和Python代码都支持通过使用AgentScope集成好的Gradio WebUI运行应用，即运行`as_gradio ${YOUR_FILE_NAME}.json` 或者 `as_gradio ${YOUR_FILE_NAME}.py`来启动一个基于Gradio实现的WebUI，同时，该WebUI支持一键在创空间发布并运行。

![img](https://img.alicdn.com/imgextra/i3/O1CN01QawpLQ1Jm6nhuMY5i_!!6000000001070-0-tps-1500-894.jpg)

### 2. 常用术语

界面左侧菜单栏有三块内容，分别是Example、Workflow和Gallery：

- Example中包含了四个简单的智能体应用教程，可以一步一步的展示多智能体应用的构建流程。
- Workflow是Workstation的核心，里面包含了各种可拖拽的智能体、大模型、工具、逻辑节点。
- Gallery 是一个由官方提供的多智能体应用展览馆，里面展出了由用户和官方共同构建的各种通过 Workstation 打造的有趣应用。

![img](https://img.alicdn.com/imgextra/i4/O1CN010QYN0q1TgkJvcXfJy_!!6000000002412-2-tps-1500-755.png)

#### a. 按钮及功能

界面中间是画布，以及画布上方的按钮分布和描述如下：

![img](https://img.alicdn.com/imgextra/i3/O1CN01cXc59j1Peg4HtsR9k_!!6000000001866-2-tps-1500-784.png)

① 主页：还原画布初始大小

② 导出JSON代码：导出当前工作流的JSON代码，可复制到本地存储或运行

③ 上传JSON代码：用来上传工作流的JSON代码

④ 检查：检查拖拽到当前画布上的应用是否满足运行要求（比如某些必填变量是否缺失、模块是否缺失等）

⑤ 清空画布：清空当前画布（注意：无法撤销！）

⑥ 导出Python代码：将当前画布上的应用导出为Python代码，可复制到本地二次编辑或直接运行

⑦ 运行：发布到魔搭创空间运行（如果是从本地的studio进入的话，则是本地运行）

⑧ 保存JSON代码：将工作流保存到网站（我们承诺不会使用或泄露用户的任何信息。）

⑨ 载入JSON代码：将保存在网站中的工作流载入

#### b. 工作流

画布中两个节点的连线表示消息在两个节点的传递，该消息对应AgentScope的Msg模块。

![img](https://img.alicdn.com/imgextra/i2/O1CN01657GZ31D8MjmWZ9TA_!!6000000000171-2-tps-1500-781.png)

为了简化消息传递流程，Workstation还提供了SequentialPipeline等逻辑节点，比如在SequentialPipeline节点中，只需要将Agent按照从上往下的顺序排列，那么消息将会从上往下进行传递。

![img](https://img.alicdn.com/imgextra/i3/O1CN01pyESTz264WHHFGyLO_!!6000000007608-2-tps-1500-781.png)

### 3. 实例演示

点击左侧导航栏Example下的Conversation进入教学页面并点击Next按钮学习按步骤搭建“用户-智能体对话”应用。

![img](https://img.alicdn.com/imgextra/i4/O1CN018Lauwx1hiwqTdZFrS_!!6000000004312-2-tps-1500-779.png)

进入后点击“下一步”进行工作流搭建学习：

- 第一步：拖拽出一个模型节点
- 第二步：拖拽出一个循环逻辑节点
- 第三步：拖拽出一个顺序执行逻辑节点，并拖到循环节点内
- 第四步：拖拽出用户节点和对话智能体节点，均拖到顺序执行逻辑节点内

完成上述四个步骤之后，一个“用户-智能体对话”应用就搭建完成了。

![img](https://img.alicdn.com/imgextra/i3/O1CN01EXhgUu1dzepYNuLFt_!!6000000003807-2-tps-1500-777.png)

学习完成后退出教程模式，点击示例Conversation，导入一个构建好的工作流，在API key的空白处填上通义千问的API KEY，并单击代码导出按钮。

![img](https://img.alicdn.com/imgextra/i3/O1CN01NYDAp01wyGKEJkN5T_!!6000000006376-2-tps-1500-782.png)

![img](https://img.alicdn.com/imgextra/i4/O1CN01bqrWmF1f1mHW3Se3c_!!6000000003947-2-tps-1500-782.png)

导出的代码如下：

```python
import agentscope
from agentscope.agents import DialogAgent, UserAgent
from agentscope.manager import ModelManager
from agentscope.pipelines import SequentialPipeline, WhileLoopPipeline


def main():
    agentscope.init(
        logger_level="DEBUG",
    )
    ModelManager.get_instance().load_model_configs(
        [
            {
                "api_key": "******",
                "config_name": "qwen",
                "messages_key": "input",
                "model_name": "qwen-max",
                "model_type": "dashscope_chat",
                "seed": 0,
                "temperature": 0,
            }
        ]
    )
    flow = None
    agent_5 = UserAgent(name="User")
    agent_6 = DialogAgent(
        model_config_name="qwen",
        name="Assistant",
        sys_prompt="You are a helpful assistant.",
    )
    pipeline_4 = SequentialPipeline([agent_5, agent_6])
    pipeline_3 = WhileLoopPipeline(
        loop_body_operators=[pipeline_4], condition_func=lambda *args: True
    )

    flow_4 = pipeline_3(flow)


if __name__ == "__main__":
    main()
```

将上述代码保存到`main.py`文件。使用AgentScope集成好的Gradio WebUI运行应用。在命令行输入 `as_gradio main.py`运行可视化界面，输入内容获得结果，你也可以二次编辑该Python文件进行二次定制。

![img](https://img.alicdn.com/imgextra/i3/O1CN01e5zZ0W1Y5FypeuscQ_!!6000000003007-2-tps-1500-664.png)

### 4. 搭建多模态复杂应用

在这一节中，我们计划试用AgentScope Workstation搭建一个图文冒险多智能体游戏，会用到的模型有通义千问和通义万相。我们设计工作流如下：

1. 游戏主持人智能体主持故事流程 -->
2. 场景智能体生成万相Prompt -->
3. 图像生成工具生成图片 -->
4. 用户进行选择冒险选项选择-->
5. 进入循环，游戏主持人继续游戏...

#### 开始搭建

第一步：我们先从Model栏拖拽出DashScope Chat模块，并填写相应的配置

![img](https://img.alicdn.com/imgextra/i3/O1CN01dtBq0d1oAmby91kuq_!!6000000005185-2-tps-1500-783.png)

第二步：拖拽出初始消息用来初始化应用中的消息Flow，填入配置，Content为“Game starts.”

![img](https://img.alicdn.com/imgextra/i3/O1CN01rIsY6E1wkyq7XHOx2_!!6000000006347-2-tps-1500-783.png)

第三步：拖拽出逻辑节点：循环WhileLoopPipeline以及SequentialPipeline（需要拖放到WhileLoopPipeline内）用来循环串行执行工作流保持对话进行，因为需要一直保持循环进行，设定While循环结束条件lambda函数为：`lambda *args: True`，并连接Msg。

![img](https://img.alicdn.com/imgextra/i3/O1CN01YngTye1OkLLucIm0A_!!6000000001743-2-tps-1500-783.png)

第四步：从左侧Agent里拖出两个Dialog Agent和一个UserAgent节点，从Tool里拖出一个Image Synthesis节点；在SequentialPipeline中由上到下排列 DialogAgent、DialogAgent、ImageSynthesis、UserAgent分别代表游戏中的游戏主持人智能体、场景智能体、 图像生成工具、用户智能体。

![img](https://img.alicdn.com/imgextra/i3/O1CN01dRWMkM1Unu7sUITsp_!!6000000002563-2-tps-1500-785.png)

第五步：分别填写每个智能体的提示词与配置
- 游戏主持人智能体
  - ![img](https://img.alicdn.com/imgextra/i4/O1CN01FeWBBK1jS7WZ8eXiS_!!6000000004546-2-tps-562-766.png)
  - Name: 游戏主持人
  - System prompt：你是一款文字冒险游戏，主题是孙悟空历险记。玩家将扮演孙悟空，探索奇幻的世界，面临各种选择和挑战。每当玩家做出选择时，你需要提供四个选项（A、B、C、D），每个选项都应包含不同的故事情节或冒险。确保每个选择都有可能导致不同的后果，增强互动性和趣味性。在每个回合开始时，简要描述当前的情境，并询问玩家的选择。

- 场景智能体

  - ![img](https://img.alicdn.com/imgextra/i4/O1CN01RtFq3x1alVkY0Irhf_!!6000000003370-2-tps-562-766.png)

  - 图片提示词生成器

  - 提示词：任务目标：根据输入的分镜，创造一组描述性短语，旨在用于指导文生图模型生成具有相应风格和特性的高质量绘本画面。 \n描述应该是简洁明了的关键词或短语，以逗号分隔，以确保在画面生成时能够清楚地反映绘本所描述的画面。\n如果包含人物的话，需要当下人物的神情。比如一个好的描述可能包含：人物+神情+动作描述+场景描述。人物的主角是孙悟空，因此你的输出应该以孙悟空开头。
- 图像生成工具
  - ![img](https://img.alicdn.com/imgextra/i1/O1CN01bbey0f1sLYO9Z5G4w_!!6000000005750-2-tps-558-1050.png)
- 用户智能体
  - ![img](https://img.alicdn.com/imgextra/i2/O1CN01jrHEQQ1QOTfxGIHij_!!6000000001966-2-tps-556-434.png)

第六步：导出代码，保存为Python文件，并在本地使用`as_gradio`运行该Python代码。

![img](https://img.alicdn.com/imgextra/i3/O1CN01aPjsRO1GEj5KnjZtk_!!6000000000591-2-tps-1500-785.png)

第七步：点击弹出的WebUI链接，进行游戏，完成多智能体应用搭建。

![img](https://img.alicdn.com/imgextra/i1/O1CN01WOfn0c1pn5LjrybVT_!!6000000005404-2-tps-1500-753.png)

![img](https://img.alicdn.com/imgextra/i2/O1CN01Pf3obc1KylMKZlZRR_!!6000000001233-2-tps-1500-756.png)

### 5. 工作流程节点对照表

| 类型                          | **术语**           | Descriptions                                  | 注 (访问url获取API Key并开通相应模型/服务)                   |
| ----------------------------- | ------------------ | --------------------------------------------- | ------------------------------------------------------------ |
| **Model**                     | DashScope Chat     | DashScope聊天模型                             | https://bailian.console.aliyun.com/?apiKey=1#/api-key        |
|                               | OpenAI Chat        | OpenAI聊天模型                                | https://platform.openai.com/api-keys                         |
|                               | Post API           | Post API模型                                  |                                                              |
| **Message**                   | Msg                | 负责存储工作流中传递的消息                    |                                                              |
| **Agent**                     | DialogAgent        | 一个可以与用户或其他智能体交互的对话智能体    |                                                              |
|                               | UserAgent          | 用户代理智能体                                |                                                              |
|                               | DictDialogAgent    | 以字典格式生成响应的智能体                    |                                                              |
|                               | ReActAgent         | 带有工具的ReAct智能体（能够推理和行动）       |                                                              |
|                               | Broadcast Agent    | 仅广播其内容的智能体                          |                                                              |
| **Pipeline**                  | Placeholder        | 一个什么也不做的占位符                        |                                                              |
|                               | MsgHug             | MsgHub用于在一组智能体之间共享消息            |                                                              |
|                               | SequentialPipeline | 用于实现自上而下顺序逻辑的模板管道            |                                                              |
|                               | ForLoopPipeline    | 用于实现类似for循环的控制流的模板管道         |                                                              |
|                               | WhileLoopPipeline  | 用于实现类似while循环的控制流的模板管道       |                                                              |
|                               | IfElsePipeline     | 用于实现具有if-else逻辑的控制流的模板管道     |                                                              |
|                               | SwitchPipeline     | 用于实现具有switch-case逻辑的控制流的模板管道 |                                                              |
| **Tool**                      | Image Synthesis    | 文本生成图像                                  | https://bailian.console.aliyun.com/?apiKey=1#/api-key        |
|                               | Image Composition  | 图片多合一                                    |                                                              |
|                               | Image Motion       | 通过视角移动将图像转换为mp4或gif              |                                                              |
|                               | Video Composition  | 视频多合一                                    |                                                              |
|                               | Post               | 发送请求并返回响应                            |                                                              |
|                               | Code               | 执行Python代码                                |                                                              |
| **Service**（For ReactAgent） | Bing search        | 必应搜索服务                                  | https://learn.microsoft.com/en-us/bing/search-apis/bing-web-search/quickstarts/rest/python |
|                               | Google search      | 谷歌搜索服务                                  | https://console.cloud.google.com/apis/credentials            |
|                               | Python interpreter | Python代码执行                                |                                                              |
|                               | Read Text          | 读取文本                                      |                                                              |
|                               | Write Text         | 写入文本                                      |                                                              |
|                               | Text to Audio      | 文本生成音频                                  | https://bailian.console.aliyun.com/?apiKey=1#/api-key        |
|                               | Text to Image      | 文本生成图像                                  | https://bailian.console.aliyun.com/?apiKey=1#/api-key        |



[[Back to the top]](#209-workstation-en)