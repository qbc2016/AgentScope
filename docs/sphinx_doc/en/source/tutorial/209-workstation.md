(209-workstation-en)=

AgentScope Workstation is a platform built on AgentScope that allows for the zero-code creation of multi-agent applications. Here, you can easily create your applications by simply dragging and dropping. After installing the full version of AgentScope, you can enter `as_studio` in the command line to use the offline version; alternatively, you can experience the online version of AgentScope Workstation by visiting  https://agentscope.io/  without installing anything.

  In this tutorial, you will learn how to use AgentScope Workstation to build multi-agent applications without writing any code. We will introduce the functionalities of the Workstation and how to utilize them in depth through the following five sections.

  1. [Principle](#1-principal)
  2. [Terms](#2-terms)
  3. [Examples](#3-examples)
  4. [Building Multimodal Complex Applications](#4-Building-Multimodal-Complex-Applications)
  5. [Look-up Table for Workflow nodes](#5-Look-up-Table-for-Workflow-nodes)

  ### 1. Principal

  AgentScope Workstation is based on AgentScope and offers a drag-and-drop, zero-code way to create multi-agent applications. You can create a variety of interesting applications on the canvas by dragging and connecting nodes. Each node in the Workstation corresponds to the capabilities of a specific module within AgentScope. By chaining these nodes together into a flowchart, a multi-agent application is created.

  Furthermore, to facilitate the sharing and usage of flowcharts created in the Workstation by users of different levels, Workstation supports exporting the content on the canvas as JSON code. Users can re-import this workflow through the import button in Workstation and edit it again in a drag-and-drop manner. Additionally, Workstation also supports exporting flowcharts as Python code, which can be run directly or further developed by programmers through Python commands.

  The exported JSON and Python codes can both be operated using AgentScope’s integrated Gradio WebUI. Applications can be run by executing `as_gradio ${YOUR_FILE_NAME}.json` or `as_gradio ${YOUR_FILE_NAME}.py` to launch a WebUI based on Gradio. Moreover, this WebUI supports one-click publication and operation in ModelScope Studio.![img](https://img.alicdn.com/imgextra/i2/O1CN01anKDRw24QNxZl2N0p_!!6000000007385-55-tps-620-372.svg)

  ### 2. Terms

  The left-hand menu of the interface consists of three sections: Example, Workflow, and Gallery:

  - Example contains four simple tutorials for multi-agent applications, which demonstrate the construction process of multi-agent applications step by step.
  -  Workflow is the core of the Workstation, containing a variety of draggable agents, large models, tools, and logic nodes.
  - Gallery is an exhibition hall for multi-agent applications provided by the official team, showcasing various interesting applications built through the Workstation by both users and the official team.

  ![img](https://img.alicdn.com/imgextra/i4/O1CN010QYN0q1TgkJvcXfJy_!!6000000002412-2-tps-1500-755.png)

  #### a. Buttons and Functions

  The center of the interface features a canvas, with buttons above it arranged and described as follows:

  ![img](https://img.alicdn.com/imgextra/i3/O1CN01cXc59j1Peg4HtsR9k_!!6000000001866-2-tps-1500-784.png)

  ① Home: Resets the canvas to its initial size.

  ② Export JSON Code: Exports the current workflow's JSON code, which can be copied to local storage or run.

  ③ Upload JSON Code: Used to upload the workflow's JSON code.

  ④ Check: Checks whether the applications dragged onto the current canvas meet the requirements for running (e.g., whether certain required variables are missing, modules are missing, etc.)

  ⑤ Clear Canvas: Clears the current canvas (Note: This cannot be undone!)

  ⑥ Export Python Code: Exports the application on the current canvas as Python code, which can be copied for local editing or directly run.

  ⑦ Run: Publishes and runs in ModelScope Studio. (If executed from a local studio, it will run locally.)

  ⑧ Save JSON Code: Saves the workflow to the website (we guarantee the confidentiality and security of all user data).

  ⑨ Load JSON Code: Loads a workflow saved on the website.

  #### b. Workflow

  In the canvas, a connection between two nodes represents the transmission of messages between the nodes, which corresponds to the Msg module in AgentScope.

  ![img](https://img.alicdn.com/imgextra/i2/O1CN01657GZ31D8MjmWZ9TA_!!6000000000171-2-tps-1500-781.png)

  To simplify the message transmission process, the Workstation also provides logical nodes such as SequentialPipeline. For example, in a SequentialPipeline node, by arranging the Agents in a top-to-bottom order, messages will be transmitted sequentially from top to bottom.

  ![img](https://img.alicdn.com/imgextra/i3/O1CN01pyESTz264WHHFGyLO_!!6000000007608-2-tps-1500-781.png)

  ### 3. Examples

  Click on "Conversation" under "Example" in the left navigation bar to enter the tutorial page, and click the "Next" button to learn how to build a "User-Agent Dialogue" application step by step.

  ![img](https://img.alicdn.com/imgextra/i4/O1CN018Lauwx1hiwqTdZFrS_!!6000000004312-2-tps-1500-779.png)

  Once entered, click "Next" to proceed with the workflow construction learning:

  - Step 1: Drag out a model node.
  - Step 2: Drag out a WhileLoopPipeline node.
  - Step 3: Drag out a SequentialPipeline node and place it inside the WhileLoopPipeline node.
  - Step 4: Drag out a user node and a dialogue agent node, both of which should be placed inside the SequentialPipeline node.

  After completing the above four steps, a "User-Agent Dialogue" application is successfully set up.

  ![img](https://img.alicdn.com/imgextra/i3/O1CN01EXhgUu1dzepYNuLFt_!!6000000003807-2-tps-1500-777.png)

  After completing the learning, exit the tutorial mode, click on the example "Conversation", import a pre-built workflow, fill in the API KEY for Tongyi Qianwen in the blank space provided for the API key, and click the code export button.

  ![img](https://img.alicdn.com/imgextra/i3/O1CN01NYDAp01wyGKEJkN5T_!!6000000006376-2-tps-1500-782.png)

  ![img](https://img.alicdn.com/imgextra/i4/O1CN01bqrWmF1f1mHW3Se3c_!!6000000003947-2-tps-1500-782.png)

  The exported code is as follows:

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

  Save the aforementioned code to a file named `main.py`. Use the integrated Gradio WebUI in AgentScope to run the application. Enter `as_gradio main.py` in the command line to run the visual interface, input content to obtain results, and you can also edit the Python file for further customization.

  ![img](https://img.alicdn.com/imgextra/i3/O1CN01e5zZ0W1Y5FypeuscQ_!!6000000003007-2-tps-1500-664.png)

  ### 4. Building Multimodal Complex Applications

  In this section, we plan to experiment with building a graphic adventure multi-agent game using the AgentScope Workstation. The models we will be using include Tongyi Qianwen and Tongyi Wanxiang. Our workflow is designed as follows:

  1. Game host agent conducts the story flow -->
  2. Scene agent generates Wanxiang prompts -->
  3. Image synthesis tool creates images -->
  4. Users make adventure choices -->
  5. Enter a loop, and the game host continues the game...

  #### Starting building

  Step 1: First, drag the DashScope Chat module from the Model column and fill in the corresponding configuration.

  ![img](https://img.alicdn.com/imgextra/i3/O1CN01dtBq0d1oAmby91kuq_!!6000000005185-2-tps-1500-783.png)

  Step 2: Drag out an initial message to initialize the message flow in the application, fill in the configuration, with the Content set to "Game starts."

  ![img](https://img.alicdn.com/imgextra/i3/O1CN01rIsY6E1wkyq7XHOx2_!!6000000006347-2-tps-1500-783.png)

  Step 3: Drag out logical nodes: a WhileLoopPipeline and a SequentialPipeline (which needs to be placed inside the WhileLoopPipeline) to maintain a continuous serial execution of the workflow and keep the dialogue ongoing. Since the loop needs to be maintained continuously, set the termination condition of the While loop to a lambda function: `lambda *args: True`, and connect it to Msg.

  ![img](https://img.alicdn.com/imgextra/i3/O1CN01YngTye1OkLLucIm0A_!!6000000001743-2-tps-1500-783.png)

  Step 4: Drag two Dialog Agents and one UserAgent node from the Agents section on the left, and an Image Synthesis node from the Tools section. Arrange these nodes in the SequentialPipeline from top to bottom in the following order: DialogAgent, DialogAgent, ImageSynthesis, UserAgent, representing the game host agent, scene agent, image synthesis tool, and user agent in the game, respectively.

  ![img](https://img.alicdn.com/imgextra/i3/O1CN01dRWMkM1Unu7sUITsp_!!6000000002563-2-tps-1500-785.png)

  Step 5: Fill in the prompts and configurations for each agent separately.

  - Game host agent
    - ![img](https://img.alicdn.com/imgextra/i4/O1CN01FeWBBK1jS7WZ8eXiS_!!6000000004546-2-tps-562-766.png)
      - Name: Host
      - Prompt: You are a text-based adventure game themed around the Journey of Sun Wukong. Players will assume the role of Sun Wukong, exploring a fantastical world, facing various choices and challenges. Whenever a player makes a choice, you need to provide four options (A, B, C, D), each containing a different storyline or adventure. Ensure that each choice can lead to different consequences to enhance interactivity and enjoyment. At the start of each round, briefly describe the current situation and ask the player for their choice.

  - Scene agent

    - ![img](https://img.alicdn.com/imgextra/i4/O1CN01RtFq3x1alVkY0Irhf_!!6000000003370-2-tps-562-766.png)

    - Name: Scene generator
    - Prompt: Task Objective: Based on the provided storyboard input, create a set of descriptive phrases aimed at guiding the text-to-image model to generate high-quality picture book images with the appropriate style and characteristics. \nDescriptions should be concise, clear keywords or phrases, separated by commas, to ensure that the images generated clearly reflect the scenes described in the picture book.\n If the scene includes characters, the current expressions of the characters should be included. For example, a good description might contain: character + expression + action description + scene description. The main character is Sun Wukong, so your output should start with Sun Wukong.

  - Image synthesis tool
    - ![img](https://img.alicdn.com/imgextra/i1/O1CN01bbey0f1sLYO9Z5G4w_!!6000000005750-2-tps-558-1050.png)

  - UserAgent
    - ![img](https://img.alicdn.com/imgextra/i2/O1CN01jrHEQQ1QOTfxGIHij_!!6000000001966-2-tps-556-434.png)

  Step 6: Export the code, save it as a Python file, and run the Python code locally using `as_gradio`.

  ![img](https://img.alicdn.com/imgextra/i3/O1CN01aPjsRO1GEj5KnjZtk_!!6000000000591-2-tps-1500-785.png)

  Step 7: Click on the popped-up WebUI link to play the game, completing the setup of the multi-agent application.

  ![img](https://img.alicdn.com/imgextra/i1/O1CN01WOfn0c1pn5LjrybVT_!!6000000005404-2-tps-1500-753.png)

  ![img](https://img.alicdn.com/imgextra/i2/O1CN01Pf3obc1KylMKZlZRR_!!6000000001233-2-tps-1500-756.png)



  ### 5. Look-up Table for Workflow nodes

| **Type**                      | **Terms**          | **Descriptions**                                             | **Remark (Visit the URL to obtain an API Key and activate the corresponding model/service.)** |
| ----------------------------- | ------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **Model**                     | DashScope Chat     | Model for DashScope chat                                     | https://bailian.console.aliyun.com/?apiKey=1#/api-key        |
|                               | OpenAI Chat        | Model for OpenAI chat                                        | https://platform.openai.com/api-keys                         |
|                               | Post API           | Model for Post API                                           | http://doc.agentscope.io/en/tutorial/203-model.html#post-request-api |
| **Message**                   | Msg                | The message which is responsible for storingthat the information of a message translated among workflow |                                                              |
| **Agent**                     | DialogAgent        | Athe a  dialog agent that can interact with users or other agents |                                                              |
|                               | UserAgent          | A proxy agent for user                                       |                                                              |
|                               | DictDialogAgent    | An agent that generates response in a dict format            |                                                              |
|                               | ReActAgent         | Agent for ReAct (reasoning and acting) with tools            |                                                              |
|                               | Broadcast Agent    | An agent that only broadcasts its content                    |                                                              |
| **Pipeline**                  | Placeholder        | A placeholder that does nothing                              |                                                              |
|                               | MsgHug             | MsgHub is used to share messages among a group of agents     |                                                              |
|                               | SequentialPipeline | A template pipeline for implementing sequential logic (from top to bottom) |                                                              |
|                               | ForLoopPipeline    | A template pipeline for implementing control flow like for-loop |                                                              |
|                               | WhileLoopPipeline  | A template pipeline for implementing control flow like while-loop |                                                              |
|                               | IfElsePipeline     | A template pipeline for implementing control flow with if-else logic |                                                              |
|                               | SwitchPipeline     | A template pipeline for implementing control flow with switch-case logic |                                                              |
| **Tool**                      | Image Synthesis    | Integrate the Text to Image                                  | https://bailian.console.aliyun.com/?apiKey=1#/api-key        |
|                               | Image Composition  | Composite images into one image                              |                                                              |
|                               | Image Motion       | Convert an image to an mp4 or a gif by shifting the perspective |                                                              |
|                               | Video Composition  | Composite videos into one video                              |                                                              |
|                               | Post               | Post a request and return the response                       |                                                              |
|                               | Code               | Execute Python Code                                          |                                                              |
| **Service**（For ReactAgent） | Bing search        | Integrate the Bing Search service                            | https://learn.microsoft.com/en-us/bing/search-apis/bing-web-search/quickstarts/rest/python |
|                               | Google search      | Integrate the Google Search service                          | https://console.cloud.google.com/apis/credentials            |
|                               | Python interpreter | Integrate the Python Interpreter                             |                                                              |
|                               | Read Text          | Integrate the Read Text service                              |                                                              |
|                               | Write Text         | Integrate the Write Text Service                             |                                                              |
|                               | Text to Audio      | Integrate the Text to Audio Service                          | https://bailian.console.aliyun.com/?apiKey=1#/api-key        |
|                               | Text to Image      | Integrate the Text to Image Service                          | https://bailian.console.aliyun.com/?apiKey=1#/api-key        |

[[Back to the top]](#209-workstation-en)