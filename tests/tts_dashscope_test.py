# -*- coding: utf-8 -*-
"""The unittests for DashScope Realtime TTS model."""
import os
from unittest import IsolatedAsyncioTestCase

from agentscope.agent import ReActAgent
from agentscope.formatter import DashScopeChatFormatter
from agentscope.message import Msg
from agentscope.model import DashScopeChatModel
from agentscope.tts import DashScopeRealtimeTTSModel


class DashScopeRealtimeTTSModelTest(IsolatedAsyncioTestCase):
    """The unittests for DashScope Realtime TTS model."""

    async def asyncSetUp(self) -> None:
        """Set up the test case asynchronously."""
        self.msg1 = Msg(
            "user",
            "",
            "user",
        )

    async def test_dashscope_realtime_tts_model(self) -> None:
        """Test the DashScope Realtime TTS model."""
        model = DashScopeRealtimeTTSModel(
            api_key=os.environ["DASHSCOPE_API_KEY"],
            stream=False,
        )
        agent = ReActAgent(
            name="Friday",
            sys_prompt="",
            model=DashScopeChatModel(
                api_key=os.environ["DASHSCOPE_API_KEY"],
                model_name="qwen3-max",
            ),
            formatter=DashScopeChatFormatter(),
        )

        msg_to_speak = Msg("user", [], "user")

        async with model:
            content = "你好啊！\n\n小明我是你爸爸你是谁啊？我不知道你是谁啊，你能告诉我吗？真的吗？真的是你吗？我不认识你啊！"
            for i in range(1, len(content)):
                self.msg1.content = content[:i]
                res = await model.push(self.msg1)
                if res.content:
                    print(i, len(res.content[0]["source"]["data"]))
                    msg_to_speak.content = res.content
                    await agent.print(msg_to_speak, last=False)

            res = await model.synthesize(self.msg1)
            if res.content:
                print(i, len(res.content[0]["source"]["data"]))
                msg_to_speak.content = res.content
                await agent.print(msg_to_speak, last=False)

            #
            # self.msg1.content = "你好啊！\n\n"
            # res1 = await model.push(self.msg1)
            # if res1.content:
            #     print(1, len(res1.content[0]["source"]["data"]))
            #     msg_to_speak.content = res1.content
            #     await agent.print(msg_to_speak, last=False)
            #
            # self.msg1.content = "你好啊！\n\n小明我是你爸爸你是谁啊？"
            # res2 = await model.push(self.msg1)
            #
            # if res2.content:
            #     print(2, len(res2.content[0]["source"]["data"]))
            #     msg_to_speak.content = res2.content
            #     await agent.print(msg_to_speak, last=False)
            #
            # self.msg1.content = "你好啊！\n\n小明我是你爸爸你是谁啊？我不知道你是谁啊，你能告诉我吗？"
            # res3 = await model.push(self.msg1)
            #
            # if res3.content:
            #     print(3, len(res3.content[0]["source"]["data"]))
            #     msg_to_speak.content = res3.content
            #     await agent.print(msg_to_speak, last=False)
            #
            # self.msg1.content = "你好啊！\n\n小明我是你爸爸你是谁啊？我不知道你是
            # 谁啊，你能告诉我吗？真的吗？真的是你吗？我不认识你啊！"
            # res4 = await model.synthesize(self.msg1)
            #
            # if model.stream:
            #     last_prefix = ""
            #     async for chunk in res4:
            #
            #         if chunk.content:
            #             print(4, len(last_prefix),
            #                   len(chunk.content[0]['source']['data']), len(
            #                     chunk.content[0]["source"]["data"].removeprefix(
            #                         last_prefix)))
            #             msg_to_speak.content = chunk.content
            #             await agent.print(msg_to_speak, last=False)
            #
            #             last_prefix = chunk.content[0]["source"]["data"]
            # else:
            #     if res4.content:
            #         print(4, len(res4.content[0]["source"]["data"]))
            #         msg_to_speak.content = res4.content
            #         await agent.print(msg_to_speak, last=True)
            #
            #
            #
            #
