# -*- coding: utf-8 -*-
"""Text to Image"""
import os
from typing import Optional, Literal

from agentscope.message import Msg
from agentscope.models import DashScopeImageSynthesisWrapper
from agentscope.utils.common import _download_file


def image_synthesis(
    msg: Msg,
    api_key: str,
    n: int = 1,
    size: Literal["1024*1024", "720*1280", "1280*720"] = "1024*1024",
    model: str = "wanx-v1",
    save_dir: Optional[str] = None,
) -> Msg:
    """Generate image(s) based on the given Msg, and return Msg.

    Args:
        msg (`Msg`):
            The msg to generate image.
        api_key (`str`):
            The api key for the dashscope api.
        n (`int`, defaults to `1`):
            The number of images to generate.
        size (`Literal["1024*1024", "720*1280", "1280*720"]`, defaults to
        `"1024*1024"`):
            Size of the image.
        model (`str`, defaults to '"wanx-v1"'):
            The model to use.
        save_dir (`Optional[str]`, defaults to 'None'):
            The directory to save the generated images. If not specified,
            will return the web urls.

    Returns:
        Msg
    """
    text2img = DashScopeImageSynthesisWrapper(
        config_name="dashscope-text-to-image-service",  # Just a placeholder
        model_name=model,
        api_key=api_key,
    )
    try:
        res = text2img(
            prompt=msg.content,
            n=n,
            size=size,
        )
        urls = res.image_urls

        # save images to save_dir
        if urls is not None:
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                urls_local = []
                # Obtain the image file names in the url
                for url in urls:
                    image_name = url.split("/")[-1]
                    image_path = os.path.abspath(
                        os.path.join(save_dir, image_name),
                    )
                    # Download the image
                    _download_file(url, image_path)
                    urls_local.append(image_path)

            return Msg(
                name="ImageSynthesis",
                content=urls,
                url=urls,
                role="assistant",
                echo=True,
            )
        else:
            return Msg(
                name="ImageSynthesis",
                content="Error: Failed to generate images",
                role="assistant",
                echo=True,
            )

    except Exception as e:
        return Msg(
            name="ImageSynthesis",
            content=str(e),
            role="assistant",
            echo=True,
        )
