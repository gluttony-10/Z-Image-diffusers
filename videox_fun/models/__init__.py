import importlib.util

from diffusers import AutoencoderKL
from transformers import (AutoProcessor, AutoTokenizer, CLIPImageProcessor,
                          CLIPTextModel, CLIPTokenizer,
                          CLIPVisionModelWithProjection, LlamaModel,
                          LlamaTokenizerFast, LlavaForConditionalGeneration,
                          Mistral3ForConditionalGeneration, PixtralProcessor,
                          Qwen3ForCausalLM, T5EncoderModel, T5Tokenizer,
                          T5TokenizerFast)

try:
    from transformers import (Qwen2_5_VLConfig,
                              Qwen2_5_VLForConditionalGeneration,
                              Qwen2Tokenizer, Qwen2VLProcessor)
except:
    Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer = None, None
    Qwen2VLProcessor, Qwen2_5_VLConfig = None, None
    print("Your transformers version is too old to load Qwen2_5_VLForConditionalGeneration and Qwen2Tokenizer. If you wish to use QwenImage, please upgrade your transformers package to the latest version.")

from .z_image_transformer2d import ZImageTransformer2DModel
from .z_image_transformer2d_control import ZImageControlTransformer2DModel
