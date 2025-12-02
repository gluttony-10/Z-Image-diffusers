import os
import gc
from mmgp import offload
import torch
import numpy as np
import gradio as gr
import socket
import psutil
import random
import argparse
import datetime
from diffusers import ZImagePipeline, ZImageTransformer2DModel
from transformers import Qwen3Model


parser = argparse.ArgumentParser() 
parser.add_argument("--server_name", type=str, default="127.0.0.1", help="IP地址，局域网访问改为0.0.0.0")
parser.add_argument("--server_port", type=int, default=7891, help="使用端口")
parser.add_argument("--share", action="store_true", help="是否启用gradio共享")
parser.add_argument("--mcp_server", action="store_true", help="是否启用mcp服务")
parser.add_argument("--compile", action="store_true", help="是否启用compile加速")
parser.add_argument("--res_vram", type=int, default=2000, help="显存保留，单位MB。数值越大，占用显存越小，速度越慢")
args = parser.parse_args()

print(" 启动中，请耐心等待 bilibili@十字鱼 https://space.bilibili.com/893892")
print(f'\033[32mPytorch版本：{torch.__version__}\033[0m')
if torch.cuda.is_available():
    device = "cuda" 
    print(f'\033[32m显卡型号：{torch.cuda.get_device_name()}\033[0m')
    total_vram_in_gb = torch.cuda.get_device_properties(0).total_memory / 1073741824
    print(f'\033[32m显存大小：{total_vram_in_gb:.2f}GB\033[0m')
    mem = psutil.virtual_memory()
    print(f'\033[32m内存大小：{mem.total/1073741824:.2f}GB\033[0m')
    if torch.cuda.get_device_capability()[0] >= 8:
        print(f'\033[32m支持BF16\033[0m')
        dtype = torch.bfloat16
    else:
        print(f'\033[32m不支持BF16，仅支持FP16\033[0m')
        dtype = torch.float16
else:
    print(f'\033[32mCUDA不可用，请检查\033[0m')
    device = "cpu"

os.makedirs("outputs", exist_ok=True)
repo_id = "./models/Z-Image-Turbo"
budgets = int(torch.cuda.get_device_properties(0).total_memory/1048576 - args.res_vram)
stop_generation = False

text_encoder = offload.fast_load_transformers_model(
    f"{repo_id}/text_encoder/mmgp.safetensors",
    do_quantize=False,
    modelClass=Qwen3Model,
    forcedConfigPath=f"{repo_id}/text_encoder/config.json",
)
transformer = offload.fast_load_transformers_model(
    f"{repo_id}/transformer/mmgp.safetensors",
    do_quantize=False,
    modelClass=ZImageTransformer2DModel,
    forcedConfigPath=f"{repo_id}/transformer/config.json",
)
pipe = ZImagePipeline.from_pretrained(
    repo_id, 
    text_encoder=text_encoder,
    transformer=transformer,
    torch_dtype=dtype,
    low_cpu_mem_usage=False, 
)
#text_encoder._dtype = dtype 
mmgp = offload.all(
    pipe, 
    pinnedMemory= ["text_encoder", "transformer"],
    budgets={'*': budgets}, 
    extraModelsToQuantize = ["text_encoder"],
    compile=True if args.compile else False,
)
"""offload.save_model(
    model=pipe.transformer, 
    file_path=f"{repo_id}/transformer/mmgp.safetensors", 
    config_file_path=f"{repo_id}/transformer/config.json",
)
offload.save_model(
    model=pipe.text_encoder, 
    file_path=f"{repo_id}/text_encoder/mmgp.safetensors", 
    #config_file_path=f"{repo_id}/text_encoder/config.json",
)"""

# 解决冲突端口（感谢licyk酱提供的代码~）
def find_port(port: int) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        if s.connect_ex(("localhost", port)) == 0:
            print(f"端口 {port} 已被占用，正在寻找可用端口...")
            return find_port(port=port + 1)
        else:
            return port
        

def exchange_width_height(width, height):
    return height, width, "✅ 宽高交换完毕"


def stop_generate():
    global stop_generation
    stop_generation = True
    return "🛑 等待生成中止"


def scale_resolution_1_5(width, height):
    """
    将宽度和高度都放大1.5倍，并按照16的倍数向下取整
    """
    new_width = int(width * 1.5) // 16 * 16
    new_height = int(height * 1.5) // 16 * 16
    return new_width, new_height, "✅ 分辨率已调整为1.5倍"


def generate(
    prompt, 
    width, 
    height, 
    num_inference_steps, 
    batch_images, 
    seed_param, 
):
    global stop_generation
    results = []
    if seed_param < 0:
        seed = random.randint(0, np.iinfo(np.int32).max)
    else:
        seed = seed_param
    prompt_embeds, _ = pipe.encode_prompt(prompt)
    for i in range(batch_images):
        if stop_generation:
            stop_generation = False
            yield results, f"✅ 生成已中止，最后种子数{seed+i-1}"
            break
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"outputs/{timestamp}.png"
        output = pipe(
            height=height,
            width=width,
            num_inference_steps=num_inference_steps, 
            guidance_scale=0.0, 
            generator=torch.Generator().manual_seed(seed+i),
            prompt_embeds=prompt_embeds,
        )
        image = output.images[0]
        image.save(filename)
        results.append(image)
        yield results, f"种子数{seed+i}，保存地址{filename}"
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    

with gr.Blocks(title="Z-Image-diffusers", theme=gr.themes.Soft(font=[gr.themes.GoogleFont("IBM Plex Sans")])) as demo:
    gr.Markdown("""
            <div>
                <h2 style="font-size: 30px;text-align: center;">Z-Image-diffusers</h2>
            </div>
            <div style="text-align: center;">
                十字鱼
                <a href="https://space.bilibili.com/893892">🌐bilibili</a> 
                |Z-Image-diffusers
                <a href="https://github.com/gluttony-10/Z-Image-diffusers">🌐github</a> 
            </div>
            <div style="text-align: center; font-weight: bold; color: red;">
                ⚠️ 该演示仅供学术研究和体验使用。
            </div>
            """)
    
    with gr.Tabs():
        with gr.TabItem("Z-Image-diffusers"):
            with gr.Row():
                with gr.Column():
                    prompt = gr.Textbox(label="提示词", placeholder="请输入提示词...")
                    generate_button = gr.Button("🎬 开始生成", variant='primary', scale=4)
                    with gr.Accordion("参数设置", open=True):
                        with gr.Row():
                            width = gr.Slider(label="宽度", minimum=256, maximum=2048, step=16, value=1024)
                            height = gr.Slider(label="高度", minimum=256, maximum=2048, step=16, value=1024)
                        with gr.Row():
                            exchange_button = gr.Button("🔄 交换宽高")
                            scale_1_5_button = gr.Button("1.5倍分辨率")
                        batch_images = gr.Slider(label="批量生成", minimum=1, maximum=100, step=1, value=1)
                        num_inference_steps = gr.Slider(label="采样步数（推荐9步）", minimum=1, maximum=100, step=1, value=9)
                        seed_param = gr.Number(label="种子，请输入自然数，-1为随机", value=-1)
                with gr.Column():
                    info = gr.Textbox(label="提示信息", interactive=False)
                    image_output = gr.Gallery(label="生成结果", interactive=False)
                    stop_button = gr.Button("中止生成", variant="stop")
        
    gr.on(
        triggers=[generate_button.click, prompt.submit],
        fn = generate,
        inputs = [
            prompt,
            width,
            height,
            num_inference_steps,
            batch_images,
            seed_param,
        ],
        outputs = [image_output, info]
    )
    exchange_button.click(
        fn=exchange_width_height, 
        inputs=[width, height], 
        outputs=[width, height, info]
    )
    scale_1_5_button.click(
        fn=scale_resolution_1_5,
        inputs=[width, height],
        outputs=[width, height, info]
    )
    stop_button.click(
        fn=stop_generate, 
        inputs=[], 
        outputs=[info]
    )

if __name__ == "__main__": 
    demo.launch(
        server_name=args.server_name, 
        server_port=find_port(args.server_port),
        share=args.share, 
        mcp_server=args.mcp_server,
        inbrowser=True,
    )