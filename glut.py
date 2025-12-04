import os
import gc
import math
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
from videox_fun.utils.utils import get_image_latent
from videox_fun.models import ZImageControlTransformer2DModel
from videox_fun.pipeline import ZImageControlPipeline
from transformers import Qwen3Model
from safetensors.torch import load_file


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
mode_loaded = None
pipe = None
mmgp = None
lora_loaded = None
lora_loaded_weights = None
lora_dir = "models/lora"
if os.path.exists(lora_dir):
    lora_files = [f for f in os.listdir(lora_dir) if f.endswith(".safetensors")]
    lora_choices = sorted(lora_files)
else:
    lora_choices = []

def load_model(mode, lora_dropdown, lora_weights):
    global pipe, mmgp
    text_encoder = offload.fast_load_transformers_model(
        f"{repo_id}/text_encoder/mmgp.safetensors",
        do_quantize=False,
        modelClass=Qwen3Model,
        forcedConfigPath=f"{repo_id}/text_encoder/config.json",
    )
    #text_encoder._dtype = dtype 
    if mode == "t2i":
        if pipe is not None:
            mmgp.release()
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
        load_lora(lora_dropdown, lora_weights)
    elif mode == "con":
        if pipe is not None:
            mmgp.release()
        """transformer = ZImageControlTransformer2DModel.from_pretrained(
            repo_id, 
            subfolder="transformer",
            low_cpu_mem_usage=True,
            torch_dtype=dtype,
        )
        state_dict = load_file("./models/Z-Image-Turbo-Fun-Controlnet-Union/Z-Image-Turbo-Fun-Controlnet-Union.safetensors")
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict
        m, u = transformer.load_state_dict(state_dict, strict=False)
        print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")"""
        transformer = offload.fast_load_transformers_model(
            f"{repo_id}/transformer/mmgp2.safetensors",
            do_quantize=False,
            modelClass=ZImageControlTransformer2DModel,
            forcedConfigPath=f"{repo_id}/transformer/config.json",
        )
        pipe = ZImageControlPipeline.from_pretrained(
            repo_id, 
            text_encoder=text_encoder,
            transformer=transformer,
            torch_dtype=dtype,
            low_cpu_mem_usage=False,
        )
        load_lora(lora_dropdown, lora_weights)
    mmgp = offload.all(
        pipe, 
        pinnedMemory= ["text_encoder", "transformer"],
        budgets={'*': budgets}, 
        extraModelsToQuantize = ["text_encoder"],
        compile=True if args.compile else False,
    )
    pipe.transformer.set_attention_backend("flash")
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


def load_lora(lora_dropdown, lora_weights):
    if lora_dropdown != []:
        global pipe
        adapter_names = []
        weightss = []
        weights = [float(w) for w in lora_weights.split(',')] if lora_weights else []
        for idx, lora_name in enumerate(lora_dropdown):
                adapter_name = os.path.splitext(os.path.basename(lora_name))[0]
                adapter_names.append(adapter_name)
                weight = weights[idx] if idx < len(weights) else 1.0
                weightss.append(weight)
                pipe.load_lora_weights(f"models/lora/{lora_name}", adapter_name=adapter_name)
                print(f"✅ 已加载LoRA模型: {lora_name} (权重: {weight})")
        pipe.set_adapters(adapter_names, adapter_weights=weightss)
        print("LoRA加载完成")

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


def adjust_width_height(image):
    image_width, image_height = image.size
    vae_width, vae_height = calculate_dimensions(1024*1024, image_width / image_height)
    calculated_height = vae_height // 32 * 32
    calculated_width = vae_width // 32 * 32
    return int(calculated_width), int(calculated_height), "✅ 根据图片调整宽高"


def calculate_dimensions(target_area, ratio):
    width = math.sqrt(target_area * ratio)
    height = width / ratio
    width = round(width / 32) * 32
    height = round(height / 32) * 32
    return width, height


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


def generate_t2i(
    prompt, 
    width, 
    height, 
    num_inference_steps, 
    batch_images, 
    seed_param, 
    lora_dropdown, 
    lora_weights,
):
    global stop_generation, mode_loaded, lora_loaded, lora_loaded_weights
    if mode_loaded != "t2i" or lora_loaded != lora_dropdown or lora_loaded_weights != lora_weights:
        load_model("t2i", lora_dropdown, lora_weights)
        mode_loaded = "t2i"
        lora_loaded, lora_loaded_weights = lora_dropdown, lora_weights
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


def generate_con(
    control_image,
    prompt, 
    width, 
    height, 
    num_inference_steps, 
    strength,
    batch_images, 
    seed_param, 
    lora_dropdown, 
    lora_weights,
):
    global stop_generation, mode_loaded, lora_loaded, lora_loaded_weights
    if mode_loaded != "con" or lora_loaded != lora_dropdown or lora_loaded_weights != lora_weights:
        load_model("con", lora_dropdown, lora_weights)
        mode_loaded = "con"
        lora_loaded, lora_loaded_weights = lora_dropdown, lora_weights
    results = []
    if seed_param < 0:
        seed = random.randint(0, np.iinfo(np.int32).max)
    else:
        seed = seed_param
    prompt_embeds, _ = pipe.encode_prompt(prompt)
    control_image = get_image_latent(control_image, sample_size=[height, width])[:, :, 0]
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
            control_image = control_image,
            control_context_scale = strength,
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
    with gr.Accordion("LoRA设置(仅文生图可用)", open=False):
        with gr.Column():
            with gr.Row():
                lora_dropdown = gr.Dropdown(label="LoRA模型", info="存放LoRA模型到models/lora，可多选", choices=lora_choices, multiselect=True)
                lora_weights = gr.Textbox(label="LoRA权重", info="Lora权重，多个权重请用英文逗号隔开。例如：0.8,0.5,0.2", value="")
    with gr.Tabs():
        with gr.TabItem("文生图"):
            with gr.Row():
                with gr.Column():
                    prompt_t2i = gr.Textbox(label="提示词", placeholder="请输入提示词...")
                    generate_button_t2i = gr.Button("🖼️ 开始生成", variant='primary', scale=4)
                    with gr.Accordion("参数设置", open=True):
                        with gr.Row():
                            width_t2i = gr.Slider(label="宽度", minimum=256, maximum=2048, step=16, value=1024)
                            height_t2i = gr.Slider(label="高度", minimum=256, maximum=2048, step=16, value=1024)
                        with gr.Row():
                            exchange_button_t2i = gr.Button("🔄 交换宽高")
                            scale_1_5_button_t2i = gr.Button("1.5倍分辨率")
                        batch_images_t2i = gr.Slider(label="批量生成", minimum=1, maximum=100, step=1, value=1)
                        num_inference_steps_t2i = gr.Slider(label="采样步数（推荐9步）", minimum=1, maximum=100, step=1, value=9)
                        seed_param_t2i = gr.Number(label="种子，请输入自然数，-1为随机", value=-1)
                with gr.Column():
                    info_t2i = gr.Textbox(label="提示信息", interactive=False)
                    image_output_t2i = gr.Gallery(label="生成结果", interactive=False)
                    stop_button_t2i = gr.Button("中止生成", variant="stop")
        with gr.TabItem("ControlNet"):
            with gr.Row():
                with gr.Column():
                    image_con = gr.Image(label="输入控制图片（支持Canny、HED、Depth、Pose、MLSD）", type="pil", height=300)
                    prompt_con = gr.Textbox(label="提示词", placeholder="请输入提示词...")
                    generate_button_con = gr.Button("🖼️ 开始生成", variant='primary', scale=4)
                    with gr.Accordion("参数设置", open=True):
                        with gr.Row():
                            width_con = gr.Slider(label="宽度", minimum=256, maximum=2048, step=16, value=1024)
                            height_con = gr.Slider(label="高度", minimum=256, maximum=2048, step=16, value=1024)
                        with gr.Row():
                            exchange_button_con = gr.Button("🔄 交换宽高")
                            scale_1_5_button_con = gr.Button("1.5倍分辨率")
                        strength_con = gr.Slider(label="strength（推荐0.75）", minimum=0, maximum=1, step=0.01, value=0.75)
                        batch_images_con = gr.Slider(label="批量生成", minimum=1, maximum=100, step=1, value=1)
                        num_inference_steps_con = gr.Slider(label="采样步数（推荐9步）", minimum=1, maximum=100, step=1, value=9)
                        seed_param_con = gr.Number(label="种子，请输入自然数，-1为随机", value=-1)
                with gr.Column():
                    info_con = gr.Textbox(label="提示信息", interactive=False)
                    image_output_con = gr.Gallery(label="生成结果", interactive=False)
                    stop_button_con = gr.Button("中止生成", variant="stop")
        
    gr.on(
        triggers=[generate_button_t2i.click, prompt_t2i.submit],
        fn = generate_t2i,
        inputs = [
            prompt_t2i,
            width_t2i,
            height_t2i,
            num_inference_steps_t2i,
            batch_images_t2i,
            seed_param_t2i,
            lora_dropdown, 
            lora_weights,
        ],
        outputs = [image_output_t2i, info_t2i]
    )
    exchange_button_t2i.click(
        fn=exchange_width_height, 
        inputs=[width_t2i, height_t2i], 
        outputs=[width_t2i, height_t2i, info_t2i]
    )
    scale_1_5_button_t2i.click(
        fn=scale_resolution_1_5,
        inputs=[width_t2i, height_t2i],
        outputs=[width_t2i, height_t2i, info_t2i]
    )
    stop_button_t2i.click(
        fn=stop_generate, 
        inputs=[], 
        outputs=[info_t2i]
    )
    # ControlNet
    gr.on(
        triggers=[generate_button_con.click, prompt_con.submit],
        fn = generate_con,
        inputs = [
            image_con,
            prompt_con,
            width_con,
            height_con,
            num_inference_steps_con,
            strength_con,
            batch_images_con,
            seed_param_con,
            lora_dropdown, 
            lora_weights,
        ],
        outputs = [image_output_con, info_con]
    )
    exchange_button_con.click(
        fn=exchange_width_height, 
        inputs=[width_con, height_con], 
        outputs=[width_con, height_con, info_con]
    )
    scale_1_5_button_con.click(
        fn=scale_resolution_1_5,
        inputs=[width_con, height_con],
        outputs=[width_con, height_con, info_con]
    )
    stop_button_con.click(
        fn=stop_generate, 
        inputs=[], 
        outputs=[info_con]
    )
    image_con.upload(
        fn=adjust_width_height, 
        inputs=[image_con], 
        outputs=[width_con, height_con, info_con]
    )

if __name__ == "__main__": 
    demo.launch(
        server_name=args.server_name, 
        server_port=find_port(args.server_port),
        share=args.share, 
        mcp_server=args.mcp_server,
        inbrowser=True,
    )