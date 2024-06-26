import sys
import os
sys.path.append(os.path.abspath(__file__).rsplit('/', 2)[0])

from model.meta import MetaModel

import argparse
import torch
import torch.distributed as dist
import gradio as gr

from PIL import Image

from util import misc
from fairscale.nn.model_parallel import initialize as fs_init

from data.alpaca import transform_val, format_prompt
from util.tensor_parallel import load_tensor_parallel_model_list
from util.quant import quantize


def get_args_parser():
    parser = argparse.ArgumentParser('Single-turn (conversation) demo', add_help=False)
    # Model parameters
    parser.add_argument('--llama_type', default='llama_qformerv2_peft', type=str, metavar='MODEL',
                        help='type of llama')
    parser.add_argument('--llama_config', default='/data/home/zhangrongyu/code/MMLLaMA2/checkpoints/llama2/Llama-2-13b/params.json configs/model/finetune/sg/llamaPeft_normBiasLora.json', type=str, nargs="+",
                        help='Path to llama model config')
    parser.add_argument('--tokenizer_path', type=str, default="/data/home/zhangrongyu/code/MMLLaMA2/checkpoints/llama2/Llama-2-13b/tokenizer.model",
                        help='path to tokenizer.model')

    parser.add_argument('--pretrained_path', default='/data/home/zhangrongyu/code/MMLLaMA2/checkpoints/LLaMA2-Accessory-hf/finetune/mm/alpacaLlava_llamaQformerv2Peft_13b/', type=str, nargs="+",
                        help='directory containing pre-trained checkpoints')

    parser.add_argument('--device', default='cuda',
                        help='device for inference')
    parser.add_argument('--model_parallel_size', default=1, type=int)

    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--quant', action="store_true", default=False,
                        help="enable quantization")
    return parser

args = get_args_parser().parse_args()
args.pretrained_path = ['/data/home/zhangrongyu/code/MMLLaMA2/checkpoints/LLaMA2-Accessory-hf/finetune/mm/alpacaLlava_llamaQformerv2Peft_13b/']
args.llama_config = ['/data/home/zhangrongyu/code/MMLLaMA2/checkpoints/llama2/Llama-2-13b/params.json', '/data/home/zhangrongyu/code/MMLLaMA2/accessory/configs/model/finetune/sg/llamaPeft_normBiasLora.json']

# define the model
misc.init_distributed_mode(args)
#torch.distributed.init()
#print(torch.distributed.is_initialized())
fs_init.initialize_model_parallel(args.model_parallel_size)
model = MetaModel(args.llama_type, args.llama_config, args.tokenizer_path, with_visual=True)
print(f"load pretrained from {args.pretrained_path}")
load_tensor_parallel_model_list(model, args.pretrained_path)

if args.quant:
    print("Quantizing model to 4bit!")

    from transformers.utils.quantization_config import BitsAndBytesConfig
    quantization_config = BitsAndBytesConfig.from_dict(
        config_dict={
            "load_in_8bit": False,
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
        },
        return_unused_kwargs=False,
    )
    quantize(model, quantization_config)

print("Model = %s" % str(model))
model.bfloat16().cuda()


@ torch.inference_mode()
def generate(
        img_path,
        prompt,
        question_input,
        max_gen_len,
        gen_t, top_p
):
    if img_path is not None:
        from PIL import ImageFile
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        image = Image.open(img_path).convert('RGB')
        image = transform_val(image).unsqueeze(0)
    else:
        image = None

    # text output
    _prompt = format_prompt(prompt, question_input)

    dist.barrier()
    dist.broadcast_object_list([_prompt, image, max_gen_len, gen_t, top_p])

    if image is not None:
        image = image.cuda()
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        results = model.generate([_prompt], image, max_gen_len=max_gen_len, temperature=gen_t, top_p=top_p)
    text_output = results[0].strip()
    print(text_output)
    return text_output

def create_demo():
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    with gr.Column() as image_input:
                        img_path = gr.Image(label='Image Input', type='filepath')
            with gr.Column():
                with gr.Row():
                    prompt = gr.Textbox(lines=4, label="Question")
                with gr.Row():
                    question_input = gr.Textbox(lines=4, label="Question Input (Optional)")
                with gr.Row() as text_config_row:
                    max_gen_len = gr.Slider(minimum=1, maximum=512, value=128, interactive=True, label="Max Length")
                    # with gr.Accordion(label='Advanced options', open=False):
                    gen_t = gr.Slider(minimum=0, maximum=1, value=0.1, interactive=True, label="Temperature")
                    top_p = gr.Slider(minimum=0, maximum=1, value=0.75, interactive=True, label="Top p")
                with gr.Row():
                    # clear_botton = gr.Button("Clear")
                    run_botton = gr.Button("Run", variant='primary')

                with gr.Row():
                    gr.Markdown("Output")
                with gr.Row():
                    text_output = gr.Textbox(lines=11, label='Text Out')

    inputs = [
        img_path,
        prompt, question_input,
        max_gen_len, gen_t, top_p,
    ]
    outputs = [text_output]
    run_botton.click(fn=generate, inputs=inputs, outputs=outputs)

    return demo


def worker_func():
    while True:
        dist.barrier()

        input_data = [None for _ in range(5)]
        dist.broadcast_object_list(input_data)
        _prompt, image, max_gen_len, gen_t, top_p = input_data
        if image is not None:
            image = image.cuda()
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            _ = model.generate([_prompt], image, max_gen_len=max_gen_len, temperature=gen_t, top_p=top_p, )
