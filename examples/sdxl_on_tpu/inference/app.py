import gradio as gr

import jax
import jax.numpy as jnp
import numpy as np

from flax.jax_utils import replicate
from diffusers import FlaxStableDiffusionXLPipeline

from flax.training.common_utils import shard
from jax import pmap

from jax.experimental.compilation_cache import compilation_cache as cc
cc.initialize_cache("/tmp/sdxl_cache")

import time

dtype = jnp.bfloat16
model_id = "pcuenq/stable-diffusion-xl-base-1.0-flax"

def to_bf16(t):
    return jax.tree_map(lambda x: x.astype(jnp.bfloat16) if x.dtype != jnp.bfloat16 else x, t)

def create_key(seed=0):
    return jax.random.PRNGKey(seed)

def get_pipeline_params(text_prompt = None, neg_text_prompt = "", seed = 0, guidance_scale = 9, steps = 25):
    rng = create_key(seed)
    rng = jax.random.split(rng, jax.device_count())

    prompt = text_prompt or 77 * "a"
    prompt = [prompt] * jax.device_count()
    prompt_ids = pipeline.prepare_inputs(prompt)
    prompt_ids = shard(prompt_ids)

    neg_prompt = neg_text_prompt
    neg_prompt = [neg_prompt] * jax.device_count()
    neg_prompt_ids = pipeline.prepare_inputs(neg_prompt)
    neg_prompt_ids = shard(neg_prompt_ids)

    num_inference_steps = steps
    height = 1024
    width = 1024 

    g = jnp.array([guidance_scale] * prompt_ids.shape[0], dtype=jnp.float32)
    g = g[:, None]

    return prompt_ids, rng, num_inference_steps, height, width, g, neg_prompt_ids

pipeline, params = FlaxStableDiffusionXLPipeline.from_pretrained(
    model_id,
    dtype=dtype,
)

params['vae'] = to_bf16(params['vae'])
params['text_encoder'] = to_bf16(params['text_encoder'])
params['text_encoder_2'] = to_bf16(params['text_encoder_2'])
params['unet'] = to_bf16(params['unet'])

p_params = replicate(params)

(prompt_ids, rng, num_inference_steps, 
 height, width, g, neg_prompt) = get_pipeline_params()

start_time = time.time()
p_generate = pmap(
    pipeline._generate, 
    static_broadcasted_argnums=[3, 4, 5, 9]
    ).lower(
        prompt_ids, 
        p_params, 
        rng, 
        num_inference_steps, 
        height, 
        width, 
        g, 
        None, 
        neg_prompt, 
        False).compile()
print("Compile time:", time.time() - start_time)
 
def generate(text_prompt, neg_text_prompt = "", seed = 0, guidance_scale = 9, steps = 25):
    print("Start...")
    print("Version", jax.__version__)

    (prompt_ids, rng, num_inference_steps, 
        _, _, g, neg_prompt) = get_pipeline_params(text_prompt, 
                                                neg_text_prompt, 
                                                seed, 
                                                guidance_scale, 
                                                steps)
    
    start_time = time.time()
    images = p_generate(prompt_ids, 
                        p_params, 
                        rng, 
                        g, 
                        None, 
                        neg_prompt)
    images = images.block_until_ready()
    end_time = time.time()

    print(f"For {num_inference_steps} steps", end_time - start_time)
    print("Avg per step", (end_time - start_time) / num_inference_steps)

    images = images.reshape((images.shape[0] * images.shape[1], ) + images.shape[-3:])
    images = pipeline.numpy_to_pil(np.array(images))
    total_time = (end_time - start_time)
    markdown=f"""
    *Benchmarks:*

    *Steps :* {steps}

    *No of images :* {jax.device_count()}

    *Generation time :* {total_time} seconds

    *Iters / sec :* {(steps * jax.device_count()) / total_time}

    """

    return images, markdown


with gr.Blocks(css="style.css") as demo:
    with gr.Box():
        with gr.Row().style(mobile_collapse=False, equal_height=True):
            prompt = gr.Textbox(
                label="Prompt",
                show_label=False,
                max_lines=1,
                placeholder="Enter your prompt"
            )
            negative_prompt = gr.Textbox(
                label="Negative prompt",
                show_label=False,
                max_lines=1,
                placeholder="Negative prompt"
            )
            generate_btn = gr.Button("Generate image").style(
                margin=False,
                rounded=(False, True, True, False),
                full_width=False,
            )
    time_taken = gr.Markdown(
        value = ""
    )
    gallery = gr.Gallery(
        label="Generated images", show_label=False
    ).style(grid=[2], height="auto")

    with gr.Row():
        scale = gr.Slider(label="Guidance Scale", minimum=0, maximum=13, value=9, step=0.1)
        seed = gr.Slider(label="Seed",
                         minimum=0,
                         maximum=2147483647,
                         step=1,
                         randomize=True)

    generate_btn.click(fn=generate, inputs=[prompt, negative_prompt, seed, scale], outputs=[gallery, time_taken])

demo.queue().launch(share=True)

