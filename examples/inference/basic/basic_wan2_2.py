from fastvideo import VideoGenerator
import time

# from fastvideo.configs.sample import SamplingParam

OUTPUT_PATH = "/workspace/outputs"
def main():
    # FastVideo will automatically use the optimal default arguments for the
    # model.
    # If a local path is provided, FastVideo will make a best effort
    # attempt to identify the optimal arguments.
    st2 = time.time()
    generator = VideoGenerator.from_pretrained(
        #"Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        # FastVideo will automatically handle distributed setup
        model_path="/workspace/models/Wan/Wan2_2_T2V_14B_Diffusers",
        num_gpus=8,
        use_fsdp_inference=True,
        dit_cpu_offload=False, # DiT need to be offloaded for MoE
        vae_cpu_offload=False,
        text_encoder_cpu_offload=False,
        # Set pin_cpu_memory to false if CPU RAM is limited and there're no frequent CPU-GPU transfer
        pin_cpu_memory=True,
        # image_encoder_cpu_offload=False,
    )
    ed2 = time.time()
    print(f"Time taken to initialize generator: {ed2 - st2} seconds")   

    # sampling_param = SamplingParam.from_pretrained("Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    # sampling_param.num_frames = 45
    # sampling_param.image_path = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg"
    # Generate videos with the same simple API, regardless of GPU count
    prompt = (
        "A curious raccoon peers through a vibrant field of yellow sunflowers, its eyes "
        "wide with interest. The playful yet serene atmosphere is complemented by soft "
        "natural light filtering through the petals. Mid-shot, warm and cheerful tones."
    )
    st1 = time.time()
    _ = generator.generate_video(prompt, output_path=OUTPUT_PATH, save_video=True, height=720, width=1280, num_frames=81,
                                 output_video_name="test.mp4")
    ed1 = time.time()
    print(f"Time taken to generate video: {ed1 - st1} seconds")
    # video = generator.generate_video(prompt, sampling_param=sampling_param, output_path="wan_t2v_videos/")

    # Generate another video with a different prompt, without reloading the
    # model!
    '''prompt2 = (
        "A majestic lion strides across the golden savanna, its powerful frame "
        "glistening under the warm afternoon sun. The tall grass ripples gently in "
        "the breeze, enhancing the lion's commanding presence. The tone is vibrant, "
        "embodying the raw energy of the wild. Low angle, steady tracking shot, "
        "cinematic.")
    _ = generator.generate_video(prompt2, output_path=OUTPUT_PATH, save_video=True, height=720, width=1280, num_frames=81)'''


if __name__ == "__main__":
    st = time.time()
    main()
    ed = time.time()
    print(f"Total time taken: {ed - st} seconds")