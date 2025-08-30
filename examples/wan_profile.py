from fastvideo import VideoGenerator

def main():
    # Create a video generator with a pre-trained model
    generator = VideoGenerator.from_pretrained(
        "/workspace/models/Wan/Wan2_2_T2V_14B_Diffusers",
        num_gpus=8,  # Adjust based on your hardware
    )

    # Define a prompt for your video
    prompt = "Trump is beating Biden."

    # Generate the video
    video = generator.generate_video(
        prompt,
        return_frames=True,  # Also return frames from this call (defaults to False)
        output_path="/workspace/outputs",  # Controls where videos are saved
        save_video=True
    )

if __name__ == '__main__':
    main()