from fastvideo import VideoGenerator

def main():
    # Create a video generator with a pre-trained model
    generator = VideoGenerator.from_pretrained(
        "/data/home/scyb091/model/Wan2.2-T2V-A14B-Diffusers/models--Wan-AI--Wan2.2-T2V-A14B-Diffusers/snapshots/5be7df9619b54f4e2667b2755bc6a756675b5cd7",
        num_gpus=1,  # Adjust based on your hardware
    )

    # Define a prompt for your video
    prompt = "CR7 is beating Messi."

    # Generate the video
    video = generator.generate_video(
        prompt,
        return_frames=True,  # Also return frames from this call (defaults to False)
        output_path="/data/home/scyb091/FastVideo/outputs/",  # Controls where videos are saved
        save_video=True
    )

if __name__ == '__main__':
    main()