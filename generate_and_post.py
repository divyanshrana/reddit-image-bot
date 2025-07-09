import os
from diffusers import StableDiffusionPipeline
import torch
import praw

def generate_image(prompt, model_id="stabilityai/stable-diffusion-2-1", output_file="daily_generated_image.png"):
    # Pick torch dtype based on GPU
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # Load pipeline
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)

    # Use GPU if available
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
    else:
        pipe = pipe.to("cpu")

    # Generate image
    image = pipe(prompt).images[0]
    image.save(output_file)
    print(f"✅ Image saved as {output_file}")

    return output_file

def post_to_reddit(image_path, subreddit_name="radwalls", title="Here’s today’s AI art! #StableDiffusion"):
    reddit = praw.Reddit(
        client_id=os.getenv("REDDIT_CLIENT_ID"),
        client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
        username=os.getenv("REDDIT_USERNAME"),
        password=os.getenv("REDDIT_PASSWORD"),
        user_agent="script:DailyImageBot:1.0 (by /u/Competitive-Fee2930)"
    )

    subreddit = reddit.subreddit(subreddit_name)
    submission = subreddit.submit_image(title, image_path)
    print(f"✅ Posted! https://www.reddit.com{submission.permalink}")

def main():
    prompt = (
        "Create a grainy, vintage black-and-white poster with a worn, textured background. "
        "Add Japanese kanji at the top and any nihilistic English words "
        "below. Include delicate white cherry blossom branches rising from the bottom, detailed and "
        "subtle. Use an old typewriter or film title font. The mood should feel dark, poetic, and cinematic."
    )

    image_file = generate_image(prompt)
    post_to_reddit(image_file)

if __name__ == "__main__":
    main()