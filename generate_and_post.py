import os
import random
from diffusers import StableDiffusionXLPipeline  # ✅ Use SDXL pipeline!
import torch
import praw
from dotenv import load_dotenv

# Load local .env when running locally
load_dotenv()

def generate_image(
    prompt,
    model_id="stabilityai/stable-diffusion-xl-base-1.0",
    output_file="daily_generated_image.png"
):
    # Use float16 on GPU if available, else CPU fallback
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # ✅ Use SDXL pipeline
    pipe = StableDiffusionXLPipeline.from_pretrained(model_id, torch_dtype=dtype)
    pipe = pipe.to("cuda") if torch.cuda.is_available() else pipe.to("cpu")

    negative_prompt = (
        "blurry, low quality, frame, border, vignette, white margin"
    )

    # Generate phone wallpaper at 1080p-ish ratio
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=1920,   # typical phone height
        width=1080,    # typical phone width
        num_inference_steps=30,  # adjust as needed for quality vs speed
        guidance_scale=8.5
    ).images[0]

    image.save(output_file)
    print(f"✅ Image saved as {output_file}")
    return output_file


def post_to_reddit(image_path, subreddit_name="pexwalls", title="Some wallpaper"):
    reddit = praw.Reddit(
        client_id=os.getenv("REDDIT_CLIENT_ID"),
        client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
        username=os.getenv("REDDIT_USERNAME"),
        password=os.getenv("REDDIT_PASSWORD"),
        user_agent="script:DailyImage:1.0 (by /u/Competitive-Fee2930)"
    )

    subreddit = reddit.subreddit(subreddit_name)
    submission = subreddit.submit_image(title, image_path)
    print(f"✅ Posted! https://www.reddit.com{submission.permalink}")

def get_dynamic_title():
    main_titles = [
        "Nothing.", "Void.", "Fading.", "Oblivion.", "Silent.",
        "Ashes.", "Drift.", "Lost.", "Decay.", "Fade."
    ]

    sub_titles = [
        "Still here", "Endless", "No escape", "Never whole",
        "Cold sun", "Bleak light", "Stay quiet",
        "Soft collapse", "Into dust", "Sleep deep"
    ]

    chosen_main = random.choice(main_titles)
    chosen_sub = random.choice(sub_titles)
    combined_title = f"{chosen_main} {chosen_sub}."
    return combined_title

def main():
    wallpaper_prompts = [
        "A soft, pastel gradient phone wallpaper, smooth fade from lavender to peach, subtle grain, minimal aesthetic.",
        "A misty pine forest at dawn, muted green tones, cinematic wide shot, dreamy soft focus, wallpaper aspect ratio.",
        "Modern abstract geometric wallpaper, overlapping translucent circles, muted blues and warm beige, clean and minimal.",
        "A lone tree on a tiny island in a reflective lake under a huge full moon, calm night colors, high resolution.",
        "Cyberpunk cityscape wallpaper, neon lights, rainy street reflections, moody atmosphere, vibrant purple and teal.",
        "Galaxy-inspired abstract wallpaper, swirling colors of deep space, stars and nebulae, dark and luminous.",
        "Close-up wallpaper of delicate green ferns with morning dew, soft bokeh background, fresh and natural.",
        "A retro paper texture wallpaper with faded ink illustrations of flowers, sepia tones, old book feel.",
        "Minimal black and white ink painting of a single mountain peak with clouds, Japanese sumi-e style, calm and spacious.",
    ]

    prompt = random.choice(wallpaper_prompts)
    print("Prompt:", prompt)

    image_file = generate_image(prompt + " ultra fine")

    # Dynamic minimal title
    title = get_dynamic_title()
    post_to_reddit(image_file, title=title)

if __name__ == "__main__":
    main()