import re
import os
import json
import time
import praw
import openai
import replicate
from elevenlabs import set_api_key, generate
import urllib.request
from moviepy.editor import ImageClip, concatenate_videoclips, AudioFileClip, VideoFileClip, TextClip, CompositeVideoClip
from datetime import datetime
import google.oauth2.credentials
import google_auth_oauthlib.flow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Constants
BASE_DIR = "reddit_shorts"
POSTS_DIR = os.path.join(BASE_DIR, "posts")
IMAGES_DIR = os.path.join(BASE_DIR, "images")
AUDIO_DIR = os.path.join(BASE_DIR, "audio")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# Create base directories
for directory in [BASE_DIR, POSTS_DIR, IMAGES_DIR, AUDIO_DIR, OUTPUT_DIR]:
    os.makedirs(directory, exist_ok=True)

# Initialize API clients
client = replicate.Client(api_token=os.environ["STABLEDIFFUSION_API_KEY"])
set_api_key(os.environ["ELEVENLABS_API_KEY"])
openai.api_key = os.environ['OPENAI_API_KEY']

reddit = praw.Reddit(
    client_id=os.environ['REDDIT_CLIENT_ID'],
    client_secret=os.environ['REDDIT_CLIENT_SECRET'],
    user_agent=os.environ['REDDIT_USER_AGENT']
)

# YouTube API setup
SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]
API_SERVICE_NAME = "youtube"
API_VERSION = "v3"
CLIENT_SECRETS_FILE = "client_secrets.json"

def get_authenticated_service():
    flow = google_auth_oauthlib.flow.InstalledAppFlow.from_client_secrets_file(
        CLIENT_SECRETS_FILE, SCOPES)
    credentials = flow.run_local_server(port=0)
    return googleapiclient.discovery.build(
        API_SERVICE_NAME, API_VERSION, credentials=credentials)

def preprocess_text(text):
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[\*\]', '', text)
    text = re.sub(r'^\d+\s+', '', text, flags=re.MULTILINE)
    return text

def generate_title(prompt, script):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": script}
            ],
            temperature=1,
            max_tokens=256
        )
        return response.choices[0].message['content']
    except Exception as e:
        print(f"Error generating title: {e}")
        return None

def generate_image_prompts(gpt_type, prompt, script):
    try:
        response = openai.ChatCompletion.create(
            model=gpt_type,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": script}
            ],
            temperature=1,
            max_tokens=1024
        )
        image_prompts_str = response.choices[0].message['content']
        image_prompts_str = re.sub(r'[^\w\s]', '', image_prompts_str)
        image_prompts = image_prompts_str.lower().split('\n')
        return [prompt.strip() for prompt in image_prompts if prompt.strip()]
    except Exception as e:
        print(f"Error generating image prompts: {e}")
        return []

def generate_stablediffusion_image(image_prompt):
    try:
        output = client.run(
            "stability-ai/stable-diffusion:ac732df83cea7fff18b8472768c88ad041fa750ff7682a21affe81863cbe77e4",
            input={
                "prompt": image_prompt,
                "height": 1024,
                "width": 576,
                "num_outputs": 1,
                "num_inference_steps": 50,
                "guidance_scale": 7.5,
                "scheduler": 'DPMSolverMultistep',
                "seed": 42
            }
        )
        return output[0]
    except Exception as e:
        print(f"Error generating image: {e}")
        return None

def create_video(images_folder, audio_file_path, output_path, captions=None):
    try:
        # Get and sort image files (for JPG images)
        image_files = sorted([f for f in os.listdir(images_folder) if f.endswith('.jpg')])
        
        if not image_files:
            print("No JPG images found in the specified folder")
            return False
            
        image_paths = [os.path.join(images_folder, f) for f in image_files]

        # Load audio
        audio_clip = AudioFileClip(audio_file_path)
        total_duration = audio_clip.duration

        # Create video clips
        video_clips = []
        for i, image_path in enumerate(image_paths):
            # Calculate duration for this image
            if i < len(image_paths) - 1:
                duration = total_duration / len(image_paths)
            else:
                # Last image takes remaining time
                duration = total_duration - (duration * (len(image_paths) - 1))
            
            img_clip = ImageClip(image_path, duration=duration).resize((1080, 1920))
            
            # Add captions if provided
            if captions:
                # Find captions that should appear during this image's duration
                image_start = i * duration
                image_end = (i + 1) * duration
                
                for caption in captions:
                    if (caption['start'] >= image_start and caption['start'] < image_end) or \
                       (caption['end'] > image_start and caption['end'] <= image_end):
                        # Calculate caption timing relative to this image
                        caption_start = max(0, caption['start'] - image_start)
                        caption_duration = min(duration, caption['end'] - image_start)
                        
                        txt_clip = TextClip(caption['text'], fontsize=40, color='white',
                                          font='Arial', stroke_color='black', stroke_width=2)
                        txt_clip = txt_clip.set_position(('center', 1800))
                        txt_clip = txt_clip.set_start(caption_start)
                        txt_clip = txt_clip.set_duration(caption_duration)
                        
                        img_clip = CompositeVideoClip([img_clip, txt_clip])
            
            video_clips.append(img_clip)

        # Combine clips
        final_clip = concatenate_videoclips(video_clips, method="compose")
        final_clip = final_clip.set_audio(audio_clip)
        final_clip.fps = 24

        # Export video
        final_clip.write_videofile(output_path, codec='libx264', audio_codec='aac', threads=4)
        return True
    except Exception as e:
        print(f"Error creating video: {e}")
        return False

def upload_to_youtube(video_path, title, description, thumbnail_path=None):
    try:
        youtube = get_authenticated_service()
        
        request_body = {
            "snippet": {
                "title": title,
                "description": description,
                "tags": ["shorts", "reddit", "scary"],
                "categoryId": "22"
            },
            "status": {
                "privacyStatus": "public",
                "selfDeclaredMadeForKids": False
            }
        }

        media_file = googleapiclient.http.MediaFileUpload(
            video_path,
            mimetype="video/mp4",
            resumable=True
        )

        request = youtube.videos().insert(
            part=",".join(request_body.keys()),
            body=request_body,
            media_body=media_file
        )

        response = request.execute()
        video_id = response["id"]

        if thumbnail_path:
            youtube.thumbnails().set(
                videoId=video_id,
                media_body=googleapiclient.http.MediaFileUpload(thumbnail_path)
            ).execute()

        return video_id
    except Exception as e:
        print(f"Error uploading to YouTube: {e}")
        return None

def generate_captions(text, max_words_per_caption=5):
    words = text.split()
    captions = []
    for i in range(0, len(words), max_words_per_caption):
        caption = " ".join(words[i:i + max_words_per_caption])
        captions.append(caption)
    return captions

def process_post(post, post_type):
    try:
        # Create a unique identifier for the post
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        post_id = f"{post_type}_{timestamp}"
        
        # Create post-specific directories
        post_images_dir = os.path.join(IMAGES_DIR, post_id)
        post_audio_dir = os.path.join(AUDIO_DIR, post_id)
        post_output_dir = os.path.join(OUTPUT_DIR, post_id)
        
        os.makedirs(post_images_dir, exist_ok=True)
        os.makedirs(post_audio_dir, exist_ok=True)
        os.makedirs(post_output_dir, exist_ok=True)

        # Generate title and description
        post["short_title"] = generate_title(
            "Generate a short YouTube Short title",
            post["content"]
        )
        
        post["description"] = generate_title(
            "Generate a short YouTube description",
            post["content"]
        )
        
        # Generate images
        image_prompts = generate_image_prompts(
            "gpt-3.5-turbo",
            "Generate 6 short image prompts",
            post["content"]
        )
        
        # Generate and save images
        for i, prompt in enumerate(image_prompts):
            image_url = generate_stablediffusion_image(prompt)
            if image_url:
                urllib.request.urlretrieve(
                    image_url,
                    os.path.join(post_images_dir, f"image_{i}.jpg")
                )

        # Generate audio with word timings
        audio = generate(
            text=post["content"],
            voice="Nicole",
            model="eleven_monolingual_v1",
            return_timestamps=True  # Get word timings
        )
        
        audio_path = os.path.join(post_audio_dir, "audio.wav")
        with open(audio_path, 'wb') as f:
            f.write(audio['audio'])  # Save the audio data

        # Generate captions with timings
        words = post["content"].split()
        word_timings = audio['timestamps']
        captions = []
        current_caption = []
        current_start_time = 0
        
        for i, (word, timing) in enumerate(zip(words, word_timings)):
            current_caption.append(word)
            if len(current_caption) >= 5 or i == len(words) - 1:  # 5 words per caption or last word
                caption_text = " ".join(current_caption)
                caption_end_time = timing['end']
                captions.append({
                    'text': caption_text,
                    'start': current_start_time,
                    'end': caption_end_time
                })
                current_caption = []
                current_start_time = caption_end_time

        # Create video
        output_path = os.path.join(post_output_dir, f"video_{timestamp}.mp4")
        
        if create_video(
            post_images_dir,
            audio_path,
            output_path,
            captions
        ):
            print(f"Successfully created video for post: {post['title']}")
            
            # Upload to YouTube
            video_id = upload_to_youtube(
                output_path,
                post["short_title"],
                post["description"],
                os.path.join(post_images_dir, "image_0.jpg")  # Use first image as thumbnail
            )
            
            if video_id:
                print(f"Successfully uploaded to YouTube: {video_id}")
            
            return True
        return False
        
    except Exception as e:
        print(f"Error processing post {post['title']}: {e}")
        return False

def main():
    try:
        # Get Reddit posts
        subreddit = reddit.subreddit("shortscarystories")
        posts = []
        
        # Get top posts
        for post in subreddit.top(time_filter="day", limit=5):
            posts.append({
                "title": post.title,
                "content": preprocess_text(post.selftext),
                "type": "top"
            })
        
        # Get controversial posts
        for post in subreddit.controversial(time_filter="day", limit=5):
            posts.append({
                "title": post.title,
                "content": preprocess_text(post.selftext),
                "type": "controversial"
            })

        # Process each post
        for post in posts:
            process_post(post, post["type"])

    except Exception as e:
        print(f"Main error: {e}")

if __name__ == "__main__":
    main() 