#!/usr/bin/env python3

import os
import json
import logging
import argparse
from typing import List, Dict, Optional, Union, Tuple
from dataclasses import dataclass
from datetime import datetime
import openai
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
from moviepy.editor import (
    VideoFileClip,
    AudioFileClip,
    TextClip,
    CompositeVideoClip,
    concatenate_videoclips,
    ImageClip
)
from gtts import gTTS
import requests
from transformers import pipeline
from langdetect import detect
from googletrans import Translator
import gradio as gr
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class VideoConfig:
    """Configuration class for video generation"""
    script: str
    video_type: str
    platform: str
    voice_type: str
    music_genre: str
    length: int
    branding: Dict
    language: str = "en"
    style: str = "modern"
    resolution: Tuple[int, int] = (1920, 1080)
    fps: int = 30

class VideoType:
    """Supported video types"""
    TALKING_HEAD = "talking_head"
    EXPLAINER = "explainer"
    SLIDESHOW = "slideshow"
    STOCK_FOOTAGE = "stock_footage"
    TUTORIAL = "tutorial"
    PROMOTIONAL = "promotional"

class Platform:
    """Supported platforms"""
    YOUTUBE = "youtube"
    INSTAGRAM = "instagram"
    TIKTOK = "tiktok"
    LINKEDIN = "linkedin"
    FACEBOOK = "facebook"

class VideoCreator:
    def __init__(self, api_keys: Dict[str, str] = None):
        """Initialize video creator with API keys"""
        self.api_keys = api_keys or self._load_api_keys()
        self._init_apis()
        
        # Initialize AI models
        self.text_generator = pipeline("text-generation")
        self.translator = Translator()
        
        # Load templates and assets
        self.templates = self._load_templates()
        self.assets = self._load_assets()
        
    def _load_api_keys(self) -> Dict[str, str]:
        """Load API keys from configuration file"""
        try:
            with open("config/video_creator_keys.json", "r") as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning("API keys file not found. Some features may be limited.")
            return {}

    def _init_apis(self) -> None:
        """Initialize API clients"""
        if self.api_keys.get("openai"):
            openai.api_key = self.api_keys["openai"]
            
        # Initialize other API clients as needed

    def generate_video_content(
        self,
        script: str,
        video_type: str,
        platform: str
    ) -> str:
        """Generate video content based on script and type"""
        logger.info(f"Generating {video_type} video for {platform}")
        
        # Validate video type and platform
        if video_type not in vars(VideoType).values():
            raise ValueError(f"Unsupported video type: {video_type}")
            
        if platform not in vars(Platform).values():
            raise ValueError(f"Unsupported platform: {platform}")
            
        # Generate video based on type
        if video_type == VideoType.TALKING_HEAD:
            return self._generate_talking_head_video(script, platform)
        elif video_type == VideoType.EXPLAINER:
            return self._generate_explainer_video(script, platform)
        elif video_type == VideoType.SLIDESHOW:
            return self._generate_slideshow_video(script, platform)
        elif video_type == VideoType.STOCK_FOOTAGE:
            return self._generate_stock_footage_video(script, platform)
        elif video_type == VideoType.TUTORIAL:
            return self._generate_tutorial_video(script, platform)
        elif video_type == VideoType.PROMOTIONAL:
            return self._generate_promotional_video(script, platform)
            
    def add_voiceover(self, script: str, voice_type: str) -> str:
        """Generate and add voiceover to video"""
        try:
            # Parse voice type
            gender, speed, accent = self._parse_voice_type(voice_type)
            
            # Generate voiceover using gTTS
            tts = gTTS(
                text=script,
                lang=self._get_language_code(accent),
                slow=(speed == "slow")
            )
            
            # Save audio file
            audio_path = f"temp/voiceover_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3"
            os.makedirs("temp", exist_ok=True)
            tts.save(audio_path)
            
            return audio_path
            
        except Exception as e:
            logger.error(f"Error generating voiceover: {str(e)}")
            raise

    def integrate_images_and_video(
        self,
        media_files: List[str],
        script: str
    ) -> str:
        """Integrate images and videos with script"""
        try:
            clips = []
            
            # Split script into segments
            segments = self._split_script_into_segments(script)
            
            # Match media with script segments
            for i, (segment, media) in enumerate(zip(segments, media_files)):
                if media.endswith(('.jpg', '.png', '.jpeg')):
                    # Create image clip
                    clip = ImageClip(media).set_duration(5)
                    # Add text overlay
                    text_clip = TextClip(
                        segment,
                        fontsize=30,
                        color='white',
                        bg_color='black',
                        size=(clip.w, None),
                        method='caption'
                    ).set_duration(5)
                    # Composite image and text
                    clip = CompositeVideoClip([clip, text_clip])
                else:
                    # Load video clip
                    clip = VideoFileClip(media)
                    # Add text overlay
                    text_clip = TextClip(
                        segment,
                        fontsize=30,
                        color='white',
                        bg_color='black',
                        size=(clip.w, None),
                        method='caption'
                    ).set_duration(clip.duration)
                    # Composite video and text
                    clip = CompositeVideoClip([clip, text_clip])
                    
                clips.append(clip)
                
            # Concatenate all clips
            final_clip = concatenate_videoclips(clips)
            
            # Save final video
            output_path = f"output/video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            os.makedirs("output", exist_ok=True)
            final_clip.write_videofile(output_path)
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error integrating media: {str(e)}")
            raise

    def add_text_overlays(self, text: str, video: str) -> str:
        """Add text overlays to video"""
        try:
            # Load video
            clip = VideoFileClip(video)
            
            # Create text clip
            text_clip = TextClip(
                text,
                fontsize=30,
                color='white',
                bg_color='black',
                size=(clip.w, None),
                method='caption'
            ).set_duration(clip.duration)
            
            # Composite video and text
            final_clip = CompositeVideoClip([clip, text_clip])
            
            # Save final video
            output_path = f"output/video_with_text_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            os.makedirs("output", exist_ok=True)
            final_clip.write_videofile(output_path)
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error adding text overlays: {str(e)}")
            raise

    def add_background_music(self, video: str, genre: str) -> str:
        """Add background music to video"""
        try:
            # Load video
            video_clip = VideoFileClip(video)
            
            # Get music file
            music_path = self._get_music_file(genre)
            music_clip = AudioFileClip(music_path)
            
            # Loop music if needed
            if music_clip.duration < video_clip.duration:
                music_clip = music_clip.loop(duration=video_clip.duration)
            else:
                music_clip = music_clip.subclip(0, video_clip.duration)
                
            # Set music volume
            music_clip = music_clip.volumex(0.3)
            
            # Combine video and music
            final_clip = video_clip.set_audio(music_clip)
            
            # Save final video
            output_path = f"output/video_with_music_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            os.makedirs("output", exist_ok=True)
            final_clip.write_videofile(output_path)
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error adding background music: {str(e)}")
            raise

    def add_sound_effects(self, video: str, effect_type: str) -> str:
        """Add sound effects to video"""
        try:
            # Load video
            video_clip = VideoFileClip(video)
            
            # Get sound effect
            effect_path = self._get_sound_effect(effect_type)
            effect_clip = AudioFileClip(effect_path)
            
            # Add effect at specific points
            # This is a simplified version - in reality, you'd want to
            # analyze the video content to determine effect placement
            final_clip = video_clip.set_audio(
                CompositeVideoClip([
                    video_clip.audio,
                    effect_clip.set_start(2).set_duration(1)
                ])
            )
            
            # Save final video
            output_path = f"output/video_with_effects_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            os.makedirs("output", exist_ok=True)
            final_clip.write_videofile(output_path)
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error adding sound effects: {str(e)}")
            raise

    def apply_video_transitions(self, video: str, transition_type: str) -> str:
        """Apply transitions to video"""
        try:
            # Load video
            clip = VideoFileClip(video)
            
            # Apply transition effect
            if transition_type == "fade":
                clip = clip.fadein(1).fadeout(1)
            elif transition_type == "crossfade":
                # For crossfade, we need two clips
                clip1 = clip.subclip(0, clip.duration/2)
                clip2 = clip.subclip(clip.duration/2)
                clip = concatenate_videoclips([
                    clip1,
                    clip2.crossfadein(1)
                ])
            elif transition_type == "slide":
                # Slide transition
                clip = clip.slide_in(duration=1, side='left')
                
            # Save final video
            output_path = f"output/video_with_transitions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            os.makedirs("output", exist_ok=True)
            clip.write_videofile(output_path)
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error applying transitions: {str(e)}")
            raise

    def add_animations_and_effects(self, video: str) -> str:
        """Add animations and effects to video"""
        try:
            # Load video
            clip = VideoFileClip(video)
            
            # Apply various effects
            # Resize effect
            clip = clip.resize(width=clip.w * 1.1)
            
            # Color effect
            clip = clip.fx(vfx.colorx, 1.2)
            
            # Mirror effect
            clip = clip.fx(vfx.mirror_x)
            
            # Save final video
            output_path = f"output/video_with_effects_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            os.makedirs("output", exist_ok=True)
            clip.write_videofile(output_path)
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error adding animations: {str(e)}")
            raise

    def format_video_for_platform(self, video: str, platform: str) -> str:
        """Format video for specific platform"""
        try:
            # Load video
            clip = VideoFileClip(video)
            
            # Get platform-specific settings
            settings = self._get_platform_settings(platform)
            
            # Resize video
            clip = clip.resize(settings["resolution"])
            
            # Adjust duration if needed
            if clip.duration > settings["max_duration"]:
                clip = clip.subclip(0, settings["max_duration"])
                
            # Add platform-specific overlays
            if platform == Platform.YOUTUBE:
                clip = self._add_youtube_overlay(clip)
            elif platform == Platform.INSTAGRAM:
                clip = self._add_instagram_overlay(clip)
            elif platform == Platform.TIKTOK:
                clip = self._add_tiktok_overlay(clip)
                
            # Save final video
            output_path = f"output/video_{platform}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            os.makedirs("output", exist_ok=True)
            clip.write_videofile(output_path)
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error formatting video: {str(e)}")
            raise

    def apply_branding_and_style(self, video: str, branding_info: Dict) -> str:
        """Apply branding and style to video"""
        try:
            # Load video
            clip = VideoFileClip(video)
            
            # Add logo
            if "logo_path" in branding_info:
                logo = ImageClip(branding_info["logo_path"])
                logo = logo.set_duration(clip.duration)
                logo = logo.set_position(("right", "bottom"))
                clip = CompositeVideoClip([clip, logo])
                
            # Add color overlay
            if "color_scheme" in branding_info:
                color_clip = ColorClip(
                    size=clip.size,
                    color=branding_info["color_scheme"],
                    opacity=0.3
                ).set_duration(clip.duration)
                clip = CompositeVideoClip([clip, color_clip])
                
            # Add text with brand font
            if "font_path" in branding_info:
                font = branding_info["font_path"]
            else:
                font = "Arial"
                
            if "tagline" in branding_info:
                text_clip = TextClip(
                    branding_info["tagline"],
                    fontsize=30,
                    font=font,
                    color='white',
                    bg_color='black',
                    size=(clip.w, None),
                    method='caption'
                ).set_duration(clip.duration)
                text_clip = text_clip.set_position(("center", "bottom"))
                clip = CompositeVideoClip([clip, text_clip])
                
            # Save final video
            output_path = f"output/video_branded_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            os.makedirs("output", exist_ok=True)
            clip.write_videofile(output_path)
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error applying branding: {str(e)}")
            raise

    def adjust_video_length(self, video: str, target_length: int) -> str:
        """Adjust video length to target duration"""
        try:
            # Load video
            clip = VideoFileClip(video)
            
            if clip.duration > target_length:
                # Trim video
                clip = clip.subclip(0, target_length)
            elif clip.duration < target_length:
                # Extend video by looping
                n_loops = int(np.ceil(target_length / clip.duration))
                clip = concatenate_videoclips([clip] * n_loops)
                clip = clip.subclip(0, target_length)
                
            # Save final video
            output_path = f"output/video_length_{target_length}s_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            os.makedirs("output", exist_ok=True)
            clip.write_videofile(output_path)
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error adjusting video length: {str(e)}")
            raise

    def analyze_video_performance(self, video: str, platform: str) -> Dict:
        """Analyze video performance"""
        try:
            # Get video metrics
            metrics = self._get_video_metrics(video, platform)
            
            # Analyze engagement
            engagement_rate = self._calculate_engagement_rate(metrics)
            
            # Analyze viewer retention
            retention_data = self._analyze_viewer_retention(metrics)
            
            # Generate suggestions
            suggestions = self._generate_video_suggestions(metrics)
            
            return {
                "engagement_rate": engagement_rate,
                "viewer_retention": retention_data,
                "suggestions": suggestions,
                "metrics": metrics
            }
            
        except Exception as e:
            logger.error(f"Error analyzing video: {str(e)}")
            raise

    def translate_video_content(self, script: str, target_language: str) -> str:
        """Translate video content to target language"""
        try:
            # Detect source language
            source_lang = detect(script)
            
            # Translate script
            translated = self.translator.translate(
                script,
                src=source_lang,
                dest=target_language
            ).text
            
            # Generate voiceover in target language
            voiceover_path = self.add_voiceover(
                translated,
                f"default,normal,{target_language}"
            )
            
            return {
                "translated_script": translated,
                "voiceover_path": voiceover_path
            }
            
        except Exception as e:
            logger.error(f"Error translating content: {str(e)}")
            raise

    def apply_template(self, template_name: str, video_content: str) -> str:
        """Apply video template"""
        try:
            # Load template
            template = self.templates.get(template_name)
            if not template:
                raise ValueError(f"Template not found: {template_name}")
                
            # Apply template to video
            clip = VideoFileClip(video_content)
            
            # Apply template effects
            if "transitions" in template:
                clip = self.apply_video_transitions(clip, template["transitions"])
                
            if "overlays" in template:
                for overlay in template["overlays"]:
                    clip = self._add_template_overlay(clip, overlay)
                    
            if "music" in template:
                clip = self.add_background_music(clip, template["music"])
                
            # Save final video
            output_path = f"output/video_template_{template_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            os.makedirs("output", exist_ok=True)
            clip.write_videofile(output_path)
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error applying template: {str(e)}")
            raise

    def integrate_with_other_modules(self, video: str, integration_type: str) -> str:
        """Integrate with other Hexa modules"""
        try:
            if integration_type == "blog":
                # Generate blog post from video
                from blog_generator import BlogGenerator
                blog_gen = BlogGenerator()
                blog_content = blog_gen.generate_blog_content(
                    "Video Summary",
                    ["video", "summary"],
                    500,
                    "formal"
                )
                return {
                    "video": video,
                    "blog_content": blog_content
                }
                
            elif integration_type == "social":
                # Generate social media posts
                from social_reporter import SocialReporter
                social = SocialReporter()
                posts = social.repurpose_content_for_social_media(
                    "Check out our new video!"
                )
                return {
                    "video": video,
                    "social_posts": posts
                }
                
            else:
                raise ValueError(f"Unknown integration type: {integration_type}")
                
        except Exception as e:
            logger.error(f"Error integrating with modules: {str(e)}")
            raise

    def generate_script(self, topic: str) -> str:
        """Generate video script from topic"""
        try:
            # Create prompt for script generation
            prompt = f"""Create a video script about {topic}.
Include an introduction, main points, and conclusion.
Make it engaging and informative.
Script:
"""
            
            # Generate script using OpenAI
            response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=prompt,
                max_tokens=500,
                temperature=0.7,
                top_p=1.0,
                frequency_penalty=0.5,
                presence_penalty=0.3
            )
            
            script = response.choices[0].text.strip()
            
            return script
            
        except Exception as e:
            logger.error(f"Error generating script: {str(e)}")
            raise

    def export_video(self, video: str, format: str) -> str:
        """Export video in specified format"""
        try:
            # Load video
            clip = VideoFileClip(video)
            
            # Export in specified format
            output_path = f"output/video_{format}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{format}"
            os.makedirs("output", exist_ok=True)
            
            if format == "mp4":
                clip.write_videofile(output_path)
            elif format == "avi":
                clip.write_videofile(output_path, codec='libx264')
            elif format == "mov":
                clip.write_videofile(output_path, codec='libx264')
            else:
                raise ValueError(f"Unsupported format: {format}")
                
            return output_path
            
        except Exception as e:
            logger.error(f"Error exporting video: {str(e)}")
            raise

def main():
    """Main function to run the video creator from command line"""
    parser = argparse.ArgumentParser(description="Hexa Assistant Video Creator")
    parser.add_argument("--script", required=True, help="Video script")
    parser.add_argument("--video-type", required=True, help="Type of video")
    parser.add_argument("--platform", required=True, help="Target platform")
    parser.add_argument("--voice", default="female,normal,en", help="Voice type")
    parser.add_argument("--music", default="ambient", help="Music genre")
    parser.add_argument("--length", type=int, help="Target length in seconds")
    parser.add_argument("--branding", help="Path to branding JSON file")
    
    args = parser.parse_args()
    
    creator = VideoCreator()
    
    # Load branding if provided
    branding = {}
    if args.branding:
        with open(args.branding, "r") as f:
            branding = json.load(f)
            
    # Generate video
    video = creator.generate_video_content(
        args.script,
        args.video_type,
        args.platform
    )
    
    # Add voiceover
    video = creator.add_voiceover(args.script, args.voice)
    
    # Add background music
    video = creator.add_background_music(video, args.music)
    
    # Apply branding
    if branding:
        video = creator.apply_branding_and_style(video, branding)
        
    # Adjust length if specified
    if args.length:
        video = creator.adjust_video_length(video, args.length)
        
    # Format for platform
    video = creator.format_video_for_platform(video, args.platform)
    
    print(f"Video created successfully: {video}")

if __name__ == "__main__":
    main()
