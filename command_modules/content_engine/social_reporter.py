#!/usr/bin/env python3

import os
import json
import logging
import argparse
from typing import List, Dict, Optional, Union
from dataclasses import dataclass
from datetime import datetime, timezone
import pytz
import openai
import tweepy
import facebook
import instabot
import linkedin
import tiktok
from transformers import pipeline
from textblob import TextBlob
import schedule
import time
import requests
from PIL import Image
import numpy as np
from moviepy.editor import VideoFileClip
import pandas as pd
from langdetect import detect
from googletrans import Translator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class SocialMediaConfig:
    """Configuration class for social media content generation"""
    platform: str
    topic: str
    tone: str
    length: int
    target_audience: str
    hashtag_count: int
    include_media: bool
    schedule_time: Optional[str] = None
    time_zone: str = "UTC"
    language: str = "en"

class Platform:
    """Supported social media platforms"""
    TWITTER = "twitter"
    FACEBOOK = "facebook"
    INSTAGRAM = "instagram"
    LINKEDIN = "linkedin"
    TIKTOK = "tiktok"

class ContentTone:
    """Available content tones"""
    FORMAL = "formal"
    CASUAL = "casual"
    PROFESSIONAL = "professional"
    TRENDY = "trendy"
    INSPIRATIONAL = "inspirational"

class SocialReporter:
    def __init__(self, api_keys: Dict[str, str] = None):
        """Initialize social media reporter with API keys"""
        self.api_keys = api_keys or self._load_api_keys()
        self._init_platform_apis()
        
        # Initialize AI models
        self.sentiment_analyzer = pipeline("sentiment-analysis")
        self.text_generator = pipeline("text-generation")
        self.translator = Translator()
        
        # Load platform-specific configurations
        self.platform_configs = self._load_platform_configs()
        
    def _load_api_keys(self) -> Dict[str, str]:
        """Load API keys from configuration file"""
        try:
            with open("config/social_media_keys.json", "r") as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning("API keys file not found. Some features may be limited.")
            return {}

    def _init_platform_apis(self) -> None:
        """Initialize API clients for each platform"""
        if self.api_keys.get("twitter"):
            auth = tweepy.OAuthHandler(
                self.api_keys["twitter"]["consumer_key"],
                self.api_keys["twitter"]["consumer_secret"]
            )
            auth.set_access_token(
                self.api_keys["twitter"]["access_token"],
                self.api_keys["twitter"]["access_token_secret"]
            )
            self.twitter_api = tweepy.API(auth)
            
        if self.api_keys.get("facebook"):
            self.facebook_api = facebook.GraphAPI(
                self.api_keys["facebook"]["access_token"]
            )
            
        # Initialize other platform APIs similarly
        
    def generate_social_media_content(
        self,
        platform: str,
        topic: str,
        tone: str,
        length: int
    ) -> str:
        """Generate platform-specific social media content"""
        logger.info(f"Generating content for {platform} about {topic}")
        
        # Validate platform and tone
        if platform.lower() not in vars(Platform).values():
            raise ValueError(f"Unsupported platform: {platform}")
            
        if tone.lower() not in vars(ContentTone).values():
            raise ValueError(f"Unsupported tone: {tone}")
            
        # Create AI prompt based on platform and tone
        prompt = self._create_platform_prompt(platform, topic, tone)
        
        try:
            # Generate content using OpenAI
            response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=prompt,
                max_tokens=length,
                temperature=0.7,
                top_p=1.0,
                frequency_penalty=0.5,
                presence_penalty=0.3
            )
            
            content = response.choices[0].text.strip()
            
            # Format content for platform
            content = self.format_content_for_platform(platform, content)
            
            # Add hashtags if appropriate
            if platform.lower() in [Platform.INSTAGRAM, Platform.TWITTER]:
                hashtags = self.suggest_hashtags(topic, platform)
                content = self._append_hashtags(content, hashtags, platform)
                
            return content
            
        except Exception as e:
            logger.error(f"Error generating content: {str(e)}")
            raise

    def schedule_post(
        self,
        content: str,
        platform: str,
        schedule_time: str,
        media_path: Optional[str] = None
    ) -> None:
        """Schedule a post for publishing"""
        try:
            # Parse schedule time
            schedule_dt = datetime.fromisoformat(schedule_time)
            
            # Create scheduling job
            def post_job():
                self._publish_to_platform(platform, content, media_path)
                
            schedule.every().day.at(schedule_dt.strftime("%H:%M")).do(post_job)
            
            logger.info(f"Post scheduled for {schedule_time} on {platform}")
            
        except Exception as e:
            logger.error(f"Error scheduling post: {str(e)}")
            raise

    def suggest_hashtags(self, topic: str, platform: str) -> List[str]:
        """Suggest relevant hashtags for the content"""
        # Get trending hashtags for platform
        trending = self._get_trending_hashtags(platform)
        
        # Get topic-specific hashtags
        topic_tags = self._generate_topic_hashtags(topic)
        
        # Combine and filter based on platform limits
        all_tags = list(set(trending + topic_tags))
        
        platform_limits = {
            Platform.INSTAGRAM: 30,
            Platform.TWITTER: 5,
            Platform.TIKTOK: 10,
            Platform.FACEBOOK: 3,
            Platform.LINKEDIN: 3
        }
        
        return all_tags[:platform_limits.get(platform.lower(), 5)]

    def set_tone_and_style(self, content: str, tone: str) -> str:
        """Adjust content tone and style"""
        if tone not in vars(ContentTone).values():
            raise ValueError(f"Invalid tone: {tone}")
            
        tone_adjustments = {
            ContentTone.FORMAL: self._adjust_formal_tone,
            ContentTone.CASUAL: self._adjust_casual_tone,
            ContentTone.PROFESSIONAL: self._adjust_professional_tone,
            ContentTone.TRENDY: self._adjust_trendy_tone,
            ContentTone.INSPIRATIONAL: self._adjust_inspirational_tone
        }
        
        return tone_adjustments[tone](content)

    def generate_caption(
        self,
        content_type: str,
        platform: str,
        content: str
    ) -> str:
        """Generate platform-specific captions"""
        # Get platform-specific caption templates
        templates = self._get_caption_templates(platform)
        
        # Select appropriate template based on content type
        template = templates.get(content_type, templates["default"])
        
        # Generate caption using AI
        caption = self._generate_ai_caption(content, template)
        
        # Add call-to-action
        cta = self._generate_cta(platform, content_type)
        
        return f"{caption}\n\n{cta}"

    def generate_video_content(
        self,
        platform: str,
        content: str,
        video_type: str
    ) -> Dict[str, str]:
        """Generate video content suggestions"""
        if platform.lower() not in [Platform.TIKTOK, Platform.INSTAGRAM]:
            raise ValueError("Video content only supported for TikTok and Instagram")
            
        # Get trending audio and effects
        trending_audio = self._get_trending_audio(platform)
        trending_effects = self._get_trending_effects(platform)
        
        # Generate script and captions
        script = self._generate_video_script(content, video_type)
        captions = self._generate_video_captions(script)
        
        return {
            "script": script,
            "captions": captions,
            "suggested_audio": trending_audio[:5],
            "suggested_effects": trending_effects[:5],
            "posting_tips": self._generate_video_tips(platform)
        }

    def format_content_for_platform(self, platform: str, content: str) -> str:
        """Format content according to platform requirements"""
        platform = platform.lower()
        
        # Get platform character limits
        char_limits = {
            Platform.TWITTER: 280,
            Platform.LINKEDIN: 3000,
            Platform.FACEBOOK: 63206,
            Platform.INSTAGRAM: 2200,
            Platform.TIKTOK: 2200
        }
        
        # Truncate content if needed
        limit = char_limits.get(platform, 280)
        if len(content) > limit:
            content = content[:limit-3] + "..."
            
        # Apply platform-specific formatting
        if platform == Platform.TWITTER:
            content = self._format_for_twitter(content)
        elif platform == Platform.LINKEDIN:
            content = self._format_for_linkedin(content)
        elif platform == Platform.INSTAGRAM:
            content = self._format_for_instagram(content)
            
        return content

    def personalize_content(self, audience_type: str, content: str) -> str:
        """Personalize content for specific audience"""
        # Load audience personas
        personas = self._load_audience_personas()
        
        if audience_type not in personas:
            raise ValueError(f"Unknown audience type: {audience_type}")
            
        persona = personas[audience_type]
        
        # Adjust content based on persona preferences
        content = self._adjust_language_complexity(content, persona["language_level"])
        content = self._adjust_tone(content, persona["preferred_tone"])
        content = self._add_audience_specific_cta(content, persona["cta_preferences"])
        
        return content

    def repurpose_content_for_social_media(self, content: str) -> Dict[str, str]:
        """Repurpose content for different social platforms"""
        return {
            Platform.TWITTER: self._create_tweet_thread(content),
            Platform.LINKEDIN: self._create_linkedin_article(content),
            Platform.FACEBOOK: self._create_facebook_post(content),
            Platform.INSTAGRAM: self._create_instagram_carousel(content),
            Platform.TIKTOK: self._create_tiktok_script(content)
        }

    def post_performance_analysis(self, platform: str, post_id: str) -> Dict:
        """Analyze post performance"""
        metrics = self._get_post_metrics(platform, post_id)
        
        return {
            "engagement_rate": self._calculate_engagement_rate(metrics),
            "sentiment_analysis": self._analyze_post_sentiment(metrics["comments"]),
            "peak_engagement_times": self._find_peak_engagement_times(metrics),
            "improvement_suggestions": self._generate_improvement_suggestions(metrics),
            "best_posting_times": self._suggest_posting_times(platform, metrics)
        }

    def generate_trending_content(self, platform: str, trending_topic: str) -> str:
        """Generate content based on trending topics"""
        # Get trending data
        trend_data = self._get_trend_data(platform, trending_topic)
        
        # Generate relevant content
        content = self.generate_social_media_content(
            platform,
            trending_topic,
            ContentTone.TRENDY,
            self._get_optimal_length(platform)
        )
        
        # Add trending hashtags
        hashtags = self.suggest_hashtags(trending_topic, platform)
        content = self._append_hashtags(content, hashtags, platform)
        
        return content

    def translate_and_localize_content(
        self,
        content: str,
        language: str,
        region: str
    ) -> str:
        """Translate and localize content"""
        try:
            # Detect source language
            source_lang = detect(content)
            
            # Translate content
            translated = self.translator.translate(
                content,
                src=source_lang,
                dest=language
            ).text
            
            # Apply regional tone adjustments
            localized = self._apply_regional_tone(translated, region)
            
            return localized
            
        except Exception as e:
            logger.error(f"Translation error: {str(e)}")
            raise

    def moderate_content(self, content: str) -> bool:
        """Check content for policy violations"""
        # Check for inappropriate content
        if self._contains_inappropriate_content(content):
            return False
            
        # Check for scam patterns
        if self._detect_scam_patterns(content):
            return False
            
        # Check for platform-specific policy violations
        if self._violates_platform_policies(content):
            return False
            
        return True

    def auto_interact_with_users(
        self,
        post_id: str,
        action: str,
        platform: str
    ) -> None:
        """Automate user interactions"""
        try:
            if action == "reply":
                comments = self._get_post_comments(platform, post_id)
                for comment in comments:
                    response = self._generate_comment_response(comment)
                    if self.moderate_content(response):
                        self._post_reply(platform, post_id, comment["id"], response)
                        
            elif action == "like":
                self._auto_like_comments(platform, post_id)
                
            elif action == "engage":
                self._engage_with_users(platform, post_id)
                
        except Exception as e:
            logger.error(f"Error in user interaction: {str(e)}")
            raise

    def _create_platform_prompt(self, platform: str, topic: str, tone: str) -> str:
        """Create platform-specific prompt for content generation"""
        return f"""Create a {tone} {platform} post about {topic}.
Follow {platform}'s best practices and formatting.
Make it engaging and shareable.
Include appropriate calls to action.
"""

    def _get_trending_hashtags(self, platform: str) -> List[str]:
        """Get trending hashtags for platform"""
        if platform.lower() == Platform.TWITTER:
            return self._get_twitter_trends()
        elif platform.lower() == Platform.INSTAGRAM:
            return self._get_instagram_trends()
        # Add other platforms
        return []

    def _generate_topic_hashtags(self, topic: str) -> List[str]:
        """Generate topic-specific hashtags"""
        # Use NLP to extract key terms
        doc = nlp(topic)
        terms = [token.text.lower() for token in doc if token.pos_ in ["NOUN", "PROPN"]]
        
        # Generate hashtag variations
        hashtags = []
        for term in terms:
            hashtags.append(f"#{term}")
            hashtags.append(f"#{term}tips")
            hashtags.append(f"#{term}advice")
            
        return hashtags

    def _append_hashtags(
        self,
        content: str,
        hashtags: List[str],
        platform: str
    ) -> str:
        """Append hashtags to content based on platform"""
        if platform.lower() == Platform.INSTAGRAM:
            return f"{content}\n\n{''.join(hashtags)}"
        elif platform.lower() == Platform.TWITTER:
            return f"{content} {' '.join(hashtags[:3])}"
        return content

def main():
    """Main function to run the social reporter from command line"""
    parser = argparse.ArgumentParser(description="Hexa Assistant Social Media Reporter")
    parser.add_argument("--platform", required=True, help="Social media platform")
    parser.add_argument("--topic", required=True, help="Content topic")
    parser.add_argument("--tone", default="casual", help="Content tone")
    parser.add_argument("--length", type=int, help="Content length")
    parser.add_argument("--schedule", help="Schedule time (ISO format)")
    parser.add_argument("--timezone", default="UTC", help="Time zone")
    
    args = parser.parse_args()
    
    reporter = SocialReporter()
    
    content = reporter.generate_social_media_content(
        args.platform,
        args.topic,
        args.tone,
        args.length or reporter._get_optimal_length(args.platform)
    )
    
    if args.schedule:
        reporter.schedule_post(
            content,
            args.platform,
            args.schedule,
            args.timezone
        )
    else:
        print(content)

if __name__ == "__main__":
    main()
