#!/usr/bin/env python3

import os
import json
import logging
import argparse
from typing import List, Dict, Optional, Union, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import pytz
import requests
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
import tweepy
from facebook import GraphAPI
from instabot import Bot
from linkedin import linkedin
import tiktok
import praw
import medium
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
class PublishConfig:
    """Configuration class for content publishing"""
    content_type: str
    content: str
    platforms: List[str]
    schedule_time: Optional[str] = None
    timezone: str = "UTC"
    repeat: bool = False
    repeat_interval: Optional[str] = None
    language: str = "en"
    template: Optional[str] = None
    media_files: Optional[List[str]] = None
    campaign: Optional[str] = None

class ContentType:
    """Supported content types"""
    TEXT = "text"
    VIDEO = "video"
    IMAGE = "image"
    ARTICLE = "article"
    PROMOTION = "promotion"

class Platform:
    """Supported platforms"""
    YOUTUBE = "youtube"
    INSTAGRAM = "instagram"
    TWITTER = "twitter"
    LINKEDIN = "linkedin"
    FACEBOOK = "facebook"
    TIKTOK = "tiktok"
    REDDIT = "reddit"
    MEDIUM = "medium"

class Publisher:
    def __init__(self, api_keys: Dict[str, str] = None):
        """Initialize publisher with API keys"""
        self.api_keys = api_keys or self._load_api_keys()
        self._init_apis()
        
        # Initialize translation service
        self.translator = Translator()
        
        # Load templates and configurations
        self.templates = self._load_templates()
        self.platform_configs = self._load_platform_configs()
        
    def _load_api_keys(self) -> Dict[str, str]:
        """Load API keys from configuration file"""
        try:
            with open("config/publisher_keys.json", "r") as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning("API keys file not found. Some features may be limited.")
            return {}

    def _init_apis(self) -> None:
        """Initialize API clients for each platform"""
        # YouTube API
        if self.api_keys.get("youtube"):
            self.youtube = self._init_youtube_api()
            
        # Twitter API
        if self.api_keys.get("twitter"):
            auth = tweepy.OAuthHandler(
                self.api_keys["twitter"]["consumer_key"],
                self.api_keys["twitter"]["consumer_secret"]
            )
            auth.set_access_token(
                self.api_keys["twitter"]["access_token"],
                self.api_keys["twitter"]["access_token_secret"]
            )
            self.twitter = tweepy.API(auth)
            
        # Facebook API
        if self.api_keys.get("facebook"):
            self.facebook = GraphAPI(self.api_keys["facebook"])
            
        # Instagram API
        if self.api_keys.get("instagram"):
            self.instagram = Bot()
            self.instagram.login(
                username=self.api_keys["instagram"]["username"],
                password=self.api_keys["instagram"]["password"]
            )
            
        # LinkedIn API
        if self.api_keys.get("linkedin"):
            self.linkedin = linkedin.LinkedInApplication(
                token=self.api_keys["linkedin"]
            )
            
        # TikTok API
        if self.api_keys.get("tiktok"):
            self.tiktok = tiktok.TikTokApi(self.api_keys["tiktok"])
            
        # Reddit API
        if self.api_keys.get("reddit"):
            self.reddit = praw.Reddit(
                client_id=self.api_keys["reddit"]["client_id"],
                client_secret=self.api_keys["reddit"]["client_secret"],
                user_agent=self.api_keys["reddit"]["user_agent"]
            )
            
        # Medium API
        if self.api_keys.get("medium"):
            self.medium = medium.MediumClient(self.api_keys["medium"])

    def publish_content(
        self,
        content_type: str,
        content: str,
        platforms: List[str]
    ) -> Dict[str, str]:
        """Publish content to specified platforms"""
        logger.info(f"Publishing {content_type} content to {platforms}")
        
        results = {}
        for platform in platforms:
            try:
                # Format content for platform
                formatted_content = self.format_for_platform(content, platform)
                
                # Generate captions and hashtags
                caption = self.generate_captions_and_hashtags(content, platform)
                
                # Publish based on content type
                if content_type == ContentType.TEXT:
                    post_id = self._publish_text(formatted_content, caption, platform)
                elif content_type == ContentType.VIDEO:
                    post_id = self._publish_video(content, caption, platform)
                elif content_type == ContentType.IMAGE:
                    post_id = self._publish_image(content, caption, platform)
                elif content_type == ContentType.ARTICLE:
                    post_id = self._publish_article(content, platform)
                elif content_type == ContentType.PROMOTION:
                    post_id = self._publish_promotion(content, platform)
                else:
                    raise ValueError(f"Unsupported content type: {content_type}")
                    
                results[platform] = post_id
                
            except Exception as e:
                logger.error(f"Error publishing to {platform}: {str(e)}")
                results[platform] = f"Error: {str(e)}"
                
        return results

    def schedule_content(
        self,
        content: str,
        platforms: List[str],
        post_time: str,
        repeat: bool = False
    ) -> Dict[str, str]:
        """Schedule content for publishing"""
        try:
            # Parse post time
            post_datetime = datetime.strptime(post_time, "%Y-%m-%d %H:%M:%S")
            timezone = pytz.timezone("UTC")
            post_datetime = timezone.localize(post_datetime)
            
            # Create schedule entry
            schedule = {
                "content": content,
                "platforms": platforms,
                "post_time": post_datetime.isoformat(),
                "repeat": repeat,
                "status": "scheduled"
            }
            
            # Save schedule
            self._save_schedule(schedule)
            
            return {"status": "scheduled", "post_time": post_time}
            
        except Exception as e:
            logger.error(f"Error scheduling content: {str(e)}")
            raise

    def track_performance(self, platform: str, post_id: str) -> Dict:
        """Track content performance"""
        try:
            if platform == Platform.YOUTUBE:
                return self._track_youtube_performance(post_id)
            elif platform == Platform.TWITTER:
                return self._track_twitter_performance(post_id)
            elif platform == Platform.FACEBOOK:
                return self._track_facebook_performance(post_id)
            elif platform == Platform.INSTAGRAM:
                return self._track_instagram_performance(post_id)
            elif platform == Platform.LINKEDIN:
                return self._track_linkedin_performance(post_id)
            elif platform == Platform.TIKTOK:
                return self._track_tiktok_performance(post_id)
            elif platform == Platform.REDDIT:
                return self._track_reddit_performance(post_id)
            elif platform == Platform.MEDIUM:
                return self._track_medium_performance(post_id)
            else:
                raise ValueError(f"Unsupported platform: {platform}")
                
        except Exception as e:
            logger.error(f"Error tracking performance: {str(e)}")
            raise

    def generate_shareable_links(
        self,
        content_id: str,
        platforms: List[str]
    ) -> Dict[str, str]:
        """Generate shareable links for content"""
        try:
            links = {}
            for platform in platforms:
                if platform == Platform.YOUTUBE:
                    links[platform] = f"https://youtube.com/watch?v={content_id}"
                elif platform == Platform.TWITTER:
                    links[platform] = f"https://twitter.com/i/web/status/{content_id}"
                elif platform == Platform.FACEBOOK:
                    links[platform] = f"https://facebook.com/{content_id}"
                elif platform == Platform.INSTAGRAM:
                    links[platform] = f"https://instagram.com/p/{content_id}"
                elif platform == Platform.LINKEDIN:
                    links[platform] = f"https://linkedin.com/feed/update/{content_id}"
                elif platform == Platform.TIKTOK:
                    links[platform] = f"https://tiktok.com/@user/video/{content_id}"
                elif platform == Platform.REDDIT:
                    links[platform] = f"https://reddit.com/comments/{content_id}"
                elif platform == Platform.MEDIUM:
                    links[platform] = f"https://medium.com/p/{content_id}"
                    
            return links
            
        except Exception as e:
            logger.error(f"Error generating links: {str(e)}")
            raise

    def generate_captions_and_hashtags(
        self,
        content: str,
        platform: str
    ) -> str:
        """Generate captions and hashtags for content"""
        try:
            # Get platform-specific hashtag rules
            rules = self.platform_configs[platform]["hashtag_rules"]
            
            # Extract keywords from content
            keywords = self._extract_keywords(content)
            
            # Generate hashtags
            hashtags = []
            for keyword in keywords:
                if len(hashtags) < rules["max_hashtags"]:
                    hashtags.append(f"#{keyword.replace(' ', '')}")
                    
            # Format caption
            caption = f"{content}\n\n{' '.join(hashtags)}"
            
            return caption
            
        except Exception as e:
            logger.error(f"Error generating captions: {str(e)}")
            raise

    def integrate_with_tools(self, content: str, tool: str) -> str:
        """Integrate with other Hexa tools"""
        try:
            if tool == "video_creator":
                from video_creator import VideoCreator
                creator = VideoCreator()
                return creator.generate_video_content(content, "explainer", "youtube")
                
            elif tool == "blog_generator":
                from blog_generator import BlogGenerator
                generator = BlogGenerator()
                return generator.generate_blog_content(content, ["blog"], 500, "formal")
                
            elif tool == "social_reporter":
                from social_reporter import SocialReporter
                reporter = SocialReporter()
                return reporter.generate_social_content(content, "twitter")
                
            else:
                raise ValueError(f"Unknown tool: {tool}")
                
        except Exception as e:
            logger.error(f"Error integrating with tools: {str(e)}")
            raise

    def apply_post_template(self, template_name: str, content: str) -> str:
        """Apply post template to content"""
        try:
            # Load template
            template = self.templates.get(template_name)
            if not template:
                raise ValueError(f"Template not found: {template_name}")
                
            # Apply template
            formatted_content = template["format"].format(content=content)
            
            # Add template-specific elements
            if "hashtags" in template:
                formatted_content += f"\n\n{' '.join(template['hashtags'])}"
                
            if "call_to_action" in template:
                formatted_content += f"\n\n{template['call_to_action']}"
                
            return formatted_content
            
        except Exception as e:
            logger.error(f"Error applying template: {str(e)}")
            raise

    def upload_media(self, platform: str, media_files: List[str]) -> List[str]:
        """Upload media files to platform"""
        try:
            uploaded_ids = []
            
            for media_file in media_files:
                if platform == Platform.YOUTUBE:
                    video_id = self._upload_to_youtube(media_file)
                    uploaded_ids.append(video_id)
                elif platform == Platform.INSTAGRAM:
                    media_id = self._upload_to_instagram(media_file)
                    uploaded_ids.append(media_id)
                elif platform == Platform.FACEBOOK:
                    media_id = self._upload_to_facebook(media_file)
                    uploaded_ids.append(media_id)
                elif platform == Platform.TWITTER:
                    media_id = self._upload_to_twitter(media_file)
                    uploaded_ids.append(media_id)
                elif platform == Platform.LINKEDIN:
                    media_id = self._upload_to_linkedin(media_file)
                    uploaded_ids.append(media_id)
                elif platform == Platform.TIKTOK:
                    media_id = self._upload_to_tiktok(media_file)
                    uploaded_ids.append(media_id)
                    
            return uploaded_ids
            
        except Exception as e:
            logger.error(f"Error uploading media: {str(e)}")
            raise

    def translate_post_content(
        self,
        content: str,
        target_language: str
    ) -> str:
        """Translate post content to target language"""
        try:
            # Detect source language
            source_lang = detect(content)
            
            # Translate content
            translated = self.translator.translate(
                content,
                src=source_lang,
                dest=target_language
            ).text
            
            return translated
            
        except Exception as e:
            logger.error(f"Error translating content: {str(e)}")
            raise

    def authenticate_with_api(self, platform: str) -> str:
        """Authenticate with platform API"""
        try:
            if platform == Platform.YOUTUBE:
                return self._authenticate_youtube()
            elif platform == Platform.TWITTER:
                return self._authenticate_twitter()
            elif platform == Platform.FACEBOOK:
                return self._authenticate_facebook()
            elif platform == Platform.INSTAGRAM:
                return self._authenticate_instagram()
            elif platform == Platform.LINKEDIN:
                return self._authenticate_linkedin()
            elif platform == Platform.TIKTOK:
                return self._authenticate_tiktok()
            elif platform == Platform.REDDIT:
                return self._authenticate_reddit()
            elif platform == Platform.MEDIUM:
                return self._authenticate_medium()
            else:
                raise ValueError(f"Unsupported platform: {platform}")
                
        except Exception as e:
            logger.error(f"Error authenticating with {platform}: {str(e)}")
            raise

    def create_dashboard_view(self, user_id: str) -> Dict:
        """Create dashboard view for user"""
        try:
            # Get user's content
            content = self._get_user_content(user_id)
            
            # Get scheduled posts
            scheduled = self._get_scheduled_posts(user_id)
            
            # Get performance metrics
            metrics = self._get_performance_metrics(user_id)
            
            return {
                "content": content,
                "scheduled": scheduled,
                "metrics": metrics
            }
            
        except Exception as e:
            logger.error(f"Error creating dashboard: {str(e)}")
            raise

    def repurpose_content(
        self,
        content: str,
        platforms: List[str]
    ) -> Dict[str, str]:
        """Repurpose content for different platforms"""
        try:
            results = {}
            
            for platform in platforms:
                # Get platform-specific format
                format_type = self.detect_platform_format(platform)
                
                # Repurpose content
                if format_type == "short_text":
                    repurposed = self._create_short_text(content)
                elif format_type == "long_text":
                    repurposed = self._create_long_text(content)
                elif format_type == "video":
                    repurposed = self._create_video_content(content)
                elif format_type == "image":
                    repurposed = self._create_image_content(content)
                else:
                    raise ValueError(f"Unsupported format: {format_type}")
                    
                results[platform] = repurposed
                
            return results
            
        except Exception as e:
            logger.error(f"Error repurposing content: {str(e)}")
            raise

    def auto_respond_to_comments(
        self,
        post_id: str,
        platform: str,
        response: str
    ) -> None:
        """Automatically respond to comments"""
        try:
            if platform == Platform.YOUTUBE:
                self._respond_to_youtube_comments(post_id, response)
            elif platform == Platform.TWITTER:
                self._respond_to_twitter_comments(post_id, response)
            elif platform == Platform.FACEBOOK:
                self._respond_to_facebook_comments(post_id, response)
            elif platform == Platform.INSTAGRAM:
                self._respond_to_instagram_comments(post_id, response)
            elif platform == Platform.LINKEDIN:
                self._respond_to_linkedin_comments(post_id, response)
            elif platform == Platform.TIKTOK:
                self._respond_to_tiktok_comments(post_id, response)
            elif platform == Platform.REDDIT:
                self._respond_to_reddit_comments(post_id, response)
            elif platform == Platform.MEDIUM:
                self._respond_to_medium_comments(post_id, response)
                
        except Exception as e:
            logger.error(f"Error responding to comments: {str(e)}")
            raise

    def manage_campaign(
        self,
        campaign_name: str,
        campaign_schedule: Dict
    ) -> Dict:
        """Manage content campaign"""
        try:
            # Create campaign
            campaign = {
                "name": campaign_name,
                "schedule": campaign_schedule,
                "status": "active",
                "created_at": datetime.now().isoformat()
            }
            
            # Save campaign
            self._save_campaign(campaign)
            
            # Schedule campaign posts
            for post in campaign_schedule["posts"]:
                self.schedule_content(
                    post["content"],
                    post["platforms"],
                    post["time"],
                    post.get("repeat", False)
                )
                
            return {"status": "created", "campaign_name": campaign_name}
            
        except Exception as e:
            logger.error(f"Error managing campaign: {str(e)}")
            raise

    def handle_posting_error(self, post_id: str, platform: str) -> None:
        """Handle posting errors"""
        try:
            # Log error
            logger.error(f"Error posting to {platform}: {post_id}")
            
            # Get error details
            error = self._get_posting_error(post_id, platform)
            
            # Retry if appropriate
            if error["retryable"]:
                self._retry_posting(post_id, platform)
                
            # Notify user
            self._notify_posting_error(post_id, platform, error)
            
        except Exception as e:
            logger.error(f"Error handling posting error: {str(e)}")
            raise

    def validate_and_preview_links(self, content: str) -> Dict:
        """Validate and preview links in content"""
        try:
            # Extract links from content
            links = self._extract_links(content)
            
            results = {}
            for link in links:
                # Validate link
                is_valid = self._validate_link(link)
                
                # Get preview
                preview = self._get_link_preview(link)
                
                results[link] = {
                    "valid": is_valid,
                    "preview": preview
                }
                
            return results
            
        except Exception as e:
            logger.error(f"Error validating links: {str(e)}")
            raise

def main():
    """Main function to run the publisher from command line"""
    parser = argparse.ArgumentParser(description="Hexa Assistant Content Publisher")
    parser.add_argument("--content-type", required=True, help="Type of content")
    parser.add_argument("--content", required=True, help="Content to publish")
    parser.add_argument("--platforms", required=True, nargs="+", help="Target platforms")
    parser.add_argument("--schedule", help="Schedule time (YYYY-MM-DD HH:MM:SS)")
    parser.add_argument("--timezone", default="UTC", help="Time zone")
    parser.add_argument("--repeat", action="store_true", help="Repeat posting")
    parser.add_argument("--template", help="Template to use")
    parser.add_argument("--media", nargs="+", help="Media files to upload")
    parser.add_argument("--campaign", help="Campaign name")
    
    args = parser.parse_args()
    
    publisher = Publisher()
    
    # Create publish config
    config = PublishConfig(
        content_type=args.content_type,
        content=args.content,
        platforms=args.platforms,
        schedule_time=args.schedule,
        timezone=args.timezone,
        repeat=args.repeat,
        template=args.template,
        media_files=args.media,
        campaign=args.campaign
    )
    
    if args.schedule:
        # Schedule content
        result = publisher.schedule_content(
            args.content,
            args.platforms,
            args.schedule,
            args.repeat
        )
    else:
        # Publish immediately
        result = publisher.publish_content(
            args.content_type,
            args.content,
            args.platforms
        )
        
    print(f"Publishing result: {result}")

if __name__ == "__main__":
    main()
