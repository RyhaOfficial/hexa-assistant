#!/usr/bin/env python3

import os
import json
import logging
import argparse
from typing import List, Dict, Optional, Union
from dataclasses import dataclass
from datetime import datetime
import nltk
import spacy
import openai
import gradio as gr
from transformers import pipeline
from bs4 import BeautifulSoup
import requests
import torch
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from textblob import TextBlob

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

# Initialize spaCy
nlp = spacy.load('en_core_web_sm')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class BlogConfig:
    """Configuration class for blog generation settings"""
    topic: str
    keywords: List[str]
    length: int
    tone: str
    mode: str
    target_audience: str
    seo_target: str
    min_keyword_density: float = 0.02
    max_keyword_density: float = 0.04
    readability_target: str = "general"  # general, technical, academic
    
class ContentMode:
    """Available content generation modes"""
    PROFESSIONAL = "professional"
    CREATIVE = "creative"
    TECHNICAL = "technical"
    MARKETING = "marketing"

class ContentTone:
    """Available content tones"""
    FORMAL = "formal"
    CASUAL = "casual"
    TECHNICAL = "technical"
    PERSUASIVE = "persuasive"

class BlogGenerator:
    def __init__(self, api_key: str = None):
        """Initialize the blog generator with optional API key"""
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if self.api_key:
            openai.api_key = self.api_key
        
        self.sentiment_analyzer = pipeline("sentiment-analysis")
        self.summarizer = pipeline("summarization")
        self.user_preferences = self._load_user_preferences()
        
    def _load_user_preferences(self) -> Dict:
        """Load user preferences from config file"""
        try:
            with open("config/user_preferences.json", "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    def generate_blog_content(
        self, 
        topic: str, 
        keywords: List[str], 
        length: int, 
        tone: str
    ) -> str:
        """Generate blog content based on given parameters"""
        logger.info(f"Generating blog content for topic: {topic}")
        
        # Create prompt for GPT model
        prompt = self._create_content_prompt(topic, keywords, length, tone)
        
        try:
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
            
            # Structure and optimize the content
            content = self.structure_content(content)
            seo_metrics = self.seo_optimization(content, keywords)
            
            if seo_metrics["keyword_density"] < 0.02:
                content = self._enhance_keyword_density(content, keywords)
                
            return content
            
        except Exception as e:
            logger.error(f"Error generating content: {str(e)}")
            raise

    def seo_optimization(self, content: str, keywords: List[str]) -> Dict:
        """Optimize content for SEO and return metrics"""
        metrics = {
            "keyword_density": self._calculate_keyword_density(content, keywords),
            "readability_score": self._calculate_readability_score(content),
            "meta_description": self._generate_meta_description(content),
            "suggested_tags": self.suggest_tags(keywords),
            "heading_structure": self._analyze_heading_structure(content),
            "internal_linking_opportunities": self._find_internal_linking_opportunities(content)
        }
        
        return metrics

    def set_tone(self, content: str, tone: str) -> str:
        """Adjust the content tone based on specified preference"""
        if tone not in vars(ContentTone).values():
            raise ValueError(f"Invalid tone: {tone}")
            
        tone_adjustments = {
            ContentTone.FORMAL: self._adjust_formal_tone,
            ContentTone.CASUAL: self._adjust_casual_tone,
            ContentTone.TECHNICAL: self._adjust_technical_tone,
            ContentTone.PERSUASIVE: self._adjust_persuasive_tone
        }
        
        return tone_adjustments[tone](content)

    def save_blog_draft(self, content: str, title: str, save_path: str) -> None:
        """Save blog content as a draft"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        metadata = {
            "title": title,
            "date": datetime.now().isoformat(),
            "keywords": self.suggest_tags(content),
            "seo_metrics": self.seo_optimization(content, self.suggest_tags(content))
        }
        
        draft_content = f"""---
{json.dumps(metadata, indent=2)}
---

{content}
"""
        
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(draft_content)
            
        logger.info(f"Draft saved to {save_path}")

    def expand_content(self, content: str) -> str:
        """Expand short content into a full blog post"""
        # Extract main points
        doc = nlp(content)
        main_points = [sent.text.strip() for sent in doc.sents]
        
        expanded_content = []
        for point in main_points:
            # Generate detailed paragraph for each point
            expanded_point = self.generate_blog_content(
                point, 
                self.suggest_tags(point),
                length=250,
                tone=ContentTone.FORMAL
            )
            expanded_content.append(expanded_point)
            
        return "\n\n".join(expanded_content)

    def generate_from_voice(self, input_audio_path: str, tone: str, length: int) -> str:
        """Generate blog content from voice input"""
        try:
            # Integrate with Hexa voice assistant for transcription
            from hexavoice_assistant import transcribe_audio
            
            # Transcribe audio to text
            transcribed_text = transcribe_audio(input_audio_path)
            
            # Generate blog from transcribed content
            return self.generate_blog_content(
                transcribed_text,
                self.suggest_tags(transcribed_text),
                length,
                tone
            )
        except ImportError:
            logger.error("Hexa voice assistant module not found")
            raise

    def suggest_tags(self, keywords: List[str]) -> List[str]:
        """Suggest SEO-friendly tags based on keywords"""
        suggested_tags = set()
        
        # Process each keyword
        for keyword in keywords:
            # Get related keywords
            related = self._get_related_keywords(keyword)
            suggested_tags.update(related)
            
            # Get LSI keywords
            lsi_keywords = self._get_lsi_keywords(keyword)
            suggested_tags.update(lsi_keywords)
            
        return list(suggested_tags)

    def structure_content(self, content: str) -> str:
        """Structure content with proper headings and formatting"""
        doc = nlp(content)
        
        # Split into sections
        sections = self._split_into_sections(doc)
        
        # Format each section
        formatted_sections = []
        for i, section in enumerate(sections):
            if i == 0:
                # Introduction
                formatted_sections.append(section)
            else:
                # Add heading
                heading = self._generate_section_heading(section)
                formatted_sections.append(f"## {heading}\n\n{section}")
                
        # Add conclusion
        conclusion = self._generate_conclusion(content)
        formatted_sections.append("\n## Conclusion\n\n" + conclusion)
        
        return "\n\n".join(formatted_sections)

    def content_analysis(self, content: str) -> Dict:
        """Analyze content and provide metrics"""
        return {
            "readability": self._calculate_readability_score(content),
            "sentiment": self._analyze_sentiment(content),
            "keyword_density": self._calculate_keyword_density(content, self.suggest_tags(content)),
            "engagement_metrics": self._calculate_engagement_metrics(content),
            "improvement_suggestions": self._generate_improvement_suggestions(content)
        }

    def set_mode(self, content: str, mode: str) -> str:
        """Set the content generation mode"""
        if mode not in vars(ContentMode).values():
            raise ValueError(f"Invalid mode: {mode}")
            
        mode_adjustments = {
            ContentMode.PROFESSIONAL: self._adjust_professional_mode,
            ContentMode.CREATIVE: self._adjust_creative_mode,
            ContentMode.TECHNICAL: self._adjust_technical_mode,
            ContentMode.MARKETING: self._adjust_marketing_mode
        }
        
        return mode_adjustments[mode](content)

    def _create_content_prompt(self, topic: str, keywords: List[str], length: int, tone: str) -> str:
        """Create a prompt for the AI model"""
        return f"""Write a {length}-word blog post about {topic}.
Use these keywords naturally: {', '.join(keywords)}
Tone: {tone}

The blog post should be informative, engaging, and well-structured.
Include an introduction, main points with subheadings, and a conclusion.

Blog post:
"""

    def _calculate_readability_score(self, content: str) -> float:
        """Calculate content readability score"""
        sentences = sent_tokenize(content)
        words = word_tokenize(content)
        
        # Calculate average sentence length
        avg_sentence_length = len(words) / len(sentences)
        
        # Calculate percentage of complex words (3+ syllables)
        complex_words = len([w for w in words if self._count_syllables(w) >= 3])
        complex_word_percentage = complex_words / len(words)
        
        # Calculate Flesch-Kincaid Grade Level
        score = 0.39 * avg_sentence_length + 11.8 * complex_word_percentage - 15.59
        return round(score, 2)

    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word"""
        return len(
            [v for v in word.lower() if v in 'aeiou']
        )

    def _analyze_sentiment(self, content: str) -> Dict:
        """Analyze content sentiment"""
        blob = TextBlob(content)
        return {
            "polarity": blob.sentiment.polarity,
            "subjectivity": blob.sentiment.subjectivity
        }

    def _calculate_engagement_metrics(self, content: str) -> Dict:
        """Calculate potential engagement metrics"""
        return {
            "estimated_read_time": len(content.split()) / 200,  # Average reading speed
            "paragraph_count": len(content.split("\n\n")),
            "image_suggestions": self._suggest_image_placements(content),
            "cta_opportunities": self._identify_cta_opportunities(content)
        }

    def _suggest_image_placements(self, content: str) -> List[Dict]:
        """Suggest where to place images in the content"""
        sections = content.split("\n\n")
        suggestions = []
        
        for i, section in enumerate(sections):
            if len(section.split()) > 150:  # Suggest image after every 150 words
                suggestions.append({
                    "position": f"After paragraph {i + 1}",
                    "suggested_type": self._suggest_image_type(section)
                })
                
        return suggestions

    def _suggest_image_type(self, section: str) -> str:
        """Suggest type of image based on content"""
        keywords = self.suggest_tags([section])
        if any(kw in section.lower() for kw in ["data", "statistics", "numbers"]):
            return "infographic"
        elif any(kw in section.lower() for kw in ["process", "steps", "workflow"]):
            return "diagram"
        else:
            return "relevant photo"

    def _identify_cta_opportunities(self, content: str) -> List[Dict]:
        """Identify opportunities for calls to action"""
        sentences = sent_tokenize(content)
        opportunities = []
        
        for i, sentence in enumerate(sentences):
            if self._is_cta_opportunity(sentence):
                opportunities.append({
                    "position": f"After sentence {i + 1}",
                    "suggested_cta": self._generate_cta_suggestion(sentence)
                })
                
        return opportunities

    def _is_cta_opportunity(self, sentence: str) -> bool:
        """Check if a sentence presents a good CTA opportunity"""
        return any(trigger in sentence.lower() for trigger in [
            "learn more",
            "discover",
            "find out",
            "get started",
            "try",
            "download"
        ])

    def _generate_cta_suggestion(self, context: str) -> str:
        """Generate a relevant call to action"""
        blob = TextBlob(context)
        if blob.sentiment.polarity > 0:
            return "Sign up now to get started!"
        else:
            return "Learn more about how we can help."

    def _generate_improvement_suggestions(self, content: str) -> List[str]:
        """Generate content improvement suggestions"""
        suggestions = []
        
        # Check readability
        readability_score = self._calculate_readability_score(content)
        if readability_score > 12:
            suggestions.append("Consider simplifying language for better readability")
            
        # Check paragraph length
        paragraphs = content.split("\n\n")
        if any(len(p.split()) > 150 for p in paragraphs):
            suggestions.append("Break down long paragraphs for better readability")
            
        # Check keyword density
        keywords = self.suggest_tags(content)
        density = self._calculate_keyword_density(content, keywords)
        if density < 0.01:
            suggestions.append("Increase keyword density for better SEO")
        elif density > 0.05:
            suggestions.append("Reduce keyword density to avoid keyword stuffing")
            
        return suggestions

def main():
    """Main function to run the blog generator from command line"""
    parser = argparse.ArgumentParser(description="Hexa Assistant Blog Generator")
    parser.add_argument("--topic", required=True, help="Blog topic")
    parser.add_argument("--keywords", required=True, help="Comma-separated keywords")
    parser.add_argument("--length", type=int, default=1000, help="Content length in words")
    parser.add_argument("--tone", default="formal", help="Content tone")
    parser.add_argument("--mode", default="professional", help="Content mode")
    parser.add_argument("--save-path", help="Path to save the blog draft")
    
    args = parser.parse_args()
    
    generator = BlogGenerator()
    
    content = generator.generate_blog_content(
        args.topic,
        args.keywords.split(","),
        args.length,
        args.tone
    )
    
    if args.save_path:
        generator.save_blog_draft(
            content,
            args.topic,
            args.save_path
        )
    else:
        print(content)

if __name__ == "__main__":
    main()
