"""
01_conversion.py - YouTube Link to Text Conversion Module

This module downloads YouTube videos, extracts audio, and converts speech to text
with punctuation restoration.
"""

import os
import tempfile
import yt_dlp
import ffmpeg
import whisper
import torch
from transformers import pipeline

class YouTubeConverter:
    def __init__(self, model_size="base", device=None):
        """
        Initialize the YouTube to text converter
        
        Args:
            model_size (str): Size of the Whisper model to use (tiny, base, small, medium, large)
            device (str): Device to run the model on (cpu, cuda)
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Loading Whisper model ({model_size}) on {self.device}...")
        self.speech_model = whisper.load_model(model_size).to(self.device)
        
        print("Loading punctuation restoration model...")
        self.punctuation_model = pipeline(
            "text2text-generation",
            model="oliverguhr/fullstop-punctuation-multilang-large", 
            device=0 if self.device == "cuda" else -1
        )
    
    def download_youtube_audio(self, youtube_url, output_path=None):
        """
        Download audio from a YouTube video
        
        Args:
            youtube_url (str): URL of the YouTube video
            output_path (str): Path to save the audio file
            
        Returns:
            str: Path to the saved audio file
        """
        if output_path is None:
            # Create a temporary file
            temp_dir = tempfile.gettempdir()
            output_path = os.path.join(temp_dir, "youtube_audio.mp3")
        
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'outtmpl': output_path.replace('.mp3', ''),
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            print(f"Downloading audio from {youtube_url}...")
            ydl.download([youtube_url])
        
        return output_path
    
    def transcribe_audio(self, audio_path):
        """
        Transcribe audio file to text
        
        Args:
            audio_path (str): Path to the audio file
            
        Returns:
            dict: Transcription result with segments and text
        """
        print(f"Transcribing audio file: {audio_path}")
        result = self.speech_model.transcribe(audio_path)
        return result
    
    def restore_punctuation(self, text):
        """
        Restore punctuation in the transcribed text
        
        Args:
            text (str): Text without proper punctuation
            
        Returns:
            str: Text with restored punctuation
        """
        # Process text in chunks to avoid exceeding model's max length
        max_chunk_size = 500
        chunks = [text[i:i+max_chunk_size] for i in range(0, len(text), max_chunk_size)]
        
        punctuated_chunks = []
        for chunk in chunks:
            result = self.punctuation_model(chunk)
            punctuated_chunks.append(result[0]['generated_text'])
        
        return ' '.join(punctuated_chunks)
    
    def process_youtube_url(self, youtube_url):
        """
        Process a YouTube URL: download audio, transcribe, and restore punctuation
        
        Args:
            youtube_url (str): URL of the YouTube video
            
        Returns:
            dict: Dictionary with transcription segments, full text, and punctuated text
        """
        # Download audio
        audio_path = self.download_youtube_audio(youtube_url)
        
        # Transcribe audio
        transcription = self.transcribe_audio(audio_path)
        
        # Restore punctuation
        punctuated_text = self.restore_punctuation(transcription['text'])
        
        # Cleanup temporary file
        if os.path.exists(audio_path):
            os.remove(audio_path)
        
        return {
            'segments': transcription['segments'],
            'text': transcription['text'],
            'punctuated_text': punctuated_text
        }

def preprocess_text(text):
    """
    Basic text preprocessing
    
    Args:
        text (str): Input text
        
    Returns:
        str: Preprocessed text
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

if __name__ == "__main__":
    # Example usage
    youtube_url = "https://youtu.be/UrUjvgnJSyA?si=Swvbe26zG1_qV_kk"  # Replace with actual URL
    
    converter = YouTubeConverter(model_size="base")
    result = converter.process_youtube_url(youtube_url)
    
    # Print the first 500 characters of the result
    print(result['punctuated_text'][:500])
    
    # Basic preprocessing
    preprocessed_text = preprocess_text(result['punctuated_text'])
    print("\nPreprocessed text (first 500 chars):")
    print(preprocessed_text[:500])