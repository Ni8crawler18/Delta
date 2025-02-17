"""
03_factcheck.py - Fact Checking Module

This module verifies claims by using Google Fact Check API and 
other fact checking resources.
"""

import requests
import json
import os
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import time
from bs4 import BeautifulSoup
from flask import Flask, request, jsonify
from flask_cors import CORS
import threading

# Load configuration
def load_config():
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.json')
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {
            "google_fact_check_api_key": os.environ.get("GOOGLE_FACT_CHECK_API_KEY", ""),
            "server_port": 5000
        }

config = load_config()

class FactChecker:
    def __init__(self, api_key=None):
        """
        Initialize the fact checker
        
        Args:
            api_key (str): Google Fact Check API key
        """
        self.api_key = api_key or config.get("google_fact_check_api_key", "")
        if not self.api_key:
            print("Warning: No Google Fact Check API key provided. Some features will be limited.")
        
        # Initialize the fact check API client
        try:
            self.service = build("factchecktools", "v1alpha1", developerKey=self.api_key)
        except Exception as e:
            print(f"Error initializing Google Fact Check API: {e}")
            self.service = None
    
    def check_claim_google_api(self, claim_text):
        """
        Check a claim using Google's Fact Check API
        
        Args:
            claim_text (str): The claim to check
            
        Returns:
            list: List of fact check results
        """
        if not self.service:
            return []
        
        try:
            result = self.service.claims().search(query=claim_text).execute()
            return result.get("claims", [])
        except HttpError as e:
            print(f"Error querying Google Fact Check API: {e}")
            return []
    
    def check_claim_external_sources(self, claim_text):
        """
        Check claim against external fact checking websites
        
        Args:
            claim_text (str): The claim to check
            
        Returns:
            list: List of fact check results from external sources
        """
        # List of fact checking websites
        fact_check_urls = [
            f"https://www.snopes.com/search/?q={claim_text.replace(' ', '+')}",
            f"https://www.politifact.com/search/?q={claim_text.replace(' ', '+')}",
            f"https://www.factcheck.org/search/q={claim_text.replace(' ', '+')}"
        ]
        
        results = []
        
        for url in fact_check_urls:
            try:
                response = requests.get(url, headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                })
                
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # Extract result snippets - this will vary by website
                    if "snopes.com" in url:
                        articles = soup.select(".article-card")
                        for article in articles[:3]:  # Take first 3 results
                            title_el = article.select_one(".title")
                            link_el = article.select_one("a")
                            rating_el = article.select_one(".rating")
                            
                            if title_el and link_el:
                                title = title_el.text.strip()
                                link = link_el.get('href')
                                rating = rating_el.text.strip() if rating_el else "Unknown"
                                
                                results.append({
                                    'source': 'Snopes',
                                    'title': title,
                                    'link': link,
                                    'rating': rating
                                })
                    
                    # Add similar parsing for other fact checking sites
                
            except Exception as e:
                print(f"Error fetching results from {url}: {e}")
        
        return results
    
    def fact_check_claim(self, claim):
        """
        Perform comprehensive fact checking on a claim
        
        Args:
            claim (dict): Claim object with text and metadata
            
        Returns:
            dict: Fact checking results
        """
        claim_text = claim['text']
        
        # Check with Google Fact Check API
        google_results = self.check_claim_google_api(claim_text)
        
        # Check with external sources
        external_results = self.check_claim_external_sources(claim_text)
        
        # Combine and analyze results
        all_results = google_results + external_results
        
        # Determine verdict based on results
        verdict = self.determine_verdict(all_results)
        
        return {
            'claim': claim,
            'google_results': google_results,
            'external_results': external_results,
            'verdict': verdict
        }
    
    def determine_verdict(self, results):
        """
        Determine an overall verdict based on fact check results
        
        Args:
            results (list): Combined fact checking results
            
        Returns:
            dict: Verdict with rating and confidence
        """
        if not results:
            return {
                'rating': 'Unverified',
                'confidence': 0.0,
                'explanation': 'No fact check results found'
            }
        
        # Count ratings from fact checks
        ratings = {
            'true': 0,
            'mostly_true': 0,
            'mixed': 0,
            'mostly_false': 0,
            'false': 0
        }
        
        # Process Google API results
        for result in results:
            if isinstance(result, dict) and 'textualRating' in result:
                rating = result['textualRating'].lower()
                
                if 'true' in rating and 'mostly' not in rating and 'partly' not in rating:
                    ratings['true'] += 1
                elif 'mostly true' in rating or 'partly true' in rating:
                    ratings['mostly_true'] += 1
                elif 'mixed' in rating or 'partly' in rating:
                    ratings['mixed'] += 1
                elif 'mostly false' in rating:
                    ratings['mostly_false'] += 1
                elif 'false' in rating or 'pants on fire' in rating:
                    ratings['false'] += 1
        
        # Process external results (simplified for this example)
        for result in results:
            if isinstance(result, dict) and 'rating' in result:
                rating = result['rating'].lower()
                
                if 'true' in rating and 'mostly' not in rating:
                    ratings['true'] += 1
                elif 'mostly true' in rating:
                    ratings['mostly_true'] += 1
                elif 'mixed' in rating or 'partly' in rating:
                    ratings['mixed'] += 1
                elif 'mostly false' in rating:
                    ratings['mostly_false'] += 1
                elif 'false' in rating:
                    ratings['false'] += 1
        
        # Calculate total ratings
        total_ratings = sum(ratings.values())
        if total_ratings == 0:
            return {
                'rating': 'Unverified',
                'confidence': 0.0,
                'explanation': 'No conclusive ratings found in fact checks'
            }
        
        # Determine overall rating
        false_score = (ratings['false'] * 1.0 + ratings['mostly_false'] * 0.75) / total_ratings
        true_score = (ratings['true'] * 1.0 + ratings['mostly_true'] * 0.75) / total_ratings
        mixed_score = ratings['mixed'] / total_ratings if total_ratings > 0 else 0
        
        if false_score > true_score and false_score > mixed_score:
            if false_score > 0.7:
                rating = 'False'
                confidence = false_score
                explanation = 'Multiple fact checks indicate this claim is false'
            else:
                rating = 'Likely False'
                confidence = false_score
                explanation = 'Fact checks suggest this claim is probably false'
        elif true_score > false_score and true_score > mixed_score:
            if true_score > 0.7:
                rating = 'True'
                confidence = true_score
                explanation = 'Multiple fact checks confirm this claim is true'
            else:
                rating = 'Likely True'
                confidence = true_score
                explanation = 'Fact checks suggest this claim is probably true'
        else:
            rating = 'Mixed'
            confidence = max(mixed_score, 0.5)
            explanation = 'Fact checks show mixed results for this claim'
        
        return {
            'rating': rating,
            'confidence': round(confidence, 2),
            'explanation': explanation
        }

# Flask server for fact checking API
app = Flask(__name__)
CORS(app)

# Initialize fact checker
fact_checker = FactChecker()

@app.route('/api/factcheck', methods=['POST'])
def api_fact_check():
    """API endpoint for fact checking claims"""
    if not request.json or 'claims' not in request.json:
        return jsonify({'error': 'Invalid request. Expected claims array.'}), 400
    
    claims = request.json['claims']
    results = []
    
    for claim in claims:
        if not isinstance(claim, dict) or 'text' not in claim:
            continue
        
        result = fact_checker.fact_check_claim(claim)
        results.append(result)
    
    return jsonify({
        'results': results,
        'count': len(results)
    })

def start_fact_check_server():
    """Start the fact checking server in a separate thread"""
    port = config.get("server_port", 5000)
    server_thread = threading.Thread(
        target=app.run,
        kwargs={'host': '0.0.0.0', 'port': port, 'debug': False}
    )
    server_thread.daemon = True
    server_thread.start()
    print(f"Fact Check API server running on port {port}")

if __name__ == "__main__":
    # Example usage
    checker = FactChecker()
    
    sample_claim = {
        'text': 'Drinking bleach cures COVID-19',
        'misinformation_score': 0.95
    }
    
    result = checker.fact_check_claim(sample_claim)
    print(json.dumps(result, indent=2))
    
    # Start the API server
    start_fact_check_server()