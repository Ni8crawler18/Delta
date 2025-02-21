import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import re
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import os

# Download NLTK data (if not already downloaded)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class ClaimsDetector:
    def __init__(self, model_path=None, device=None):
        """
        Initialize the claims detection model
        
        Args:
            model_path (str): Path to the pretrained model
            device (str): Device to run the model on (cpu, cuda)
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        # Load pretrained tokenizer and model
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        
        if model_path and os.path.exists(model_path):
            print(f"Loading fine-tuned model from {model_path}")
            self.model = DistilBertForSequenceClassification.from_pretrained(model_path).to(self.device)
        else:
            print("Loading pretrained DistilBERT model")
            self.model = DistilBertForSequenceClassification.from_pretrained(
                'distilbert-base-uncased',
                num_labels=2  # binary classification: factual vs. misinformation
            ).to(self.device)
        
        # TF-IDF vectorizer for claim identification
        self.vectorizer = TfidfVectorizer(max_features=5000)
        
        # Claim identification patterns
        self.claim_patterns = [
            r'(according to|reports indicate|sources say|experts claim)',
            r'(statistics show|research suggests|studies indicate)',
            r'(it is (reported|said|claimed|alleged) that)',
            r'(\d+% of|millions of|billions of)',
            r'(all|none|never|always|everyone|nobody)',
            r'(causes|leads to|results in|linked to)',
            r'(scientists discovered|researchers found)',
        ]
        self.claim_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.claim_patterns]
    
    def extract_sentences(self, text):
        """
        Extract sentences from text
        
        Args:
            text (str): Input text
            
        Returns:
            list: List of sentences
        """
        return sent_tokenize(text)
    
    def identify_claims(self, sentences):
        """
        Identify potential claims from sentences
        
        Args:
            sentences (list): List of sentences
            
        Returns:
            list: List of sentences identified as potential claims
        """
        # Method 1: Pattern matching
        pattern_claims = []
        for sentence in sentences:
            for pattern in self.claim_patterns:
                if pattern.search(sentence):
                    pattern_claims.append(sentence)
                    break
        
        # Method 2: TF-IDF based importance
        if len(sentences) > 10:  # Only apply for longer texts
            try:
                tfidf_matrix = self.vectorizer.fit_transform(sentences)
                sentence_importance = np.array(tfidf_matrix.sum(axis=1)).flatten()
                
                # Get top 30% important sentences
                threshold = np.percentile(sentence_importance, 70)
                important_indices = np.where(sentence_importance >= threshold)[0]
                tfidf_claims = [sentences[i] for i in important_indices]
            except:
                tfidf_claims = []
        else:
            tfidf_claims = sentences
        
        # Combine both methods
        claims = list(set(pattern_claims + tfidf_claims))
        
        return claims
    
    def classify_claim(self, claim):
        """
        Classify a claim as potentially containing misinformation or not
        
        Args:
            claim (str): The claim text
            
        Returns:
            dict: Classification result with scores
        """
        # Tokenize
        inputs = self.tokenizer(
            claim,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        ).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
        
        # Convert to numpy for easier handling
        probs = probabilities.cpu().numpy()[0]
        
        return {
            'text': claim,
            'factual_score': float(probs[0]),
            'misinformation_score': float(probs[1]),
            'is_misinformation': bool(probs[1] > 0.5)
        }
    
    def process_text(self, text, threshold=0.7):
        """
        Process text to identify and classify claims
        
        Args:
            text (str): Input text
            threshold (float): Confidence threshold for misinformation classification
            
        Returns:
            dict: Dictionary with identified claims and their classifications
        """
        # Extract sentences
        sentences = self.extract_sentences(text)
        
        # Identify potential claims
        potential_claims = self.identify_claims(sentences)
        
        # Classify each potential claim
        classified_claims = []
        for claim in potential_claims:
            result = self.classify_claim(claim)
            if result['misinformation_score'] > threshold:
                classified_claims.append(result)
        
        return {
            'total_sentences': len(sentences),
            'potential_claims': len(potential_claims),
            'misinformation_claims': len(classified_claims),
            'claims': classified_claims
        }

# Fine-tuning function (for future use)
def fine_tune_model(training_data, output_path):
    """
    Fine-tune the model on specific misinformation datasets
    
    Args:
        training_data (str): Path to the training data
        output_path (str): Path to save the fine-tuned model
    """
    # This would be implemented based on specific dataset format
    pass

if __name__ == "__main__":
    # Example usage
    sample_text = """
    According to recent studies, 95% of people who use this product saw results in just 2 days.
    The weather today is sunny with a high of 75 degrees.
    Scientists discovered that this herb cures all types of cancer with no side effects.
    Research suggests that regular exercise can improve mental health.
    """
    
    detector = ClaimsDetector()
    results = detector.process_text(sample_text)
    
    print(f"Total sentences: {results['total_sentences']}")
    print(f"Potential claims: {results['potential_claims']}")
    print(f"Misinformation claims: {results['misinformation_claims']}\n")
    
    for claim in results['claims']:
        print(f"Claim: {claim['text']}")
        print(f"Misinformation score: {claim['misinformation_score']:.4f}")
        print(f"Factual score: {claim['factual_score']:.4f}")
        print(f"Is misinformation: {claim['is_misinformation']}")
        print()