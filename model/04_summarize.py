import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from sentence_transformers import SentenceTransformer
import numpy as np
import os

class FactSummarizer:
    def __init__(self, model_name="t5-base", device=None):
        """
        Initialize the fact summarizer
        
        Args:
            model_name (str): Name of the T5 model to use
            device (str): Device to run the model on (cpu, cuda)
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"Loading T5 model on {self.device}...")
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(self.device)
        
        print("Loading sentence transformer for semantic analysis...")
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def generate_claim_summary(self, claim_result, max_length=150):
        """
        Generate a summary for a single fact-checked claim
        
        Args:
            claim_result (dict): Result of fact checking a claim
            max_length (int): Maximum length of the summary
            
        Returns:
            str: Summary of the fact check result
        """
        # Extract information
        claim_text = claim_result['claim']['text']
        verdict = claim_result['verdict']['rating']
        confidence = claim_result['verdict']['confidence']
        explanation = claim_result['verdict']['explanation']
        
        # Prepare input text
        input_text = f"summarize: claim: {claim_text} verdict: {verdict} confidence: {confidence} explanation: {explanation}"
        
        # Tokenize input
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)
        
        # Generate summary
        output_ids = self.model.generate(
            input_ids,
            max_length=max_length,
            num_beams=4,
            no_repeat_ngram_size=2,
            early_stopping=True
        )
        
        # Decode summary
        summary = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        return summary
    
    def find_most_important_claims(self, fact_check_results, top_n=3):
        """
        Find the most important claims based on misinformation score and similarity
        
        Args:
            fact_check_results (list): List of fact check results
            top_n (int): Number of top claims to return
            
        Returns:
            list: Top important claims
        """
        # If there are fewer claims than top_n, return all of them
        if len(fact_check_results) <= top_n:
            return fact_check_results
        
        # Extract claim texts and misinformation scores
        claims = [result['claim']['text'] for result in fact_check_results]
        scores = [result['claim'].get('misinformation_score', 0.5) for result in fact_check_results]
        
        # Get sentence embeddings
        embeddings = self.sentence_model.encode(claims)
        
        # Calculate similarity matrix
        similarity_matrix = np.dot(embeddings, embeddings.T)
        
        # Initialize list of selected claims
        selected_indices = []
        
        # Start with the highest misinformation score
        remaining_indices = list(range(len(claims)))
        
        while len(selected_indices) < top_n and remaining_indices:
            # Get claim with highest score from remaining claims
            best_idx = max(remaining_indices, key=lambda i: scores[i])
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)
            
            # Remove similar claims (cosine similarity > 0.8)
            remaining_indices = [
                i for i in remaining_indices 
                if similarity_matrix[best_idx, i] < 0.8
            ]
        
        # Return selected claims
        return [fact_check_results[i] for i in selected_indices]
    
    def generate_overall_summary(self, fact_check_results, full_text=None, max_length=500):
        """
        Generate an overall summary of fact checking results
        
        Args:
            fact_check_results (list): List of fact check results
            full_text (str): Original full text (optional)
            max_length (int): Maximum length of the summary
            
        Returns:
            dict: Summary information
        """
        # Count verdicts
        verdicts = {
            'True': 0,
            'Likely True': 0,
            'Mixed': 0,
            'Likely False': 0,
            'False': 0,
            'Unverified': 0
        }
        
        for result in fact_check_results:
            verdict = result['verdict']['rating']
            verdicts[verdict] = verdicts.get(verdict, 0) + 1
        
        # Find most important claims
        important_claims = self.find_most_important_claims(fact_check_results)
        
        # Generate summaries for important claims
        claim_summaries = [
            {
                'claim': result['claim']['text'],
                'verdict': result['verdict']['rating'],
                'summary': self.generate_claim_summary(result)
            }
            for result in important_claims
        ]
        
        # Calculate overall misinformation score
        false_claims = verdicts['False'] + verdicts['Likely False']
        true_claims = verdicts['True'] + verdicts['Likely True']
        total_claims = sum(verdicts.values())
        
        if total_claims > 0:
            misinformation_score = false_claims / total_claims
        else:
            misinformation_score = 0.0
        
        # Prepare content for T5 summarization
        if full_text and len(full_text) > 100:
            # Generate overall context summary
            input_text = f"summarize: {full_text[:4000]}"  # Truncate to avoid exceeding model's limits
            
            input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)
            
            output_ids = self.model.generate(
                input_ids,
                max_length=max_length // 2,
                num_beams=4,
                no_repeat_ngram_size=2,
                early_stopping=True
            )
            
            content_summary = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        else:
            content_summary = "No content provided for summarization."
        
        # Generate misinformation analysis
        misinformation_analysis = self.generate_misinformation_analysis(verdicts, misinformation_score)
        
        return {
            'verdicts': verdicts,
            'misinformation_score': round(misinformation_score, 2),
            'misinformation_analysis': misinformation_analysis,
            'content_summary': content_summary,
            'important_claims': claim_summaries
        }
    
    def generate_misinformation_analysis(self, verdicts, misinformation_score):
        """
        Generate analysis of misinformation based on verdicts
        
        Args:
            verdicts (dict): Count of each verdict type
            misinformation_score (float): Overall misinformation score
            
        Returns:
            str: Analysis of misinformation
        """
        total_claims = sum(verdicts.values())
        
        if total_claims == 0:
            return "No claims were analyzed."
        
        false_claims = verdicts['False'] + verdicts['Likely False']
        true_claims = verdicts['True'] + verdicts['Likely True']
        mixed_claims = verdicts['Mixed']
        unverified_claims = verdicts['Unverified']
        
        if misinformation_score >= 0.7:
            severity = "high degree"
        elif misinformation_score >= 0.3:
            severity = "moderate degree"
        else:
            severity = "low degree"
        
        analysis = f"Analysis indicates a {severity} of potential misinformation. "
        
        if false_claims > 0:
            analysis += f"Found {false_claims} false or likely false claim"
            analysis += "s" if false_claims > 1 else ""
            analysis += " out of {total_claims} total claims. "
        
        if true_claims > 0:
            analysis += f"{true_claims} claim"
            analysis += "s" if true_claims > 1 else ""
            analysis += " appear to be factually accurate. "
        
        if mixed_claims > 0:
            analysis += f"{mixed_claims} claim"
            analysis += "s" if mixed_claims > 1 else ""
            analysis += " show mixed factual content. "
        
        if unverified_claims > 0:
            analysis += f"{unverified_claims} claim"
            analysis += "s" if unverified_claims > 1 else ""
            analysis += " could not be verified by available sources."
        
        return analysis

if __name__ == "__main__":
    # Example usage
    summarizer = FactSummarizer()
    
    # Sample fact check results
    sample_results = [
        {
            'claim': {'text': 'Drinking bleach cures COVID-19'},
            'verdict': {
                'rating': 'False',
                'confidence': 0.95,
                'explanation': 'Multiple fact checks confirm this claim is dangerous and false'
            }
        },
        {
            'claim': {'text': 'Exercise improves mental health'},
            'verdict': {
                'rating': 'True',
                'confidence': 0.92,
                'explanation': 'Scientific studies confirm regular exercise benefits mental health'
            }
        }
    ]
    
    # Generate summaries
    for result in sample_results:
        summary = summarizer.generate_claim_summary(result)
        print(f"Claim: {result['claim']['text']}")
        print(f"Summary: {summary}\n")
    
    # Generate overall summary
    overall_summary = summarizer.generate_overall_summary(sample_results)
    print("Overall Summary:")
    print(f"Misinformation Score: {overall_summary['misinformation_score']}")
    print(f"Analysis: {overall_summary['misinformation_analysis']}")
    print("Important Claims:")
    for claim in overall_summary['important_claims']:
        print(f"- {claim['claim']} ({claim['verdict']}): {claim['summary']}")