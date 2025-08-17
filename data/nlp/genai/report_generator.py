import openai
import pandas as pd
from config import OPENAI_API_KEY, GENAI_MODEL

openai.api_key = OPENAI_API_KEY

def generate_insight_report(df, analysis_results):
    """Generate natural language insights using GPT-4"""
    # Prepare data summary
    organic_perc = df['is_organic'].mean() * 100
    price_diff = df[df['is_organic'] == 1]['price'].mean() - df[df['is_organic'] == 0]['price'].mean()
    
    prompt = f"""
    You are a food market analyst specializing in organic products. 
    Generate a comprehensive report based on the following dataset analysis:
    
    Dataset Overview:
    - Total products: {len(df)}
    - Organic products: {organic_perc:.1f}%
    - Average price premium for organic: ${price_diff:.2f}
    
    Key Analysis Findings:
    {analysis_results}
    
    Create a professional report with these sections:
    1. Executive Summary
    2. Market Trends (organic vs conventional)
    3. Pricing Analysis
    4. Nutritional Comparison
    5. Consumer Sentiment Insights
    6. Strategic Recommendations
    
    Use markdown formatting with appropriate headers. Include 3-5 key takeaways at the end.
    """
    
    response = openai.ChatCompletion.create(
        model=GENAI_MODEL,
        messages=[
            {"role": "system", "content": "You are an expert data analyst in the food industry."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=1500
    )
    
    return response.choices[0].message['content']

def generate_product_description(features):
    """Generate product descriptions using GenAI"""
    prompt = f"""
    Create an appealing product description for a food item with these characteristics:
    - Category: {features['category']}
    - Organic: {'Yes' if features['is_organic'] else 'No'}
    - Key Features: {features.get('key_features', '')}
    - Certifications: {features.get('certification', 'None')}
    
    Description should be 2-3 sentences, engaging, and highlight unique selling points.
    Target audience: health-conscious consumers.
    """
    
    response = openai.ChatCompletion.create(
        model=GENAI_MODEL,
        messages=[
            {"role": "system", "content": "You are a marketing copywriter for a food company."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=200
    )
    
    return response.choices[0].message['content']
