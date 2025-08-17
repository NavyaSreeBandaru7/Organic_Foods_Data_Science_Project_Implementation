import pandas as pd
from data.simulated_data import generate_food_data
from models.train import train_organic_classifier
from genai.report_generator import generate_insight_report
from agents.insight_agent import InsightAgent
import matplotlib.pyplot as plt

def main():
    print("🚀 Starting Organic Food Analysis Project")
    
    # Generate or load data
    try:
        df = pd.read_csv('data/raw/food_dataset.csv')
        print("✅ Loaded existing dataset")
    except FileNotFoundError:
        print("🔧 Generating new dataset...")
        df = generate_food_data(2000)
        df.to_csv('data/raw/food_dataset.csv', index=False)
        print(f"📊 Generated new dataset with {len(df)} records")
    
    # Train ML model
    print("🤖 Training classification model...")
    model, accuracy, report = train_organic_classifier(df)
    print(f"🎯 Model trained with accuracy: {accuracy:.2f}")
    
    # Generate insights
    print("💡 Generating market insights...")
    analysis_results = f"""
    - Model Accuracy: {accuracy:.2f}
    - Organic Price Premium: ${df[df['is_organic']==1]['price'].mean() - df[df['is_organic']==0]['price'].mean():.2f}
    - Top Organic Categories: {df[df['is_organic']==1]['category'].value_counts().nlargest(3).to_dict()}
    """
    report = generate_insight_report(df, analysis_results)
    with open('market_insights.md', 'w') as f:
        f.write(report)
    print("📝 Insight report generated: market_insights.md")
    
    # Initialize agent
    print("🧠 Starting insight agent...")
    agent = InsightAgent(df)
    
    # Example queries
    questions = [
        "What's the price difference between organic and conventional dairy products?",
        "Are there nutritional differences between organic and conventional vegetables?",
        "Which country produces the most organic meat?"
    ]
    
    print("\n💬 Sample agent responses:")
    for q in questions:
        print(f"\nQ: {q}")
        print(f"A: {agent.query(q)[:150]}...")
    
    print("\n🎉 Project execution complete!")

if __name__ == "__main__":
    main()
