import numpy as np
import pandas as pd
from faker import Faker
from datetime import date

def generate_food_data(num_samples=1000):
    fake = Faker()
    np.random.seed(42)
    
    # Generate synthetic data
    data = []
    for _ in range(num_samples):
        is_organic = np.random.choice([0, 1], p=[0.6, 0.4])
        price_multiplier = 1.5 if is_organic else 1.0
        
        item = {
            'product_id': fake.uuid4(),
            'product_name': fake.word().capitalize() + " " + fake.word(),
            'category': np.random.choice(['Fruit', 'Vegetable', 'Dairy', 'Meat', 'Grain']),
            'price': round(np.random.uniform(1.0, 15.0) * price_multiplier, 2),
            'weight': round(np.random.uniform(0.1, 5.0), 2),
            'calories': np.random.randint(20, 500),
            'protein': round(np.random.uniform(0.1, 25.0), 1),
            'carbs': round(np.random.uniform(0.1, 100.0), 1),
            'fat': round(np.random.uniform(0.1, 40.0), 1),
            'origin_country': fake.country(),
            'certification': np.random.choice(['USDA Organic', 'EU Organic', 'None', 'Non-GMO']),
            'description': generate_description(is_organic),
            'is_organic': is_organic,
            'last_updated': fake.date_between(start_date='-2y', end_date='today')
        }
        data.append(item)
    
    return pd.DataFrame(data)

def generate_description(is_organic):
    organic_phrases = [
        "Organically grown", "Pesticide-free", "Sustainable farming",
        "No artificial additives", "Non-GMO verified", "Eco-friendly packaging"
    ]
    conventional_phrases = [
        "Conventionally farmed", "Great value", "Family favorite",
        "Locally sourced", "Fresh daily", "Premium quality"
    ]
    
    base = f"A {np.random.choice(['delicious', 'fresh', 'juicy', 'crisp', 'creamy'])} "
    base += np.random.choice(['product', 'food item']) + " "
    
    if is_organic:
        base += f"that is {np.random.choice(organic_phrases)}. "
    else:
        base += f"that is {np.random.choice(conventional_phrases)}. "
    
    base += "Perfect for your daily nutrition needs."
    return base

if __name__ == "__main__":
    df = generate_food_data(1500)
    df.to_csv('data/raw/food_dataset.csv', index=False)
