import pandas as pd
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain import hub
from langchain_openai import ChatOpenAI
from config import OPENAI_API_KEY
from .data_agent import DataAnalysisAgent

class InsightAgent:
    def __init__(self, df):
        self.df = df
        self.llm = ChatOpenAI(
            model="gpt-4-turbo",
            temperature=0,
            openai_api_key=OPENAI_API_KEY
        )
        self.data_agent = DataAnalysisAgent(df)
        self.tools = self._create_tools()
        self.agent = self._create_agent()
    
    def _create_tools(self):
        return [
            Tool(
                name="Data Analysis",
                func=self.data_agent.analyze,
                description="Useful for statistical analysis of food data"
            ),
            Tool(
                name="Organic Pricing",
                func=self._get_organic_pricing,
                description="Get pricing insights for organic vs conventional products"
            ),
            Tool(
                name="Nutrition Comparison",
                func=self._compare_nutrition,
                description="Compare nutritional profiles between organic and conventional foods"
            )
        ]
    
    def _create_agent(self):
        prompt = hub.pull("hwchase17/react")
        agent = create_react_agent(self.llm, self.tools, prompt)
        return AgentExecutor(agent=agent, tools=self.tools, verbose=True)
    
    def _get_organic_pricing(self, query):
        org_price = self.df[self.df['is_organic'] == 1]['price'].mean()
        conv_price = self.df[self.df['is_organic'] == 0]['price'].mean()
        premium = org_price - conv_price
        premium_pct = (premium / conv_price) * 100
        return f"Organic Premium: ${premium:.2f} ({premium_pct:.1f}%)"
    
    def _compare_nutrition(self, query):
        nutrients = ['calories', 'protein', 'carbs', 'fat']
        results = {}
        for nutrient in nutrients:
            org_mean = self.df[self.df['is_organic'] == 1][nutrient].mean()
            conv_mean = self.df[self.df['is_organic'] == 0][nutrient].mean()
            results[nutrient] = {
                'organic': org_mean,
                'conventional': conv_mean,
                'difference': org_mean - conv_mean
            }
        return results
    
    def query(self, question):
        response = self.agent.invoke({"input": question})
        return response['output']

# Example usage:
# agent = InsightAgent(df)
# print(agent.query("What's the price difference between organic and conventional fruits?"))
