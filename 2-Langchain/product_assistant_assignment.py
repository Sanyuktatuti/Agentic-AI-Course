### ASSIGNMENT SOLUTION - Product Assistant

# Import required libraries
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from typing import Union

# Define the Pydantic model for product information
class ProductInfo(BaseModel):
    product_name: str = Field(description="Name of the product")
    product_details: str = Field(description="Detailed description of the product")
    tentative_price: Union[int, float] = Field(description="Tentative price in USD")

# Initialize the output parser with our Pydantic model
output_parser = PydanticOutputParser(pydantic_object=ProductInfo)

# Create the ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful product assistant. Extract product information from the user query and provide it in the specified format. {format_instructions}"),
    ("user", "{query}")
])

# Initialize the LLM
llm = ChatOpenAI(model="gpt-4o")

# Create the chain
chain = prompt | llm | output_parser

# Test the assistant with different product queries
test_queries = [
    "I want to buy an iPhone 15 Pro Max",
    "Looking for a Samsung 4K Smart TV",
    "Need a gaming laptop with RTX 4080"
]

# Run the assistant
for query in test_queries:
    print(f"\n{'='*50}")
    print(f"Query: {query}")
    print(f"{'='*50}")
    
    try:
        result = chain.invoke({
            "query": query,
            "format_instructions": output_parser.get_format_instructions()
        })
        
        print(f"Product Name: {result.product_name}")
        print(f"Product Details: {result.product_details}")
        print(f"Tentative Price: USD {result.tentative_price}")
        
    except Exception as e:
        print(f"Error: {e}")

# Interactive version - you can test with your own queries
def get_product_info(query: str):
    """
    Function to get product information for any query
    """
    try:
        result = chain.invoke({
            "query": query,
            "format_instructions": output_parser.get_format_instructions()
        })
        return result
    except Exception as e:
        return f"Error: {e}"

# Example usage:
# result = get_product_info("Tell me about MacBook Air M2")
# print(result) 