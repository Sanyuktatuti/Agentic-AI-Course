from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import Union, Optional

model = ChatOpenAI(temperature=0.7)

class Product(BaseModel):
    product_name: str = Field(description="The name of the Product")
    product_details: str = Field(description="The details of the Product")
    tentative_price: Optional[Union[int, float]] = Field(description="The tentative price of the product in USD (provide a realistic price)")

parser = PydanticOutputParser(pydantic_object=Product)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that can answer questions about products and provide answers in a specified format. Always provide realistic prices for products. {format_instructions}"),
    ("user", "{query}")
])

chain = prompt | model | parser

# Interactive function to get product information
def get_product_info(query: str):
    """
    Function to get product information for any query.
    """
    try:
        result = chain.invoke({
            "query": query,
            "format_instructions": parser.get_format_instructions()
        })
        return result
    
    except Exception as e:
        return f"Error: {e}"

# Main interactive loop
def main():
    print("=" * 60)
    print("🤖 PRODUCT ASSISTANT")
    print("=" * 60)
    print("Ask me about any product and I'll provide:")
    print("• Product Name")
    print("• Product Details") 
    print("• Tentative Price in USD")
    print("=" * 60)
    
    while True:
        # Get user input
        user_query = input("\n🔍 Enter your product query (or 'quit' to exit): ").strip()
        
        # Check if user wants to quit
        if user_query.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Thank you for using Product Assistant!")
            break
        
        # Check if query is empty
        if not user_query:
            print("❌ Please enter a valid query.")
            continue
        
        print(f"\n{'='*50}")
        print(f"Query: {user_query}")
        print(f"{'='*50}")
        
        # Get and display result
        try:
            result = get_product_info(user_query)
            
            if isinstance(result, str) and result.startswith("Error:"):
                print(f"❌ {result}")
            else:
                print(f"✅ Product Name: {result.product_name}")
                print(f"📝 Product Details: {result.product_details}")
                print(f"💰 Tentative Price: USD {result.tentative_price}")
                
        except Exception as e:
            print(f"❌ Error: {e}")

# Run the interactive program
if __name__ == "__main__":
    main()

# Alternative: Simple one-time query
def simple_query():
    """
    Simple function for one-time product queries
    """
    query = input("Enter your product query: ")
    result = get_product_info(query)
    
    if isinstance(result, str) and result.startswith("Error:"):
        print(f"Error: {result}")
    else:
        print(f"\nProduct Name: {result.product_name}")
        print(f"Product Details: {result.product_details}")
        print(f"Tentative Price: USD {result.tentative_price}")

# Uncomment the line below for simple one-time query instead of interactive loop
# simple_query() 