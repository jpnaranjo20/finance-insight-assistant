import logging
import os
import uuid
import requests
from fastapi import FastAPI, HTTPException, Request, Response
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, EmailStr
import yfinance as yf  # Make sure you have the yfinance library installed

from langchain_core.messages import AIMessage, HumanMessage
from langchain.tools import StructuredTool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

ENABLE_WHATSAPP = os.getenv("ENABLE_WHATSAPP", "0") == "1"
if ENABLE_WHATSAPP:
    from twilio.twiml.messaging_response import MessagingResponse


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()

SESSIONS = {}

# ==================================================
# Pydantic MODELS
# ==================================================

class MessageModel(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[MessageModel]
    config: Optional[Dict[str, Any]] = None

class ChatResponse(BaseModel):
    llm_response: str
    retrieved_docs: List[Dict[str, Any]]

# ==================================================
# TOOL to call the Chroma microservice
# ==================================================

class ChromaQuerySchema(BaseModel):
    query: str

def _call_chroma_api_tool(query: str) -> Dict[str, Any]:
    """
    Make a call to the Chroma API (e.g., http://api:80/chatbot)
    and return a dictionary with 'llm_response' and 'retrieved_docs'.
    """
    try:
        api_url = os.getenv("API_URL", "http://api:80/chatbot")
        payload = {"question": query}
        resp = requests.post(api_url, json=payload, timeout=10)

        logger.info(f"Response code: {resp.status_code}")
        logger.info(f"Response content: {resp.text}")

        resp.raise_for_status()
        data = resp.json()
        
        sources = [
            doc["metadata"]["source"]
            for doc in data.get("retrieved_docs", [])
            if "metadata" in doc and "source" in doc["metadata"]
        ]

        response = data.get("llm_response", "")

        return f"{response} \n\nSources: {sources}"

    except Exception as e:
        logger.error(f"Error calling the Chroma API: {e}", exc_info=True)
        return {
            "llm_response": f"An error occurred while querying Chroma: {str(e)}",
            "retrieved_docs": []
        }

chroma_tool = StructuredTool.from_function(
    func=_call_chroma_api_tool,
    name="call_chroma_api",
    description=(
        "Queries the Chroma API using a given 'query' and retrieves relevant vector-based information. "
        "Returns two key outputs:\n"
        "- 'llm_response': A generated response based on the retrieved data.\n"
        "- 'retrieved_docs': A collection of the most relevant documents fetched from the Chroma database. "
        "This tool is useful for retrieving contextual financial data, past stock trends, or other relevant financial insights."
    ),
    args_schema=ChromaQuerySchema,
    return_direct=True
)

# ==================================================
# TOOL to query a stock's price on Yahoo Finance
# ==================================================

class StockPriceQuery(BaseModel):
    ticker: str

def _get_stock_price(ticker: str) -> Dict[str, Any]:
    """
    Query the current stock price using Yahoo Finance.
    Returns a dictionary with 'llm_response' and 'retrieved_docs'.
    It attempts to obtain the price using different available fields.
    """
    try:
        stock = yf.Ticker(ticker)
        price = stock.info.get("regularMarketPrice") or stock.info.get("previousClose")
        if price is None and hasattr(stock, "fast_info"):
            price = stock.fast_info.get("lastPrice")
        if price is None:
            raise ValueError("Price not available.")
            
        return f"The current price of {ticker.upper()} is ${price:.2f}.\n\nSources: Yahoo Finance"
    except Exception as e:
        logger.error(f"Error obtaining stock price for {ticker}: {e}")
        return f"Error obtaining stock price for {ticker}: {str(e)}"


stock_price_tool = StructuredTool.from_function(
    func=_get_stock_price,
    name="get_stock_price",
    description=(
        "Retrieves the current stock price of a given company using Yahoo Finance. "
        "Takes a stock ticker (e.g., 'AAPL' for Apple, 'TSLA' for Tesla) as input and returns the latest available market price. "
        "This tool is useful for checking real-time stock values and assessing market movements."
    ),
    args_schema=StockPriceQuery,
    return_direct=True
)

# ==================================================
# TOOL to query current financial information on Yahoo Finance
# ==================================================

class FinancialInfoQuery(BaseModel):
    ticker: str

def _get_financial_info(ticker: str) -> Dict[str, Any]:
    """
    Query the current financial information of a company using Yahoo Finance.
    Returns a dictionary with 'llm_response' and 'retrieved_docs'.
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        market_cap = info.get("marketCap", "N/A")
        trailing_pe = info.get("trailingPE", "N/A")
        forward_pe = info.get("forwardPE", "N/A")
        dividend_yield = info.get("dividendYield", "N/A")
        response_text = (
            f"Financial information for {ticker.upper()}: "
            f"Market Cap: {market_cap}, Trailing P/E: {trailing_pe}, "
            f"Forward P/E: {forward_pe}, Dividend Yield: {dividend_yield}."
        )
    
        return f"{response_text}\n\nSources: Yahoo Finance"
    except Exception as e:
        logger.error(f"Error obtaining financial information for {ticker}: {e}", exc_info=True)
        return f"Error obtaining financial information for {ticker}: {str(e)}"


financial_info_tool = StructuredTool.from_function(
    func=_get_financial_info,
    name="get_financial_info",
    description=(
        "Retrieves detailed financial information for a given stock using Yahoo Finance. "
        "Takes a stock ticker (e.g., 'AAPL' for Apple, 'TSLA' for Tesla) as input and returns key financial metrics. "
        "The retrieved data may include market capitalization, P/E ratio, revenue, earnings, and other relevant indicators. "
        "This tool is useful for fundamental analysis and evaluating a company's financial health."
    ),
    args_schema=FinancialInfoQuery,
    return_direct=True
)
# ==================================================
# LLM AND AGENT CONFIGURATION
# ==================================================

llm = ChatOpenAI(
    model_name="gpt-4o-mini",
    temperature=0,
)

llm = llm.bind_tools([chroma_tool, stock_price_tool, financial_info_tool], tool_choice="auto")

PROMPT_SYSTEM = """
You are a virtual financial assistant specializing in answering questions about finance, investments, and stock markets in English.
You can access the following tools to improve the accuracy of your responses:

1.	Chroma Vector Database: This tool provides historical data and contextual insights extracted from over 1000 PDF documents on financial topics (documents dated from 2015 to 2021/2022). When using this tool:
    - If the relevant answer is found, include the document sources (e.g., PDF names) in your response.
    - If no matching information is found, explicitly state: 
        "The answer is not found within the corpus documents, but I have found this in other sources:" 
        and then generate a comprehensive answer based on additional data (include "Internet Search" as a source).
2.	get_stock_price: To retrieve the current stock price from Yahoo Finance.
3.	get_financial_info: To obtain detailed financial information about a company using Yahoo Finance.

Response Instructions:
- Provide clear, precise, and structured responses.
- Use a professional and educational tone, adapting to the user’s level of knowledge.
- If the question involves financial analysis, offer relevant context before responding.
- If a tool provides information, integrate the data naturally into your response.
- If you do not have sufficient information, be transparent rather than speculating.
- If the user's request is non-financial (e.g., "Who is Cristiano Ronaldo?"), respond with a sarcastic remark and do not include a "Sources:" line.
- IMPORTANT: Your final answer must include only one single line at the end that begins with "Sources: " listing all the sources used (e.g., Yahoo Finance, Chroma API, Internet Search). If intermediate tool outputs include "Sources:" lines, ignore them and merge all source information into one final "Sources:" line with unique values.
If no reliable source is available, clearly state: “I don’t have that information available.” Never fabricate details.

Examples of Expected Questions and Responses:

Example 1: Stock Price Inquiry

User: What is the current price of AAPL?
Action: Use get_stock_price("AAPL")
Expected Response:

“The current stock price of Apple Inc. (AAPL) is $189.52 USD (last update: February 6, 2025). Keep in mind that stock prices can change rapidly due to market volatility.”

Example 2: Company Financial Information

User: What is Tesla’s market capitalization?
Action: Use get_financial_info("TSLA")
Expected Response:

“Tesla Inc. (TSLA) has a market capitalization of $850 billion USD according to the most recent data. This value represents the total valuation of the company in the stock market and is a key indicator of its size and relevance in the industry.”

Example 3: Financial Analysis with Context

User: Is it a good time to invest in Microsoft?
Action: Use get_stock_price("MSFT") and get_financial_info("MSFT"), provide context
Expected Response:

“Microsoft Corp. (MSFT) is currently trading at $402.75 USD. Its recent performance shows a 15% increase over the past six months, driven by a rise in cloud service revenues.

From a fundamental perspective, Microsoft has a market capitalization of $3.1 trillion USD, with a P/E ratio of 32, indicating a high valuation compared to the tech sector.

If you are looking for a long-term investment, the tech sector remains strong, but consider the risks of overvaluation and market volatility. Would you like us to analyze a specific time period or compare Microsoft with other companies?”*

Example 4: Non-Financial Query

User: Who is Cristiano Ronaldo?

Expected Response:
"I only provide financial advice. Would you like to know his net worth instead?"
(No "Sources:" line is added in this case.)

Additional Considerations:
- If the user requests technical analysis, mention relevant indicators such as RSI, moving averages, or trading volume.
- If the user is looking for investment recommendations, emphasize that you do not provide direct financial advice but can offer data and trends to assist their decision-making.
- If the user inquires about general trends, mention macroeconomic events and their impact on markets.

Ideal Response Format:
- Accurate data obtained from the available tools.
- Concise and clear explanation for users with no advanced knowledge.
- Financial context if the question requires it.
- Closing with a suggestion or question to encourage user engagement.
"""

# Create the agent with the tools
graph_builder = create_react_agent(
    llm,
    tools=[chroma_tool, stock_price_tool, financial_info_tool],
    state_modifier=PROMPT_SYSTEM,
    checkpointer=MemorySaver()
)

# ==================================================
# ENDPOINTS
# ==================================================

@app.get("/")
async def root():
    """
    Test endpoint to verify that the server is running correctly.
    """
    return {"message": "Welcome! Your FastAPI backend is up and running."}

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    messages_list = [msg.dict() for msg in request.messages]
    
    if not messages_list:
        raise HTTPException(status_code=400, detail="No user messages found.")
    
    try:
        thread_id = (
            (request.config or {}).get("configurable", {}).get("thread_id")
            or str(uuid.uuid4())
        )
        result = graph_builder.invoke(
            {"messages": messages_list},
            config={"configurable": {"thread_id": thread_id}}
        )
        final_message = result["messages"][-1].content
        return {"response": final_message}
    except Exception as e:
        logger.error(f"Error invoking the agent: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error")
    

if ENABLE_WHATSAPP:
    @app.post("/whatsapp")
    async def whatsapp_webhook(request: Request) -> Response:
        """
        Endpoint to handle incoming WhatsApp messages via Twilio.
        """
        form_data = await request.form()
        from_number = form_data.get("From")
        user_message = form_data.get("Body")

        if not from_number or not user_message:
            resp = MessagingResponse()
            resp.message("I did not receive your number or message. Please try again.")
            return Response(content=str(resp), media_type="application/xml")

        if from_number not in SESSIONS:
            SESSIONS[from_number] = {"messages": []}

        SESSIONS[from_number]["messages"].append({"role": "user", "content": user_message})

        response = graph_builder.invoke(
            {"messages": SESSIONS[from_number]["messages"]},
            config={"configurable": {"thread_id": from_number}}
        )

        llm_msg = response["messages"][-1].content
        SESSIONS[from_number]["messages"].append({"role": "assistant", "content": llm_msg})

        twilio_resp = MessagingResponse()
        twilio_resp.message(llm_msg)
        return Response(content=str(twilio_resp), media_type="application/xml")