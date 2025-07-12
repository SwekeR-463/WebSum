import base64, asyncio
from dotenv import load_dotenv
from typing import Annotated, Sequence, List, TypedDict, Union
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from playwright.async_api import async_playwright, Page, Browser


load_dotenv()

browser = Union[Browser, None] 
page = Union[Page, None]

# agent state -> which will be passed between nodes
class AgentState(TypedDict):
    # for convo
    messages: Annotated[Sequence[BaseMessage], add_messages]
    url: Union[str, None]
    # for screenshot while on webpage
    current_ss: Union[List[str], None]
    # for summary
    summaries: Annotated[Sequence[BaseMessage], add_messages]
    # for scroll decision -> whether to scroll more or not
    scroll_decision: Union[str, None]
    task: str


async def initialize_browser():
    """
    initialize browser(Chrome) with page
    """

    global browser, page
    print('====Initializing the Browser====')

    try:
        pw = await async_playwright().start()
        browser = await pw.chromium.launch(headless = False)

        page = await browser.new_page()
        print('====Browser Initialized====')

    except Exception as e:
        print(f'Failed to initialized browser due the following exception: {e}')

async def close_browser():
    """
    closes the Browser.
    """

    global browser, page

    if browser:
        print('====Closing the Browser====')
        try:
            await browser.close()
            print('====Browser Closed====')
        except Exception as e:
            print(f'Error in closing the browser:{e}')

        finally:
            browser = None
            page = None


@tool
async def navigate_url(url: str) -> str:
    """
    tool takes the browser to navigate to the URL provided
    """
    global page
    print('====Navigating to URL====')
    try: 
        await page.goto(url, wait_until = 'domcontentloaded')  
        return f"====Navigated to URL===="  
      
    except Exception as e:
        return f'The Error that occured during navigating url is:{e}'

# why returning strings? -> coz using base64

@tool
async def take_ss() -> str:
    """
    takes screenshot of the current page
    """

    global page

    if page is None:
        return '====Browser page not initialized===='
    
    else:
        print('====Taking Screenshot====') 

        try:
            binary_ss = await page.screenshot()
            b64_ss = base64.b64encode(binary_ss).decode("utf-8")

            print('====Screenshot successfully captured====')
            return b64_ss
        
        except Exception as e:
            return f'Error that occured during taking screenshot:{e}' 


@tool
async def scroll_down() -> str:
    """
    scrolls the page down by a fixed amount.
    """
    global page

    if page is None:
        return "====Page not initialised===="

    viewport_height = await page.evaluate("window.innerHeight")
    scroll_amount   = int(viewport_height * 0.8)

    await page.evaluate(f"window.scrollBy(0, {scroll_amount});")
    return f"====Scrolled {scroll_amount}px===="

agent_tools = [navigate_url, take_ss, scroll_down]
llm = ChatGoogleGenerativeAI(model = 'gemini-2.5-flash').bind_tools(tools = agent_tools)

async def init_node(state: AgentState) -> AgentState:
    """
    initializes broswer and navigates to the provided URL
    """

    print('====Initial Node====')
    task = state['task']

    base_url = "https://www.akramz.space/p/on-applied-researchhtml"

    await initialize_browser()
    navigate_output = await navigate_url.ainvoke(base_url)

    return {
        **state,
        'url': base_url,
        'messages': [SystemMessage(content=f'Navigated to the provided URL:{base_url}. {navigate_output}')]
    }

async def ss_node(state: AgentState) -> AgentState:
    """
    takes screenshot of the current page & stores it in the state as a List
    """

    print('====Screenshot Node====')
    try:
        b64_ss = await take_ss.ainvoke(input=None)
        print("====Screenshot captured and returned from tool====")

        current_ss_list = state.get("current_ss") 
        if current_ss_list is None:
            current_ss_list = []

        current_ss_list.append(b64_ss)

        updated_messages = [SystemMessage(content= "Screenshot captured and saved to state variable.")]

        return {
            **state,
            "current_ss": current_ss_list,
            "messages": updated_messages,
        }

    except Exception as e:
        error_msg = f"Error during ss_node: {e}"
        print(error_msg)
        return {
            **state,
            "messages": [SystemMessage(content = error_msg)]
        }



async def summarizer_node(state: AgentState) -> AgentState:
    """
    LLM summarizes the current screenshot and page state
    the screenshot is sent as base64 image to LLM
    """

    print("====Summarizer Node====")
    task = state.get("task", "Summarize this page as briefly as possible")
    screenshots = state.get("current_ss")

    if not screenshots:
        print("====No screenshot available to summarize====")
        return {
            **state,
            "summaries": [SystemMessage(content= "No screenshot available for summarization")]
        }

    latest_ss = screenshots[-1]

    user_prompt = HumanMessage(content=[
        {"type": "text", "text": f"Summarize this screenshot for the following task:{task}"},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{latest_ss}"}}
    ])

    try:
        summary = await llm.ainvoke([user_prompt])
        print("====Summary Generated====")

        return {
            **state,
            "summaries":[summary],
            "messages": [SystemMessage(content="Page summary generated.")]
        }

    except Exception as e:
        error_msg = f"Error during summarization: {e}"
        print(error_msg)
        return {
            **state,
            "messages": [SystemMessage(content=error_msg)]
        }




async def scroll_decision_node(state: AgentState) -> AgentState:
    """
    LLM decides whether to scroll more or not
    """
    global page
    if page is None:
        return {**state,
                "messages": state["messages"]
                + [SystemMessage(content="Scroll skipped - page not initialised.")]}
    
    before = await page.evaluate("window.scrollY")

    tool_result = await scroll_down.ainvoke(input=None)

    after  = await page.evaluate("window.scrollY")
    moved  = after - before

    return {**state,
            "messages": state["messages"]
            + [SystemMessage(content=f"{tool_result}  (Δy = {moved}px)")]}




async def aggregate_node(state: AgentState) -> AgentState:
    """
    aggregates all summaries into a final report.
    """
    print("====Aggregation Node====")
    summaries = state.get("summaries", [])
    task = state.get("task", "")

    messages = [
        SystemMessage(content=f"Aggregate the following summaries for the task: {task}"),
        HumanMessage(content="\n\n".join([msg.content for msg in summaries if hasattr(msg, "content")]))
    ]

    try:
        final_summary = await llm.ainvoke(messages)
        print("====LLM aggregation successful====")
        return {
            **state,
            "messages": state["messages"] + [SystemMessage(content="Final summary created"), final_summary],
        }

    except Exception as e:
        return {
            **state,
            "messages": state["messages"] + [SystemMessage(content=f"Error during aggregation: {e}")],
        }


def route_scroll_decision(state: AgentState) -> str:
    """
    routes based on scroll decision or scroll count.
    """
    print("Routing based on scroll decision")

    # check if browser initialization failed, as we cannot scroll in that case
    messages = state.get("messages", [])
    init_failed = any("Browser initialization failed." in msg.content for msg in messages if isinstance(msg, SystemMessage))

    if init_failed:
        print("Browser initialization failed, routing to aggregate.")
        return "aggregate"

    # force the routing to the 'scroll' node for testing the scroll tool
    print("Forcing route to 'scroll' node.")
    return "scroll"


workflow = StateGraph(AgentState)

# nodes
workflow.add_node("init", init_node)
workflow.add_node("screenshot", ss_node)
workflow.add_node("summarizer", summarizer_node)
workflow.add_node("decide_scroll", scroll_decision_node)
workflow.add_node("scroll", lambda state: scroll_down.ainvoke(input=None).then(lambda _: state))
workflow.add_node("aggregate", aggregate_node)

# entry
workflow.set_entry_point("init")

# flow
workflow.add_edge("init", "screenshot")
workflow.add_edge("screenshot", "summarizer")
workflow.add_edge("summarizer", "decide_scroll")

workflow.add_conditional_edges(
    "decide_scroll",
    route_scroll_decision,
    {
        "scroll": "screenshot",     
        "aggregate": "aggregate"    
    }
)

workflow.add_edge("aggregate", END)

app = workflow.compile()


async def run_graph():
    initial_state = {
        "messages": [],
        "url": None,
        "current_ss": [],
        "summaries": [],
        "scroll_decision": None,
        "task": "Explain the steps in brief about doing applied ML research from this blog"
    }
    print("\n====Starting LangGraph Agent====\n")

    try:
        # use astream to see the state changes step-by-step
        # set recursion_limit to prevent infinite loops
        async for step in app.astream(initial_state, {"recursion_limit": 75}): 
            step_name = list(step.keys())[0]
            print(f"\n====Step: {step_name}====")

            latest_state = step[step_name]

            # print specific information based on the node that just completed
            if step_name == "summarizer":
                if latest_state.get('summaries'):
                    latest_summary_message = latest_state['summaries'][-1]
                    if isinstance(latest_summary_message, (AIMessage, HumanMessage)) and latest_summary_message.content:
                        print(">>> Individual Screenshot Summary:")
                        print(latest_summary_message.content)
                    elif isinstance(latest_summary_message, SystemMessage):
                        print(">>> Summarizer Status:", latest_summary_message.content)


            elif step_name == "decide_scroll":
                decision = latest_state.get('scroll_decision')
                print(f">>> Scroll Decision: {decision}")

            elif step_name == "aggregate":
                # the aggregation node adds the final summary as a HumanMessage to the messages list
                final_summary_message = None

                for msg in reversed(latest_state.get('messages', [])):
                    if isinstance(msg, HumanMessage) and final_summary_message is None:
                        final_summary_message = msg 
                    elif isinstance(msg, SystemMessage) and msg.content == "Final summary created." and final_summary_message is not None:
                        print(">>> Final Aggregated Summary:")
                        print(final_summary_message.content)
                        break 
                    
                # fallback in case the heuristic fails or no valid summary was produced    
                if final_summary_message is None:
                    print(">>> Aggregation Node Finished (No valid final summary found in messages).")
                    


    except Exception as e:
        print(f"\n====An error occurred during graph execution: {e}====")
    finally:
        print("\n====Agent execution finished. Attempting to close browser====")
        await close_browser()


if __name__ == "__main__":
    asyncio.run(run_graph())