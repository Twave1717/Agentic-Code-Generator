from typing import TypedDict
import pandas as pd

from langchain_openai import ChatOpenAI
from langchain_experimental.tools.python.tool import PythonAstREPLTool
from langgraph.graph import StateGraph, START, END


# State 정의
class State(TypedDict):
    input: str
    data: str
    processed_data: str
    code: str
    is_error: bool
    result: str


# 각 Node에 해당하는 함수 정의
def load_data(state: State) -> State:
    # 데이터 로드 로직 구현
    data = pd.read_csv("./data/example.csv")
    return State(data=data)


def preprocess_data(state: State) -> State:
    # 데이터 전처리 로직 구현
    data = state["data"]
    processed_data = data.head(100)
    return State(processed_data=processed_data)


def analyze_data(state: State) -> State:
    # 데이터 분석 로직 구현
    model = ChatOpenAI(model="gpt-4o-mini")
    if state["is_error"] is False:
        response = model.invoke(
            (
                f"{state['input']} "
                "단, 오직 Code만 출력하세요. "
                f"데이터: {state['processed_data']} "
            )
        )
    else:
        response = model.invoke(
            (
                f"{state['input']} "
                "단, 오직 Code만 출력하세요. "
                f"데이터: {state['processed_data']} "
                "====================== "
                f"[AI] {state['code']} "
                f"{state['result']} "
                "====================== "
                "실행 중 발생한 버그를 참고하여 다시 코드를 작성해주세요."
            )
        )
    code = response.content
    return State(code=code)


def visualization(state: State) -> State:
    # 코드 실행
    import seaborn as sns
    import matplotlib.pyplot as plt

    code = state["code"]
    processed_code = None
    if code.startswith("```"):
        processed_code = "\n".join(code.split("\n")[1:-1])
    else:
        processed_code = code

    python_repl_tool = PythonAstREPLTool(locals={"sns": sns, "plt": plt})
    try:
        exec_result = str(python_repl_tool.invoke(processed_code))
        return State(result=exec_result, is_error=False)
    except Exception as e:
        return State(result=e, is_error=True)


def build_flow():
    # flow init
    flow = StateGraph(State)

    # node 정의
    flow.add_node("load_data", load_data)
    flow.add_node("preprocess_data", preprocess_data)
    flow.add_node("analyze_data", analyze_data)
    flow.add_node("visualization", visualization)

    # edge 연결
    flow.add_edge(START, "load_data")
    flow.add_edge("load_data", "preprocess_data")
    flow.add_edge("preprocess_data", "analyze_data")
    flow.add_edge("analyze_data", "visualization")

    # codintional edge 연결
    def is_error_check(state: State) -> str:
        return "analyze_data" if state["is_error"] else "end"

    flow.add_conditional_edges(
        "visualization",
        is_error_check,
        {"analyze_data": "analyze_data", "end": END},
    )

    # compile
    return flow.compile()


if __name__ == "__main__":
    flow = build_flow()
    initial_state = State(
        input="다음 데이터를 전문적으로 시각화하여 하나의 Figure로 표현해주세요.",
        is_error=False,
        code="",
        reulst="",
    )
    for event in flow.stream(initial_state, stream_mode="values"):
        for node_name, value in event.items():
            print(f"\n==============\nSTEP: {node_name}\n==============\n")
            print(value)
