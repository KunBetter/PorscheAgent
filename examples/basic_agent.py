import os
from porsche_agent import Agent, DeepSeekProvider, tool


@tool(description="Get the current weather for a city")
def get_weather(city: str) -> str:
    # Mock weather API — replace with real API call
    weather_data = {
        "beijing": "Sunny, 25°C",
        "shanghai": "Cloudy, 22°C",
        "tokyo": "Rain, 18°C",
    }
    return weather_data.get(city.lower(), f"No data for {city}")


def main():
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("Set DEEPSEEK_API_KEY environment variable")
        return

    llm = DeepSeekProvider(api_key=api_key)
    agent = Agent(
        llm=llm,
        tools=[get_weather],
        system_prompt="You are a helpful assistant. Answer in Chinese.",
    )
    result = agent.run("北京今天天气怎么样？")
    print(result)


if __name__ == "__main__":
    main()
