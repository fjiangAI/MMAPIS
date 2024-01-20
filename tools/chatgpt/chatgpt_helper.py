import requests

class ChatGPTHelper:
    def __init__(self, api_key, base_url):
        """
        Initialize the ChatGPT helper with necessary credentials and settings.

        :param api_key: API key for authenticating with the ChatGPT service.
        :param base_url: Base URL of the ChatGPT API service.
        """
        self.api_key = api_key
        self.base_url = base_url

    def send_query(self, prompt, max_tokens=150):
        """
        Send a query to ChatGPT and get the response.

        :param prompt: The prompt or question to send to ChatGPT.
        :param max_tokens: Maximum number of tokens in the response.
        :return: The response text from ChatGPT.
        """
        # Implement the logic to send a query to ChatGPT
        # Typically involves making a POST request to the ChatGPT API
        # The implementation will depend on the specific API's requirements
        return "Response from ChatGPT"

    def summarize_text(self, text):
        """
        Use ChatGPT to summarize a given text.

        :param text: The text to be summarized.
        :return: The summarized text.
        """
        # Implement the summarization logic
        # This might involve formatting the prompt in a specific way to ask for a summary
        return self.send_query(f"Summarize the following text: {text}")

# Example usage
if __name__ == "__main__":
    api_key = "YOUR_API_KEY"
    base_url = "https://api.chatgpt.service.com"
    
    chatgpt = ChatGPTHelper(api_key, base_url)
    summary = chatgpt.summarize_text("Your long text here...")
    print(summary)
