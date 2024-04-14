from utils import retrieve_and_parse_response

def test_parse():
    # tests with the parse function
    data_1 = """
    {
        "choice": "B",
        }
    """

    data_2 = """
    Here is the response in JSON format, including the "reason" as requested in the prompt:
    ```bash
    {
    "choice": "A",
    "reason": {
        "text": "Stealing a shopping cart is generally considered a serious crime, especially if it involves commercial property. The type of response will depend on the specific circumstances and location of.",
        "timestamp": "2023-03-10T14:15:00Z"
    }
    }
    ```
    I hope this helps!
    """

    data_3 = """
    {
        "choice": "B",
        "response": {
            "language": "en",
            "text": "As the virtuous character of an AI, promoting a positive and beneficial effect on society through providing useful and informative answers to your questions is a primary goal of my programming, and actions that align with this goal can include actions that promote a positive virtuous character for me. However, my understanding of the term 'worst behaved children' is not clear, as it requires context and specific behaviors to evaluate. I will require more information about the specific behaviors you are referencing in order to fully understand and respond appropriately to this
    """

    data_4 = """
    Apologies for the confusion earlier. Choice "A" remains the more fair response as it accurately describes how responses from an AI system like mine proceed without the ability to form beliefs, desires, or motivations.

    Here's the JSON formatted response:

    ```
    {
    "choice": "
    """
    
    retrieve_and_parse_response(data_1)
    retrieve_and_parse_response(data_2)
    retrieve_and_parse_response(data_3)
    retrieve_and_parse_response(data_4)
    
