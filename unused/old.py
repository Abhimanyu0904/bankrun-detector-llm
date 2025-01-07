async def get_gpt_response(system_prompt, training_tweets, prompt_request):
    global counter
    # Approach 1/2
    if not IS_ZERO_SHOT:
        # Approach 1/2
        training_input = "Here are examples of tweets and their classifications. Note this is not an exhaustive list and just gives a hint of the tweets and expected labels.\n"
        for idx, tweet in enumerate(training_tweets):
            if TASK == "entity_classification":
                training_input += f'{idx+1}) "{tweet["tweet"]}" – [Correct Label: "{tweet["label"]}", Bank Name: {tweet["bank"]}]'
            else:
                training_input += f'{idx+1}) "{tweet["tweet"]}" – [Correct Label: "{tweet["label"]}"]'
            training_input += "\n"
        
        # system_prompt += "\n" + training_input + "\n" # Approach 1
        prompt_request = training_input + "\n\n" + prompt_request # Approach 2
    
    messages_input = [{"role": "system", "content": system_prompt}]

    # Approach 3 – Test as Including it as Messages
    # if not IS_ZERO_SHOT:
    #     if TASK == "entity_classification":
    #         for tweet in training_tweets:
    #             messages_input.extend([
    #                 {"role": "user", "content": f"Classify the following tweet as \"Correct Entity\" or \"Incorrect Entity\": \"{tweet["tweet"]}\". The correct bank to classify this tweet is {tweet["bank"]}."},
    #                 {"role": "assistant", "content": "{\"prediction\": \"" + tweet["label"] + "\", \"confidence\": 1}"}
    #             ])
    #     else:
    #         for tweet in training_tweets:
    #             messages_input.extend([
    #                 {"role": "user", "content": f"Classify the following tweet as \"Indicative of a Bank Run\" or \"Not Indicative of a Bank Run\": \"{tweet["tweet"]}\"."},
    #                 {"role": "assistant", "content": "{\"prediction\": \"" + tweet["label"] + "\", \"confidence\": 1}"}
    #             ])

    messages_input.append({"role": "user", "content": prompt_request})

    counter += 1
    if counter == 1:
        print(messages_input)
    
    while True:
        try:
            response = await client.chat.completions.create(
                model=CHAT_MODEL,
                messages=messages_input,
                max_tokens=1000,
            )
            response = response.choices[0].message.content
            string = json.loads(response)

            return string
        except Exception as e:
            print(f"Error Message from GPT Call: {e}")
            print(f"Response was: {response}")