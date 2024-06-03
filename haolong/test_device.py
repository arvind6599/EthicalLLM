from transformers import pipeline


if __name__ == "__main__":
    pipe_SL = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.2", device_map="auto",
                    # quantization_config=quantization_config, # remove this after
                    )
    pipe_constitution = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.2", device_map="auto",
                                # quantization_config=quantization_config, # remove this after
                                )
    
    print(pipe_SL.device)
    print(pipe_constitution.device)