# Fine tuning GPT2 for text summarization

[![Open in Layer](https://development.layer.co/assets/badge.svg)](https://app.layer.ai/aymane_hachcham/GPT2_text_summarization)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/12oT-CXV-ddtx6-jtIxWJJSoJd3FOnIa3?usp=sharing)

This project focuses on fine tuning GPT2 model to perform text summarization on the public Amanzon reviews dataset.

Make sure you installed the latest version of Layer:

``` 
!pip install layer --upgrade -q 
!pip install sentencepiece -q
!pip install transformers -q
```

Then you can fetch the model and the tokenizer from Layer and start summarizing text:

```python
import layer

def topk(probs, n=9):
    # The scores are initially softmaxed to convert to probabilities
    probs = torch.softmax(probs, dim= -1)
    
    # PyTorch has its own topk method, which we use here
    tokensProb, topIx = torch.topk(probs, k=n)
    
    # The new selection pool (9 choices) is normalized
    tokensProb = tokensProb / torch.sum(tokensProb)

    # Send to CPU for numpy handling
    tokensProb = tokensProb.cpu().detach().numpy()

    # Make a random choice from the pool based on the new prob distribution
    choice = np.random.choice(n, 1, p = tokensProb)
    tokenId = topIx[choice][0]

    return int(tokenId)

def model_infer(model, tokenizer, review, max_length=15):
    # Preprocess the init token (task designator)
    review_encoded = tokenizer.encode(review)
    result = review_encoded
    initial_input = torch.tensor(review_encoded).unsqueeze(0).to('cpu')

    with torch.set_grad_enabled(False):
        # Feed the init token to the model
        output = model(initial_input)

        # Flatten the logits at the final time step
        logits = output.logits

        # Make a top-k choice and append to the result
        result.append(topk(logits))

        # For max_length times:
        for _ in range(max_length):
            # Feed the current sequence to the model and make a choice
            input = torch.tensor(result).unsqueeze(0).to('cpu')
            output = model(input)
            logits = output.logits
            res_id = topk(logits)

            # If the chosen token is EOS, return the result
            if res_id == tokenizer.eos_token_id:
                return tokenizer.decode(result)
            else: # Append to the sequence 
                result.append(res_id)

    # IF no EOS is generated, return after the max_len
    return tokenizer.decode(result)

samples = [review.split('TL;DR')[0] for review in list(reviews[['training']].sample(n=1, random_state=1)['training'])]

gtp2_model = layer.get_model('aymane_hachcham/GPT2_text_summarization/models/gpt2_text_summarization:3.1').get_train()

# Then you can output the summary prediction from the samples using model_infer.
for review in samples:
    summaries = set()
    print(review)
    while len(summaries) < 3:
        summary = model_infer(model, tokenizer, review + " TL;DR ").split(" TL;DR ")[1].strip()
        if summary not in summaries:
            summaries.add(summary)
    print("Summaries: "+ str(summaries) +"\n")

```
Result:

Review 1
- Text Review: “Love these chips. Good taste,very crispy and very easy to clean up the entire 3 oz. bag in one sitting.  NO greasy after-taste.  Original and barbecue flavors are my favorites but I haven't tried all flavors.  Great product.”

- Associated Summary: {'very yummy', 'Love these chips!', 'My favorite Kettle chip'}



