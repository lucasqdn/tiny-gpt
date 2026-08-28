# Training 1 (test):
steps = 1000
sequence_length = 128
d_model = 384
n_heads = 6
dropout = 0.0
n_layers = 8

Result: Final loss at 1.8718
Initialization at the start produce massive loss at 233.25

## What to improve
- Add in training loss and validation loss
- Don't skip stories shorter than 129 characters
- Sample stories more logically and better
- Provide end of story token
- Teach the model when to stop
- Encode stories ahead of time --> Pack into token stream --> Slice training from the stream


# Run 2
## Values
steps = 1000
sequence_length = 128
d_model = 384
n_heads = 6
dropout = 0.0
n_layers = 8

## Summary
Initial training loss: 5.0794
Final training loss: 1.5999
Average validation loss: 1.5145
Elapsed training time: 25.62 seconds
Tokens per second: 39,976
Total tokens trained: 1,024,000

# Run 3
## Values
steps = 10000
sequence_length = 128
d_model = 384
n_heads = 6
dropout = 0.0
n_layers = 8

## Summary
Initial training loss: 5.0794
Final training loss: 0.7628
Average validation loss: 0.9019
Elapsed training time: 244.58 seconds
Tokens per second: 41,868
Total tokens trained: 10,240,000

## Questions
Character perplexity and bits per character