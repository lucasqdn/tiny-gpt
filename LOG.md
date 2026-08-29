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

## Things to improve next
Text generation
Checkpoint saving
Averaged training evaluation, matching validation evaluation
Fixed validation batches so every run uses identical examples


# Run 4
## Values
steps = 50000
sequence_length = 128
d_model = 384
n_heads = 6
dropout = 0.0
n_layers = 8

## Run summary
Initial training loss: 5.3590
Final training loss: 0.7569
Average training loss: 0.7604
Average validation loss: 0.7378
Elapsed training time: 1188.33 seconds
Tokens per second: 43,086
Total tokens trained: 51,200,000

## Generated sample
Once upon a time, there was a little girl named Lily. One day, Lily was outside when her mommy accidentally past the ball down and said, "I can frown this car, but you must have passed them all." 

Molly smiled and said, "Okay, I love you!"

The carefully tasted the cord and dressed and the lawyer went to play, fee

## Verdict

It maintained correct sentence structures as well as grammer
But the story gradually loses its plot
(Maybe because of 128 characters context which is not that much) so it loses its details quickly

## Things to work on

Add EOS token between stories
Pack stories together instead of discarding those less than 129 characters


# Run 5
Fixed using EOS tokens and incorporate stories less than 128 characters by creating a stream

## Values
steps = 50000
sequence_length = 128
d_model = 384
n_heads = 6
dropout = 0.0
n_layers = 8

## Run summary
Initial training loss: 5.1649
Final training loss: 0.7933
Average training loss: 0.7494
Average validation loss: 0.7431
Elapsed training time: 1220.46 seconds
Tokens per second: 41,951
Total tokens trained: 51,200,000

## Verdict
The model successfully learn how to use EOS

## What to improve on next run
Switch to BPE, maybe 2048 tokens
Still keeping other values the same