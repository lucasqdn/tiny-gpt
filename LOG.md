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

# BPE Optimization
Priority queue with linked-list indexing
The current problem is that if a story needs 200 merges, the story will be rebuilt and scan 200 times
1. Finds all mergable pairs once
2. Stores them in a priority queue
3. Merges the highest priority pair
4. Updates the 2 neighbouring pairs affected by the merge


# Run 6 (BPE Run)
steps = 50000
vocab_size = 2048 (BPE)
sequence_length = 128
d_model = 384
n_heads = 6
dropout = 0.0
n_layers = 8

## Run summary
Initial training loss: 7.6950
Final training loss: 2.1878
Average training loss: 2.2547
Average validation loss: 2.2663
Elapsed training time: 1512.71 seconds
Tokens per second: 33,847
Total tokens trained: 51,200,000

## Generated sample:
Once upon a time. Lucy tried trying to resist, but it didn't work. She tried and tried, but it still still landed in the ground. She started to get frustrated.

Suddenly, Lucy saw something unusual. It was a big green bunny! It was cutting from a leaf. It was the owner's bunny's name. The bunny was very happy to see Lucy and kept her promise. Lucy was happy to have her promise!

## Verdict
Initial loss roughly ln(2048)
Lower token throughput
But 4.174 source bytes per BPE token
51.2 million BPE tokens --> 214 million source bytes
4.17x more text than run 5
Effective textual throughput is 33.847 * 4.174 = 141000

## Evaluate.py
20/20 stopped at EOS
Average story length: 92.2 BPE tokens
Reached story endings rather than stopping early
More coherent in stories but still lacking

## Things to improve on
- Semantic coherence and word repetition
- Malformed grammar and words "timerid" or "teachingme"

## Final conclusion
Run 6 improved effective context, textual throughput and reliable story completition


# Run 7

## Run summary
Initial training loss: 7.6867
Final training loss: 2.9596
Average training loss: 3.0176
Average validation loss: 3.0411
Elapsed training time: 74.60 seconds
Tokens per second: 68,634
Total tokens trained: 5,120,000

## Generated sample:
Once upon a time. Lucy tried them all about her upet to go home inback to the world. She scollected them together all the different things day it would get her find her way home. So she went back home. 

When it was time for writ, itchies were out there. 

"Wow!" said Sam. "Thank you to go home now" and go outside and explore the big field.
Stopped at EOS: True
Generated tokens: 80

## Verdict
Throughput sped up by 2.03x --> roughly 286500 source bytes/second

## What to work on next
- Pretokenize the data
- BF16
- Test larger batch sizes


# Run 8
## Run summary
Initial training loss: 7.6972
Final training loss: 2.2453
Average training loss: 2.3109
Average validation loss: 2.2981
Elapsed training time: 105.84 seconds
Tokens per second: 387,013
Total tokens trained: 40,960,000

## Generated sample:
Once upon a time, a fox was all alone and very big I was scared. I wanted to be your friend, Tim." Tim could not believe his eyes. He tried to be a giraffe, but it was too fast. Tim was too small and scared to go back to his friends.

Tim's friends noticed how sad he was. They came to come with him. The giraffe said, "Timmy, let's make a zigzag safe place!" Tim and his friends went to the zigzag. They used white paws to put on books and paper in, and sang. One, two, the zigaffed giraffe came up with a cat inside. It looked like the mouse had pictures. Timmy felt happy knowing he did something.

From that day on, Timmy and his friends would pass up in the zigzag and watch the cat for a while. They would admire their games together and sing around it. Timmy showed his new team full of the zigzage that was the one who could play with him. He thanked Tim and his friends for letting them play in their opconyst.
Stopped at EOS: True
Generated tokens: 231

## Verdict
The model trained very well at 64 batches
Runs at 387000 tokens/s, 5 times faster than run 7