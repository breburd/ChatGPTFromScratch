## ChatGPT from Scratch Final Project
A rotary positional embedding custom class has been implemented
within the GPT model that was built for this course. The previous 
implementation only used absolute positional embeddings, which
is more speed efficient but may miss important relationships between
two words. 

The **Ro**tary **P**osition **E**mbedding (RoPE) 
was implemented within the `rotary_position_embedding.py` Python file
using the PyTorch library. Changes were also made to the 
`TransformerDecoderBlock` class within the `gpt.py` Python file 
because RoPE embeddings are applied several times within the
`transfer decoder` in order to "keep the position information 
'fresh'" (Module 3 notes, slide 45).

The `train_model.py` file that was created in Module 6 is used 
to train the new GPT model. The losses, total tokens, and model 
weights are saved periodically because it takes a long time to 
train the model without a GPU, and I did not have access to a
GPU. The `main.py` Python file has been created for the purpose
of calling `train_model.train()` so that other users are readily
able to train the model with ease. This file contains a for loop 
that performs all the experiments tested (trains on 2, 4, 8, and 
12 transformer layers).

No additional libraries are required for the implementation
of the RoPE algorithm.

The analysis performed to compare the RoPE implementation to the 
absolute position embeddings is included for reference within the 
`FinalProjectAnalysis.ipynb` Jupyter Notebook file.