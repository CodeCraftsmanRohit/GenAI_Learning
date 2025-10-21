# GenAI_Learning

An LLM, or Large Language Model, is a type of artificial intelligence (AI) program that can understand, process, and generate human-like text. These models are "large" because they are trained on massive amounts of text data, often billions of words, from the internet, books, and other sources.

Think of an LLM as an incredibly advanced auto-complete. At its core, it's a statistical tool that calculates the probability of what word (or part of a word, called a "token") should come next in a sequence.

How LLMs Work
Training: An LLM is built using a type of neural network called a Transformer. During its training, it analyzes a vast dataset of text, learning grammar, facts, reasoning patterns, and the complex relationships between words.

Self-Attention: A key mechanism called "self-attention" allows the model to weigh the importance of different words in a sentence, no matter how far apart they are. This helps it grasp context (e.g., knowing what "it" refers to in a long paragraph).

Generating Text: When you give an LLM a prompt (an instruction or question), it analyzes your text and begins generating a response one token at a time, repeatedly predicting the most likely next token to form coherent sentences and paragraphs.

What LLMs Are Used For
LLMs are the technology behind many modern AI tools and have a wide range of applications:

Chatbots & Virtual Assistants: Powering conversational AI like Gemini or ChatGPT.

Content Creation: Writing emails, articles, marketing copy, and even poetry.

Summarization: Condensing long documents or articles into key points.

Translation: Translating text from one language to another.

Coding: Writing, debugging, and explaining computer code.

Sentiment Analysis: Determining the emotional tone (positive, negative, neutral) of a piece of text.

transformer-> for only input token it just predict the only next input token and just keep repeating the process

llm models require gpt

computers are good in math  , tokenization is diff. for diff. models 
TIKTOKENIZER

numbers are given to tranformer and transformer just predict next token
Converting the user input to a set of numbers understandable by llm is known as tokenization ,
detokenization converts it back to text

tiktoken made by openAi helps to tokenizse and detokenize the text 

Vector Embeddings give semantic meanings to tokens

Positional Encoding ensures positions of the vectors are maintained

Self attention ensures vectora talk to each other
Multi-Head attention where we keep attention on multiple aspects of a thing 

Linear-> probability
softmax-> taking out the most probable answer