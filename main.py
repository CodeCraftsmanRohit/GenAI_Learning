import tiktoken

enc=tiktoken.encoding_for_model("gpt-4o")

text="Hey There, My name is Rohit"
tokens=enc.encode(text)
print("Tokens",tokens)

decoded=enc.decode( [25216, 3274, 11, 3673, 1308, 382, 65416, 278])
print(decoded)