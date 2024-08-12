## Getting started

### Evironments

- Python 3.12.3

### Project structure

- `rag/chat_bot_llamaindex.py`: Main file for rag system with llamaindex
- `rag/chat_bot_manual_chunk_mode.py`: Main file for rag system without llamaindex
- `rag/test_chat_bot.py`: unit test chatbot
- `rag/helper`: helper function

## Run rag system:

- Install package

```
pip install -r requirements.txt
```

- Move to rag folder
- Run:

```
python -m chat_bot_llamaindex
```

Or

```
python -m chat_bot_manual_chunk_mode
```

## Run unit test

```
python -m unittest test_chat_bot.py
```
