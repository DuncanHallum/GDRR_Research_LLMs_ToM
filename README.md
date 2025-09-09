# LLMs and Theory of Mind (ToM) Research Project

Research Project on LLMs and Theory of Mind for the University of Exeter, through the Google DeepMind Research Ready programme.

## User instructions

1. Create and activate virtual environment
```sh
python -m venv venv
.\venv\Scripts\Activate
```

2. Install requirements
```sh
pip install -r requirements.txt  
```

3. Add openai API key to .env
```sh
OPENAI_API_KEY="your_api_key"
```

5. Setup tests
```sh
python testing/gen_belief_dist.py
python testing/gen_messages_user_recognition.py
```

6. Run tests - jupyter notebooks in analysis folder

7. (Optional) Run system
```sh
python model/main.py
```
