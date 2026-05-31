with open("README.md", "r") as f:
    content = f.read()

import re
# Replace the testing memory section to match the requested format
new_testing_memory = """## Testing Memory

1. python main.py --session grader_test
2. Ask: "Show me 3 examples from the REFUND category"
3. Ask: "Show me 3 more"  (must use context from previous turn)
4. Ask: "How many complaints did we get?"
5. Ask: "What about refunds?"  (must resolve "refunds" from history)
6. Exit and restart: python main.py --session grader_test
7. Ask: "What did we just discuss?"  (must restore conversation) """

content = re.sub(r'## Testing Memory\n.*', new_testing_memory, content, flags=re.DOTALL)

with open("README.md", "w") as f:
    f.write(content)
