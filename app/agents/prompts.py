# prompts.py
# keeping all prompts here so they're easy to find and tweak without digging through agent code

PLANNER_PROMPT = """
You are a routing agent. Look at the input below and figure out what the user wants to do.

If it's not obvious what they want, or if it could mean multiple things, don't guess.
Set `is_ambiguous` to True and ask them a follow-up in `clarification_question`.

When asking for clarification, look at what kind of content it is (code, resume, meeting notes, invoice, etc.)
and ask something relevant, like:
"I detected meeting notes in the upload. Would you like:
- a summary
- action item extraction
- sentiment analysis?"

Some examples of clear intent - route these directly WITHOUT asking for clarification:
- "Summarize this audio transcript." -> 'summarization'
- "What is the sentiment of this text?" -> 'sentiment_analysis'
- "Explain this code snippet.", "an explanation of the query", "explain it" -> 'code_explanation'
- "What does the document say about project timelines?" -> 'rag_qa'
- "What are the action items in this meeting notes PDF?" -> 'extract_action_items'
- "a summary", "summarize it", "give me a summary" -> 'summarization'
- "sentiment analysis", "analyze the sentiment" -> 'sentiment_analysis'
- "How does gravity work?", "What is langgraph?", "Tell me about X", "What is X?", "Give me info on X" -> 'conversational'
- Any short follow-up reply like "a summary", "explain it", "sentiment analysis" -> match to closest task above
- Hi, hello, greetings -> 'conversational'

Key rule: if the input is a plain question, a general knowledge request ("tell me about...", "what is...", "explain..."), or a short follow-up reply, route it to 'conversational'. Do NOT flag these as ambiguous.

Input:
{content}
"""

# summarization output format is fixed per requirements
SUMMARIZATION_PROMPT = """
Summarize the content below. Output format:
- 1-line summary at the top
- 3 bullet points covering the key details
- a 5-sentence summary paragraph at the end

Content:
{content}
"""

SENTIMENT_ANALYSIS_PROMPT = """
Analyze the sentiment of the text below.
Return:
Label: positive / negative / neutral
Confidence: high / medium / low
Reason: one sentence explaining why

Content:
{content}
"""

CODE_EXPLANATION_PROMPT = """
Look at the code below and break it down.
Cover:
- what language it is
- what it does
- any obvious bugs or issues
- rough time complexity if relevant

Code:
{content}
"""

EXTRACT_ACTION_ITEMS_PROMPT = """
Pull out any action items from the content below.
List them as bullet points. If there aren't any, just say so.

Content:
{content}
"""

CONVERSATIONAL_PROMPT = """
You're a helpful AI assistant. Respond naturally to whatever the user said below.

Input:
{content}
"""

RAG_QA_PROMPT = """
Answer the user's question using only the context provided below.
If the answer isn't in the context, say you couldn't find it in the uploaded documents.

Context:
{context}

Question:
{query}
"""
