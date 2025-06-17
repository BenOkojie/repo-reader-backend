# gemini.py

import os
import google.generativeai as genai
from dotenv import load_dotenv
from google import genai
from google.genai import types
load_dotenv()
# genai.configure(api_key=)

def embed_chunk_with_gemini(text: str) -> list[float]:
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    try:
        result = client.models.embed_content(
            model="gemini-embedding-exp-03-07",
            contents=text,  # ✅ Use the actual text chunk!
            config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")
        )
        # print(result.values)
        # ✅ Extract the embedding vector (list of floats)
        # embedding = result.embedding.values  # or whatever field holds the float list
        
        # print("Generated embedding type:", type(embedding))
        # print("Generated embedding preview:", embedding[:5])

        return result.embeddings[0].values

    except Exception as e:
        print("❌ Error generating embedding:", e)
        return []


    # try:
    #     response = genai.embed_content(
    #         model="gemini-embedding-exp-03-07",
    #         content=text,
    #         config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
    #     )
    #     return response["embedding"]
    # except Exception as e:
    #     print("❌ Error generating embedding:", e)
    #     return []
