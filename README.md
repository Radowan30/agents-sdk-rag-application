# agents-sdk-rag-application

## Challenges faced - 
1) Concurrently hadling file uploads and user messages, solution was using asyncio queues.
2) Excel file format not supported by the file search tool. Tried to convert excel file to markdown, but the results were still unsatisfactory. 

### Conclusion: Better to implement your own RAG pipeline with a newer type of RAG such as LightRAG
